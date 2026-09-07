"""Independent evaluator for the post-training model-convergence protocol.

This module is deliberately data-only: it reads JSON manifests, result records and
stored prediction records.  It never imports an adapter, optional detection
package, dataset, checkpoint, or model.  A result is useful only when all of the
frozen matrix and sealing invariants can be demonstrated from the saved evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EVALUATOR_VERSION = "POST-TRAINING-MODEL-CONVERGENCE-EVALUATOR 1.0.0"
PROTOCOL_VERSION = "post-training-model-convergence-1.0.0"
WORKLOADS = ("cifar10_resnet18", "cifar10_resnet50", "voc_yolo11n")
CLASSIFICATION_WORKLOADS = WORKLOADS[:2]
DETECTION_WORKLOAD = WORKLOADS[2]
BASE_SEEDS = (501, 502, 503)
SWARM_SEEDS = (601, 602, 603)
SPLIT_SEED = 20260908
PROJECTION_SEED = 20260909
BOOTSTRAP_SEED = 20260910
PARTICLES = 12
PRIMARY_GENERATIONS = 60
ENSEMBLE_GENERATIONS = 20
PRIMARY_QUERIES = PARTICLES * PRIMARY_GENERATIONS
ENSEMBLE_QUERIES = PARTICLES * ENSEMBLE_GENERATIONS
PRIMARY_RUNS = len(WORKLOADS) * len(BASE_SEEDS) * len(SWARM_SEEDS)
ENSEMBLE_RUNS = len(WORKLOADS) * len(SWARM_SEEDS)
TOTAL_PSO_QUERIES = PRIMARY_RUNS * PRIMARY_QUERIES + ENSEMBLE_RUNS * ENSEMBLE_QUERIES
OBJECTIVE_SAMPLES = {"cifar10_resnet18": 1024, "cifar10_resnet50": 1024, "voc_yolo11n": 512}
TOTAL_CANDIDATE_SAMPLES = sum(
    (len(BASE_SEEDS) * len(SWARM_SEEDS) * PRIMARY_QUERIES + len(SWARM_SEEDS) * ENSEMBLE_QUERIES)
    * OBJECTIVE_SAMPLES[w] for w in WORKLOADS
)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05 / 6.0

ISSUE_CATEGORIES = (
    "schema", "provenance", "matrix", "accounting", "seal", "selection",
    "finite", "metrics", "plateau", "overfit", "leakage", "bootstrap", "gates",
)


def _issue(issues: dict[str, list[str]], category: str, message: str) -> None:
    issues.setdefault(category, []).append(message)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _walk_nonfinite(value: Any, path: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        out.append(path or "$")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            out.extend(_walk_nonfinite(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            out.extend(_walk_nonfinite(item, f"{path}[{i}]"))
    return out


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _number(value: Any, *keys: str) -> float | None:
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if _finite(candidate):
                return float(candidate)
        for candidate in value.values():
            found = _number(candidate, *keys)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for candidate in value:
            found = _number(candidate, *keys)
            if found is not None:
                return found
    return None


def _int(value: Any, *keys: str) -> int | None:
    number = _number(value, *keys)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _same(a: Any, b: Any, tol: float = 1e-9) -> bool:
    return _finite(a) and _finite(b) and math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _resolve_json(root: Path, value: Any) -> Any:
    """Resolve a saved JSON prediction reference without opening model/data files."""
    if isinstance(value, Mapping):
        for key in ("path", "file", "artifact", "prediction_artifact", "predictions_path"):
            ref = value.get(key)
            if isinstance(ref, str) and ref.lower().endswith((".json", ".jsonl", ".pt")):
                return _resolve_json(root, ref)
        return value
    if isinstance(value, str) and value.lower().endswith((".json", ".jsonl", ".pt")):
        path = (root / value).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"prediction artifact escapes run root: {value}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".jsonl":
            return [_json_line for _json_line in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) if _json_line]
        if path.suffix == ".pt":
            try:
                import torch
                return torch.load(path, map_location="cpu", weights_only=True)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(f"unable to load weights-only prediction artifact: {path}: {exc}") from exc
        return _json(path)
    return value


def _hash_manifest(root: Path, manifest: Mapping[str, Any], issues: dict[str, list[str]]) -> bool:
    good = True
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        _issue(issues, "provenance", f"frozen manifest protocol mismatch: {manifest.get('protocol_version')!r}")
        good = False
    if manifest.get("state") != "frozen":
        _issue(issues, "seal", f"frozen manifest state must be 'frozen', got {manifest.get('state')!r}")
        good = False
    declared = manifest.get("manifest_hash")
    payload = {key: manifest[key] for key in ("protocol_version", "config", "artifacts", "state") if key in manifest}
    if not isinstance(declared, str) or hashlib.sha256(_canonical(payload)).hexdigest() != declared:
        _issue(issues, "seal", "frozen_manifest.json self-hash mismatch")
        good = False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        _issue(issues, "schema", "frozen manifest requires a non-empty artifacts map")
        return False
    for name, expected in artifacts.items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            _issue(issues, "schema", f"invalid frozen artifact hash declaration: {name!r}")
            good = False
            continue
        path = (root / name).resolve()
        if root.resolve() not in path.parents:
            _issue(issues, "seal", f"frozen artifact escapes run root: {name}")
            good = False
        elif not path.is_file():
            _issue(issues, "seal", f"frozen artifact is missing: {name}")
            good = False
        elif _sha256(path) != expected:
            _issue(issues, "seal", f"frozen artifact hash drift: {name}")
            good = False
    return good


def _config_checks(config: Any, issues: dict[str, list[str]], workload_id: str | None = None) -> None:
    if not isinstance(config, Mapping):
        _issue(issues, "schema", "missing or non-object config")
        return
    expected: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION, "split_seed": SPLIT_SEED,
        "projection_seed": PROJECTION_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "particle_count": PARTICLES, "pso_generations": PRIMARY_GENERATIONS,
        "residual_dimension": 64, "residual_bound": 1.0, "initial_radius": 0.25,
        "objective_checkpoints": [0, 10, 20, 30, 40, 50, 60],
        "base_seeds": list(BASE_SEEDS), "swarm_seeds": list(SWARM_SEEDS),
    }
    for key, value in expected.items():
        got = config.get(key)
        if got != value and not (isinstance(got, tuple) and list(got) == value):
            _issue(issues, "provenance", f"config.{key} must be {value!r}, got {got!r}")
    ids = config.get("workload_ids")
    if ids is not None and sorted(ids) != sorted(WORKLOADS):
        _issue(issues, "matrix", f"config.workload_ids must be {list(WORKLOADS)!r}")
    if workload_id and config.get("workload_id") not in (None, workload_id):
        _issue(issues, "matrix", f"config.workload_id disagrees with {workload_id}")


def _counter(value: Any, keys: Sequence[str]) -> int | None:
    if not isinstance(value, Mapping):
        return None
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _check_leakage(result: Mapping[str, Any], issues: dict[str, list[str]], workload: str) -> None:
    leakage = result.get("leakage_counters")
    if not isinstance(leakage, Mapping):
        _issue(issues, "schema", f"{workload}: missing leakage_counters")
        leakage = {}
    loaded_before = leakage.get("official_test_data_loaded_before_freeze", leakage.get("test_data_loaded_before_freeze"))
    if loaded_before is not False:
        _issue(issues, "leakage", f"{workload}: official test data must be explicitly marked not loaded before freeze")
    evaluated_before = leakage.get("official_test_evaluations_before_freeze", leakage.get("test_evaluations_before_freeze", leakage.get("official_test_forward_passes_before_freeze")))
    if evaluated_before != 0:
        _issue(issues, "leakage", f"{workload}: official test exposure before freeze must be explicitly zero")
    for key in ("official_test_data_loaded_before_freeze", "official_test_evaluations_before_freeze"):
        if key in result and ((key.endswith("freeze") and result[key] not in (False, 0))):
            _issue(issues, "leakage", f"{workload}: contradictory top-level {key}")
    construction = _counter(leakage, ("official_test_construction", "official_test_dataset_construction"))
    if construction != 1:
        _issue(issues, "leakage", f"{workload}: official test construction must equal one, got {construction}")
    forwards = _counter(leakage, ("official_test_forward_passes", "official_test_evaluations"))
    if forwards is None or forwards < 1:
        _issue(issues, "leakage", f"{workload}: official test forward/evaluation ledger must be positive, got {forwards}")
    confirmation = result.get("confirmation")
    if not isinstance(confirmation, Mapping):
        _issue(issues, "schema", f"{workload}: missing confirmation record")
        return
    for key in ("second_confirmation", "confirmation_repeated", "post_test_tuning", "post_test_reruns"):
        if confirmation.get(key) not in (None, False, 0, []):
            _issue(issues, "leakage", f"{workload}: forbidden repeated confirmation/tuning flag {key}")
    for key in ("official_test_data_loaded_before_freeze", "official_test_evaluations_before_freeze"):
        if key in confirmation and confirmation[key] not in (False, 0):
            _issue(issues, "leakage", f"{workload}: confirmation contradicts sealed pre-freeze {key}")


def _record_queries(record: Mapping[str, Any]) -> tuple[int | None, int | None]:
    counters = record.get("counters") if isinstance(record.get("counters"), Mapping) else record
    queries = _int(counters, "objective_queries", "queries", "query_count", "total_queries", "evaluated_queries")
    samples = _int(counters, "objective_samples", "samples", "sample_evaluations", "candidate_sample_evaluations", "total_sample_evaluations")
    return queries, samples


def _seed_from_key(key: Any) -> int | None:
    try:
        text = str(key)
        if text.isdigit():
            return int(text)
    except Exception:
        pass
    return None


def _collect_cells(node: Any, base: int | None = None, swarm: int | None = None) -> list[dict[str, Any]]:
    """Collect cells from either explicit records or base->swarm mappings."""
    found: list[dict[str, Any]] = []
    if isinstance(node, Mapping):
        b = _int(node, "base_seed") if _int(node, "base_seed") is not None else base
        s = _int(node, "swarm_seed") if _int(node, "swarm_seed") is not None else swarm
        if s is None:
            s = _int(node, "seed")
        if b is not None and s is not None and any(k in node for k in ("counters", "objective_queries", "objective_samples", "queries", "samples", "generation", "endpoint", "metrics")):
            item = dict(node); item.setdefault("base_seed", b); item.setdefault("swarm_seed", s); found.append(item)
        for key, value in node.items():
            key_seed = _seed_from_key(key)
            if key_seed in BASE_SEEDS:
                found.extend(_collect_cells(value, key_seed, s))
            elif key_seed in SWARM_SEEDS:
                found.extend(_collect_cells(value, b, key_seed))
            elif key not in {"base_seed", "swarm_seed"}:
                found.extend(_collect_cells(value, b, s))
    elif isinstance(node, list):
        for value in node:
            found.extend(_collect_cells(value, base, swarm))
    return found


def _collect_base_cells(node: Any, base: int | None = None) -> list[dict[str, Any]]:
    """Collect one record per base seed from flattened or base->record schemas."""
    found: list[dict[str, Any]] = []
    if isinstance(node, Mapping):
        b = _int(node, "base_seed") if _int(node, "base_seed") is not None else base
        if b is not None and any(k in node for k in ("updates", "gradient_updates", "counters", "metrics", "checkpoint", "endpoint", "objective")):
            item = dict(node); item.setdefault("base_seed", b); found.append(item)
        for key, value in node.items():
            key_seed = _seed_from_key(key)
            if key_seed in BASE_SEEDS: found.extend(_collect_base_cells(value, key_seed))
            elif key not in {"base_seed", "swarm_seed"}: found.extend(_collect_base_cells(value, b))
    elif isinstance(node, list):
        for value in node: found.extend(_collect_base_cells(value, base))
    return found


def _method_base_cells(result: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    arms = result.get("arms")
    return _collect_base_cells(arms.get(method)) if isinstance(arms, Mapping) and method in arms else []


def _method_evidence(value: Any, method: str, family: str, root: Path) -> bool:
    """Require non-empty metric or prediction evidence below an exact method key."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized == method:
                if _prediction_records(item, root) is not None: return True
                if isinstance(item, Mapping) and (_number(item, "nll", "loss", "accuracy", "map50_95", "map50", "mAP50-95") is not None): return True
                if _method_evidence(item, method, family, root): return True
            if _method_evidence(item, method, family, root): return True
    elif isinstance(value, list):
        return bool(value) and any(_method_evidence(item, method, family, root) for item in value)
    return False


def _require_confirmation_methods(result: Mapping[str, Any], workload: str, family: str, root: Path, issues: dict[str, list[str]]) -> None:
    confirmation = result.get("confirmation")
    required = {"feature_pso", "feature_random", "feature_adam", "head_adam"}
    required |= ({"uniform", "uniform_temperature", "slsqp_weights", "ensemble_pso"} if family == "classification" else {"uniform_wbf", "ensemble_pso", "ensemble_random"})
    for method in sorted(required):
        if not _method_evidence(confirmation, method, family, root):
            _issue(issues, "metrics", f"{workload}: confirmation lacks predictions/metrics for required method {method}")

def _method_cells(result: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    arms = result.get("arms")
    if not isinstance(arms, Mapping):
        return []
    value = arms.get(method)
    return _collect_cells(value) if value is not None else []


def _verify_matrix(result: Mapping[str, Any], workload: str, issues: dict[str, list[str]]) -> dict[str, Any]:
    arms = result.get("arms")
    if not isinstance(arms, Mapping):
        _issue(issues, "schema", f"{workload}: arms must be an object")
        arms = {}
    required = {"feature_pso", "feature_random", "feature_adam", "head_adam"}
    missing = required - set(arms)
    if missing:
        _issue(issues, "matrix", f"{workload}: missing arms {sorted(missing)}")
    stats: dict[str, Any] = {"primary_queries": 0, "primary_random_queries": 0, "ensemble_queries": 0, "primary_samples": 0, "primary_random_samples": 0, "ensemble_samples": 0, "cells": {}}
    for method in ("feature_pso", "feature_random"):
        cells = _method_cells(result, method)
        stats["cells"][method] = len(cells)
        expected = len(BASE_SEEDS) * len(SWARM_SEEDS)
        if len(cells) != expected:
            _issue(issues, "matrix", f"{workload}: {method} requires {expected} base/swarm cells, got {len(cells)}")
        seen: set[tuple[int, int]] = set()
        for cell in cells:
            key = (_int(cell, "base_seed") or -1, _int(cell, "swarm_seed") or -1)
            if key in seen or key[0] not in BASE_SEEDS or key[1] not in SWARM_SEEDS:
                _issue(issues, "matrix", f"{workload}: invalid or duplicate {method} cell {key}")
            seen.add(key)
            queries, samples = _record_queries(cell)
            if queries != PRIMARY_QUERIES:
                _issue(issues, "accounting", f"{workload}: {method} {key} queries must be {PRIMARY_QUERIES}, got {queries}")
            if samples != PRIMARY_QUERIES * OBJECTIVE_SAMPLES[workload]:
                _issue(issues, "accounting", f"{workload}: {method} {key} samples must be {PRIMARY_QUERIES * OBJECTIVE_SAMPLES[workload]}, got {samples}")
            if method == "feature_pso":
                if queries is not None: stats["primary_queries"] += queries
                if samples is not None: stats["primary_samples"] += samples
            else:
                if queries is not None: stats["primary_random_queries"] += queries
                if samples is not None: stats["primary_random_samples"] += samples
    for method in ("feature_adam", "head_adam"):
        cells = _method_base_cells(result, method); stats["cells"][method] = len(cells)
        if len(cells) != len(BASE_SEEDS): _issue(issues, "matrix", f"{workload}: {method} requires exactly three base cells, got {len(cells)}")
        seen = set()
        for cell in cells:
            seed = _int(cell, "base_seed")
            if seed in seen or seed not in BASE_SEEDS: _issue(issues, "matrix", f"{workload}: invalid or duplicate {method} base cell {seed}")
            if seed is not None: seen.add(seed)

    ensemble = result.get("ensemble")
    if not isinstance(ensemble, Mapping):
        _issue(issues, "schema", f"{workload}: ensemble must be an object")
        ensemble = {}
    required_ensemble = ("uniform", "uniform_temperature", "slsqp_weights", "ensemble_pso") if workload in CLASSIFICATION_WORKLOADS else ("uniform_wbf", "ensemble_pso", "ensemble_random")
    for method in required_ensemble:
        if method not in ensemble or ensemble.get(method) in (None, {}, []): _issue(issues, "matrix", f"{workload}: missing required ensemble method {method}")
    ens_pso = ensemble.get("ensemble_pso", ensemble.get("pso"))
    cells = _collect_cells(ens_pso) if ens_pso is not None else []
    stats["cells"]["ensemble_pso"] = len(cells)
    if len(cells) != len(SWARM_SEEDS):
        _issue(issues, "matrix", f"{workload}: ensemble_pso requires three swarm cells, got {len(cells)}")
    seen_swarm: set[int] = set()
    for cell in cells:
        seed = _int(cell, "swarm_seed", "seed")
        if seed in seen_swarm or seed not in SWARM_SEEDS:
            _issue(issues, "matrix", f"{workload}: invalid or duplicate ensemble swarm cell {seed}")
        if seed is not None: seen_swarm.add(seed)
        queries, samples = _record_queries(cell)
        if queries != ENSEMBLE_QUERIES:
            _issue(issues, "accounting", f"{workload}: ensemble_pso queries must be {ENSEMBLE_QUERIES}, got {queries}")
        if samples != ENSEMBLE_QUERIES * OBJECTIVE_SAMPLES[workload]:
            _issue(issues, "accounting", f"{workload}: ensemble_pso samples must be {ENSEMBLE_QUERIES * OBJECTIVE_SAMPLES[workload]}, got {samples}")
        if queries is not None: stats["ensemble_queries"] += queries
        if samples is not None: stats["ensemble_samples"] += samples
    if workload == DETECTION_WORKLOAD:
        random_cells = _collect_cells(ensemble.get("ensemble_random")) if ensemble.get("ensemble_random") is not None else []
        stats["cells"]["ensemble_random"] = len(random_cells)
        if len(random_cells) != len(SWARM_SEEDS): _issue(issues, "matrix", f"{workload}: ensemble_random requires three swarm cells, got {len(random_cells)}")
        seen_random = set()
        for cell in random_cells:
            seed = _int(cell, "swarm_seed", "seed")
            if seed in seen_random or seed not in SWARM_SEEDS: _issue(issues, "matrix", f"{workload}: invalid or duplicate ensemble_random seed {seed}")
            if seed is not None: seen_random.add(seed)
    return stats


def _verify_selection(result: Mapping[str, Any], workload: str, issues: dict[str, list[str]]) -> None:
    selection = result.get("development_selection")
    if not isinstance(selection, Mapping):
        _issue(issues, "selection", f"{workload}: missing development_selection")
        return
    selected = selection.get("primary", selection.get("feature_pso", selection.get("selected")))
    if not isinstance(selected, Mapping):
        _issue(issues, "selection", f"{workload}: missing primary selected endpoints")
        return
    for base in BASE_SEEDS:
        entry = selected.get(str(base), selected.get(base))
        if not isinstance(entry, Mapping):
            _issue(issues, "selection", f"{workload}: no selected endpoint for base seed {base}")
            continue
        swarm = _int(entry, "swarm_seed", "seed")
        generation = _int(entry, "generation", "final_generation")
        if swarm not in SWARM_SEEDS:
            _issue(issues, "selection", f"{workload}: selected seed {base} has invalid swarm {swarm}")
        if generation != PRIMARY_GENERATIONS:
            _issue(issues, "selection", f"{workload}: selected endpoint {base} is not final generation 60")
        matches = [c for c in _method_cells(result, "feature_pso") if _int(c, "base_seed") == base and _int(c, "swarm_seed") == swarm]
        if not matches:
            _issue(issues, "selection", f"{workload}: selected endpoint {base}/{swarm} is not a feature_pso cell")
        elif entry.get("vector_hash") and matches[0].get("vector_hash") and entry["vector_hash"] != matches[0]["vector_hash"]:
            _issue(issues, "selection", f"{workload}: selected vector hash drift for base {base}")


def _prediction_records(value: Any, root: Path) -> list[dict[str, Any]] | None:
    try: value = _resolve_json(root, value)
    except (OSError, ValueError, json.JSONDecodeError): return None
    if isinstance(value, Mapping):
        if "probabilities" in value and "targets" in value:
            probabilities, targets = value["probabilities"], value["targets"]
            for attr in ("detach", "cpu"):
                if hasattr(probabilities, attr): probabilities = getattr(probabilities, attr)()
                if hasattr(targets, attr): targets = getattr(targets, attr)()
            if hasattr(probabilities, "tolist"): probabilities = probabilities.tolist()
            if hasattr(targets, "tolist"): targets = targets.tolist()
            if isinstance(probabilities, (list, tuple)) and isinstance(targets, (list, tuple)) and len(probabilities) == len(targets):
                return [{"probabilities": list(probability), "target": int(target)} for probability, target in zip(probabilities, targets)]
            return None
        for key in ("records", "predictions", "images", "examples", "data"):
            if key in value:
                got = _prediction_records(value[key], root)
                if got is not None: return got
        return None
    if isinstance(value, list) and all(isinstance(x, Mapping) for x in value):
        return [dict(x) for x in value]
    return None


def _find_prediction_sets(value: Any, root: Path, prefix: str = "$") -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}"
            if "prediction" in str(key).lower() or str(key).lower() in {"base", "pso", "feature_pso", "test"}:
                records = _prediction_records(item, root)
                if records is not None: found[name] = records
            found.update(_find_prediction_sets(item, root, name))
    elif isinstance(value, list) and value and isinstance(value[0], Mapping):
        records = _prediction_records(value, root)
        if records is not None: found[prefix] = records
    return found


def classification_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute unrounded NLL and accuracy from per-image probabilities/targets."""
    probs: list[list[float]] = []; targets: list[int] = []
    for i, record in enumerate(records):
        p = record.get("probabilities", record.get("probs", record.get("prob")))
        target = record.get("target", record.get("label", record.get("class_id")))
        if not isinstance(p, (list, tuple)) or not p or not isinstance(target, int) or isinstance(target, bool) or target < 0 or target >= len(p):
            raise ValueError(f"invalid classification prediction at index {i}")
        vals = [float(x) for x in p]
        if not all(math.isfinite(x) and x >= 0.0 for x in vals) or not math.isclose(math.fsum(vals), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"probabilities must be finite and sum to one at index {i}")
        probs.append(vals); targets.append(target)
    nll = math.fsum(
        -math.log(max(p[t], 1e-300)) for p, t in zip(probs, targets)
    ) / len(probs)
    predictions = [
        max(range(len(p)), key=p.__getitem__) for p in probs
    ]
    accuracy = sum(
        prediction == target
        for prediction, target in zip(predictions, targets)
    ) / len(probs)
    brier = math.fsum(
        math.fsum(
            (probability - float(index == target)) ** 2
            for index, probability in enumerate(p)
        )
        for p, target in zip(probs, targets)
    ) / len(probs)
    confidences = [p[prediction] for p, prediction in zip(probs, predictions)]
    ece = 0.0
    for bin_index in range(15):
        lower = bin_index / 15.0
        upper = (bin_index + 1) / 15.0
        members = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence <= upper
            if bin_index == 14 or confidence < upper
        ]
        if members:
            bin_accuracy = math.fsum(
                predictions[index] == targets[index] for index in members
            ) / len(members)
            bin_confidence = math.fsum(
                confidences[index] for index in members
            ) / len(members)
            ece += (
                abs(bin_accuracy - bin_confidence)
                * len(members)
                / len(probs)
            )
    return {
        "n": len(probs),
        "nll": nll,
        "accuracy": accuracy,
        "brier": brier,
        "ece15": ece,
        "probabilities": probs,
        "targets": targets,
    }


def _box(record: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    value = record.get("box", record.get("bbox", record.get("xyxy")))
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_finite(x) for x in value): return None
    x1, y1, x2, y2 = map(float, value)
    return (x1, y1, x2, y2) if x2 >= x1 and y2 >= y1 else None


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter > 0 else 0.0


def _interp_ap_101(recall: Sequence[float], precision: Sequence[float]) -> float:
    """Ultralytics compute_ap: precision envelope and 101-point trapezoid."""
    mrec = [0.0, *map(float, recall), 1.0]
    mpre = [1.0, *map(float, precision), 0.0]
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    values: list[float] = []
    for k in range(101):
        x = k / 100.0; j = 0
        while j + 1 < len(mrec) and mrec[j + 1] <= x: j += 1
        if j + 1 >= len(mrec): values.append(mpre[-1]); continue
        span = mrec[j + 1] - mrec[j]
        values.append(mpre[j] if span <= 0 else mpre[j] + (mpre[j + 1] - mpre[j]) * (x - mrec[j]) / span)
    return sum((values[i] + values[i + 1]) * 0.5 / 100.0 for i in range(100))


def detection_metrics(records: Sequence[Mapping[str, Any]], class_count: int | None = None) -> dict[str, Any]:
    """Recompute Ultralytics-style whole-dataset AP at IoU .50:.95.

    Matching is performed independently per image.  Candidate matches are sorted
    by IoU and deduplicated by prediction and ground truth, as in
    DetectionValidator.process_batch; AP then uses the pinned 101-point
    interpolated trapezoid.
    """
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    parsed: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []; max_class = -1
    for i, image in enumerate(records):
        predictions = image.get("predictions", image.get("detections", image.get("pred", [])))
        truth = image.get("ground_truth", image.get("targets", image.get("gt", image.get("labels", []))))
        if not isinstance(predictions, list) or not isinstance(truth, list): raise ValueError(f"invalid detection image record {i}")
        pp: list[dict[str, Any]] = []; gg: list[dict[str, Any]] = []
        for item in predictions:
            if not isinstance(item, Mapping) or _box(item) is None: raise ValueError(f"invalid detection prediction {i}")
            cls = item.get("class_id", item.get("class", item.get("cls"))); score = item.get("score", item.get("confidence", item.get("conf")))
            if not isinstance(cls, int) or isinstance(cls, bool) or not _finite(score): raise ValueError(f"invalid detection prediction fields {i}")
            pp.append({"box": _box(item), "class_id": cls, "score": float(score)}); max_class = max(max_class, cls)
        for item in truth:
            if not isinstance(item, Mapping) or _box(item) is None: raise ValueError(f"invalid ground truth {i}")
            cls = item.get("class_id", item.get("class", item.get("cls")))
            if not isinstance(cls, int) or isinstance(cls, bool): raise ValueError(f"invalid ground truth class {i}")
            gg.append({"box": _box(item), "class_id": cls}); max_class = max(max_class, cls)
        parsed.append((pp, gg))
    present = sorted({g["class_id"] for _, gt in parsed for g in gt})
    classes = present if class_count is None else [c for c in range(class_count) if c in present]
    if not classes: classes = list(range(class_count or (max_class + 1))) or [0]
    aps: dict[str, list[float]] = {}; precision50: list[float] = []; recall50: list[float] = []
    for cls in classes:
        gt_count = sum(sum(x["class_id"] == cls for x in gt) for _, gt in parsed); class_aps: list[float] = []
        for threshold in thresholds:
            true_by_image: list[list[bool]] = []
            for preds, gt in parsed:
                candidates = []
                for pi, pred in enumerate(preds):
                    if pred["class_id"] != cls: continue
                    for gi, target in enumerate(gt):
                        if target["class_id"] == cls:
                            overlap = _iou(pred["box"], target["box"])
                            if overlap >= threshold: candidates.append((overlap, pi, gi))
                candidates.sort(key=lambda x: -x[0]); used_pred: set[int] = set(); used_gt: set[int] = set(); matched: set[int] = set()
                for overlap, pi, gi in candidates:
                    if pi not in used_pred and gi not in used_gt: used_pred.add(pi); used_gt.add(gi); matched.add(pi)
                true_by_image.append([i in matched for i in range(len(preds))])
            ranked = sorted(((pred["score"], hit) for (preds, _), hits in zip(parsed, true_by_image) for pred, hit in zip(preds, hits) if pred["class_id"] == cls), key=lambda x: -x[0])
            tp=[]; fp=[]; ctp=cfp=0
            for _, hit in ranked:
                ctp += int(hit); cfp += int(not hit); tp.append(ctp); fp.append(cfp)
            if gt_count == 0 or not ranked:
                # Ultralytics ap_per_class skips classes with no predictions; AP is zero.
                class_aps.append(0.0); continue
            recalls = [x / gt_count for x in tp]; precisions = [x / max(x + y, 1) for x, y in zip(tp, fp)]
            class_aps.append(_interp_ap_101(recalls, precisions))
            if threshold == 0.5:
                precision50.append(precisions[-1] if precisions else 0.0); recall50.append(recalls[-1] if recalls else 0.0)
        aps[str(cls)] = class_aps
    map50 = math.fsum(v[0] for v in aps.values()) / len(aps); map5095 = math.fsum(x for values in aps.values() for x in values) / (len(aps) * 10)
    return {"per_class_ap": aps, "n": len(records), "map50": map50, "map50_95": map5095,
            "precision": math.fsum(precision50) / len(precision50) if precision50 else 0.0,
            "recall": math.fsum(recall50) / len(recall50) if recall50 else 0.0,
            "ground_truth": sum(len(gt) for _, gt in parsed), "predictions": sum(len(preds) for preds, _ in parsed)}

def _metric_from_record(record: Any, family: str) -> dict[str, float] | None:
    if not isinstance(record, Mapping): return None
    keys = ("nll", "loss") if family == "classification" else ("map50_95", "map50-95", "mAP50-95", "map5095")
    primary = _number(record, *keys); accuracy = _number(record, "accuracy", "acc")
    map50 = _number(record, "map50", "mAP50")
    if family == "classification" and primary is not None and accuracy is not None: return {"nll": primary, "accuracy": accuracy}
    if family == "detection" and primary is not None: return {"map50_95": primary, **({"map50": map50} if map50 is not None else {})}
    return None


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered: raise ValueError("cannot quantile an empty sequence")
    position = (len(ordered) - 1) * q; lower = int(math.floor(position)); upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _record_identity(record: Mapping[str, Any], index: int) -> Any:
    for key in ("image_id", "id", "key", "filename", "path", "index"):
        if key in record: return (key, str(record[key]))
    return ("position", index)


def _record_ground_truth(record: Mapping[str, Any]) -> Any:
    return record.get("ground_truth", record.get("targets", record.get("gt", record.get("labels", []))))


def _bootstrap_alignment(pairs: Sequence[tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]], family: str) -> tuple[bool, str]:
    if not pairs or any(len(a) != len(b) or not a for a, b in pairs): return False, "incomplete or length-mismatched paired records"
    reference_ids = [_record_identity(x, i) for i, x in enumerate(pairs[0][0])]
    reference_gt = [_record_ground_truth(x) for x in pairs[0][0]]
    for pair_index, (base, pso) in enumerate(pairs):
        if [_record_identity(x, i) for i, x in enumerate(base)] != reference_ids or [_record_identity(x, i) for i, x in enumerate(pso)] != reference_ids:
            return False, f"pair {pair_index} image IDs/order differ"
        if family == "classification":
            base_targets = [x.get("target", x.get("label", x.get("class_id"))) for x in base]
            pso_targets = [x.get("target", x.get("label", x.get("class_id"))) for x in pso]
            ref_targets = [x.get("target", x.get("label", x.get("class_id"))) for x in pairs[0][0]]
            if base_targets != ref_targets or pso_targets != ref_targets: return False, f"pair {pair_index} targets differ"
        elif [_canonical(_record_ground_truth(x)) for x in base] != [_canonical(x) for x in reference_gt] or [_canonical(_record_ground_truth(x)) for x in pso] != [_canonical(x) for x in reference_gt]:
            return False, f"pair {pair_index} ground truth differs"
    return True, "aligned"


def _bootstrap_from_records(pairs: Sequence[tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]], family: str) -> dict[str, Any]:
    aligned, reason = _bootstrap_alignment(pairs, family)
    if not aligned: return {"available": False, "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES, "reason": reason}
    rng = random.Random(BOOTSTRAP_SEED); stats: list[float] = []
    if family == "classification":
        parsed = [(classification_metrics(a), classification_metrics(b)) for a, b in pairs]
        by_class: dict[int, list[int]] = {}
        for i, target in enumerate(parsed[0][0]["targets"]): by_class.setdefault(target, []).append(i)
        for _ in range(BOOTSTRAP_RESAMPLES):
            # One class-stratified draw is shared by every paired base/PSO model.
            indices = [rng.choice(class_indices) for class_indices in by_class.values() for _ in class_indices]
            deltas = []
            for base, pso in parsed:
                b_nll = math.fsum(-math.log(max(base["probabilities"][i][base["targets"][i]], 1e-300)) for i in indices) / len(indices)
                p_nll = math.fsum(-math.log(max(pso["probabilities"][i][pso["targets"][i]], 1e-300)) for i in indices) / len(indices)
                deltas.append((b_nll - p_nll) / b_nll if b_nll else 0.0)
            stats.append(math.fsum(deltas) / len(deltas))
    else:
        for _ in range(BOOTSTRAP_RESAMPLES):
            # One whole-image draw is shared by every paired base/PSO model.
            indices = [rng.randrange(len(pairs[0][0])) for _ in pairs[0][0]]
            deltas = []
            for base, pso in pairs:
                b = detection_metrics([base[i] for i in indices])["map50_95"]
                p = detection_metrics([pso[i] for i in indices])["map50_95"]
                deltas.append(p - b)
            stats.append(math.fsum(deltas) / len(deltas))
    lo = _quantile(stats, BOOTSTRAP_ALPHA); hi = _quantile(stats, 1.0 - BOOTSTRAP_ALPHA)
    return {"available": True, "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES, "alpha": BOOTSTRAP_ALPHA, "lower": lo, "upper": hi, "statistic": math.fsum(stats) / len(stats), "excludes_zero": lo > 0 or hi < 0}

def _prediction_pairs(result: Mapping[str, Any], root: Path, family: str) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]]:
    """Find one test base/selected pair per frozen base seed by explicit names."""
    sets = _find_prediction_sets(result.get("confirmation", {}), root)
    out: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for seed in BASE_SEEDS:
        candidates = [(name, records) for name, records in sets.items() if str(seed) in name]
        base = next((records for name, records in candidates if any(x in name.lower() for x in ("base", "frozen"))), None)
        pso = next((records for name, records in candidates if any(x in name.lower() for x in ("feature_pso", "selected", "pso")) and "ensemble" not in name.lower()), None)
        if base is not None and pso is not None: out.append((seed, base, pso))
    return out


def _audit_series(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for item in value.values(): found.extend(_audit_series(item))
    elif isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
        if any("epoch" in item or "step" in item for item in value) and any(_number(item, "loss", "audit_loss") is not None for item in value):
            found.append({"records": value})
        else:
            for item in value: found.extend(_audit_series(item))
    return found


def _plateau_flags(result: Mapping[str, Any], workload: str, family: str, issues: dict[str, list[str]]) -> dict[str, Any]:
    baselines = result.get("baselines", {})
    if not isinstance(baselines, Mapping): return {"available": False, "passed": False}
    per_seed: dict[str, bool] = {}; details: dict[str, Any] = {}
    for seed in BASE_SEEDS:
        entry = baselines.get(str(seed), baselines.get(seed))
        series = _audit_series(entry)
        records = series[0]["records"] if series else []
        records = records[-11:] if len(records) >= 11 else []
        losses = [_number(x, "loss", "audit_loss") for x in records]
        metric_keys = ("accuracy", "primary_metric", "selection_accuracy") if family == "classification" else ("map50_95", "mAP50-95", "primary_metric", "selection_metric")
        metrics = [_number(x, *metric_keys) for x in records]
        loss_ok = len(losses) == 11 and all(x is not None for x in losses)
        metric_ok = len(metrics) == 11 and all(x is not None for x in metrics)
        if loss_ok:
            mean = math.fsum(float(x) for x in losses) / len(losses)
            loss_ok = (max(losses) - min(losses)) / max(abs(mean), 1e-12) <= 0.01
        if metric_ok: metric_ok = max(metrics) - min(metrics) <= 0.005
        passed = bool(loss_ok and metric_ok); per_seed[str(seed)] = passed
        details[str(seed)] = {"loss_ok": bool(loss_ok), "metric_ok": bool(metric_ok), "observations": len(records)}
        declared = _number(entry, "baseline_plateau") if isinstance(entry, Mapping) else None
        if isinstance(entry, Mapping) and "baseline_plateau" in entry and bool(entry["baseline_plateau"]) != passed:
            _issue(issues, "plateau", f"{workload}: baseline seed {seed} inflated/incorrect plateau flag")
    return {"available": bool(per_seed), "per_seed": per_seed, "details": details, "passed": bool(per_seed) and all(per_seed.values())}


def _workload_gates(result: Mapping[str, Any], workload: str, family: str, root: Path, issues: dict[str, list[str]]) -> dict[str, Any]:
    pairs = _prediction_pairs(result, root, family)
    per_seed: dict[str, Any] = {}
    pair_records: list[tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = []
    for seed, base_records, pso_records in pairs:
        try:
            base = classification_metrics(base_records) if family == "classification" else detection_metrics(base_records, 20)
            pso = classification_metrics(pso_records) if family == "classification" else detection_metrics(pso_records, 20)
        except (ValueError, ZeroDivisionError) as exc:
            _issue(issues, "metrics", f"{workload}: test pair {seed} cannot be recomputed: {exc}"); continue
        pair_records.append((base_records, pso_records))
        if family == "classification": per_seed[str(seed)] = {"base": {"nll": base["nll"], "accuracy": base["accuracy"]}, "pso": {"nll": pso["nll"], "accuracy": pso["accuracy"]}, "relative_nll_reduction": (base["nll"] - pso["nll"]) / base["nll"], "accuracy_delta": pso["accuracy"] - base["accuracy"]}
        else: per_seed[str(seed)] = {"base": {"map50_95": base["map50_95"], "map50": base["map50"]}, "pso": {"map50_95": pso["map50_95"], "map50": pso["map50"]}, "map50_95_delta": pso["map50_95"] - base["map50_95"], "map50_delta": pso["map50"] - base["map50"]}
    ci = _bootstrap_from_records(pair_records, family)
    # Confirmation predictions and the paired CI are required integrity evidence.
    # A numerically negative CI is a valid scientific result; an unavailable CI
    # means the frozen/confirmed artifact set is incomplete or tampered.
    if len(pairs) != len(BASE_SEEDS):
        _issue(issues, "metrics", f"{workload}: required confirmation pairs are incomplete ({len(pairs)}/{len(BASE_SEEDS)})")
    if not ci.get("available", False):
        _issue(issues, "metrics", f"{workload}: required paired bootstrap unavailable: {ci.get('reason', 'missing prediction evidence')}")
    if family == "classification" and per_seed:
        reductions = [x["relative_nll_reduction"] for x in per_seed.values()]; acc_deltas = [x["accuracy_delta"] for x in per_seed.values()]
        gate = len(reductions) == 3 and math.fsum(reductions) / 3 >= 0.01 and math.fsum(acc_deltas) / 3 >= -0.002 and min(acc_deltas) >= -0.005 and sum(x > 0 for x in reductions) >= 2 and ci.get("excludes_zero", False)
    elif family == "detection" and per_seed:
        deltas = [x["map50_95_delta"] for x in per_seed.values()]
        gate = len(deltas) == 3 and math.fsum(deltas) / 3 >= 0.005 and min(deltas) >= -0.005 and sum(x > 0 for x in deltas) >= 2 and ci.get("excludes_zero", False)
    else: gate = False
    if len(pairs) != len(BASE_SEEDS): _issue(issues, "bootstrap", f"{workload}: missing complete official-test base/feature_pso prediction pairs ({len(pairs)}/3)")
    return {"available": bool(per_seed), "per_seed": per_seed, "bootstrap": ci, "generalization_pass": bool(gate)}


def _verify_hashes(result: Mapping[str, Any], root: Path, issues: dict[str, list[str]], workload: str) -> None:
    declared = result.get("artifact_hashes")
    if not isinstance(declared, Mapping):
        _issue(issues, "schema", f"{workload}: missing artifact_hashes")
        return
    for name, expected in declared.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            _issue(issues, "seal", f"{workload}: malformed artifact hash entry")
            continue
        path = (root / name).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            _issue(issues, "seal", f"{workload}: missing/escaping artifact {name}")
        elif _sha256(path) != expected:
            _issue(issues, "seal", f"{workload}: artifact hash drift {name}")


def _compare_development_snapshot(development: Any, current: Mapping[str, Any], workload: str, issues: dict[str, list[str]]) -> None:
    """Ensure confirmation did not alter any sealed development decision/state."""
    if not isinstance(development, Mapping):
        _issue(issues, "schema", f"{workload}: development_result.json must contain an object")
        return
    development_leakage = development.get("leakage_counters")
    if not isinstance(development_leakage, Mapping):
        _issue(issues, "leakage", f"{workload}: development snapshot lacks leakage_counters")
    else:
        loaded = development_leakage.get("official_test_data_loaded_before_freeze", development_leakage.get("test_data_loaded_before_freeze"))
        evaluated = development_leakage.get("official_test_evaluations_before_freeze", development_leakage.get("test_evaluations_before_freeze", development_leakage.get("official_test_forward_passes_before_freeze")))
        if loaded is not False or evaluated != 0:
            _issue(issues, "leakage", f"{workload}: development snapshot records pre-freeze official-test exposure")
        for key in ("official_test_construction", "official_test_dataset_construction", "official_test_forward_passes", "official_test_evaluations"):
            if key in development_leakage and development_leakage[key] != 0:
                _issue(issues, "leakage", f"{workload}: development snapshot {key} must be zero")
    development_confirmation = development.get("confirmation")
    if development_confirmation not in ({}, None):
        _issue(issues, "leakage", f"{workload}: development snapshot confirmation must be empty")
    fields = ("config", "manifests", "provenance", "baselines", "arms", "ensemble", "development_selection", "integrity", "resource_ledger", "artifact_hashes")
    for field in fields:
        if field not in development:
            _issue(issues, "seal", f"{workload}: development_result.json missing sealed field {field}")
            continue
        if field not in current:
            _issue(issues, "seal", f"{workload}: current result missing sealed field {field}")
            continue
        if field == "artifact_hashes":
            # Confirmation may add test prediction hashes. Every development hash
            # must nevertheless remain present and byte-identical.
            old_hashes = development[field]; new_hashes = current[field]
            if not isinstance(old_hashes, Mapping) or not isinstance(new_hashes, Mapping):
                _issue(issues, "seal", f"{workload}: artifact_hashes changed shape across confirmation")
            else:
                for name, value in old_hashes.items():
                    if new_hashes.get(name) != value: _issue(issues, "seal", f"{workload}: sealed artifact hash drift for {name}")
        elif field == "integrity":
            old_integrity = development[field]; new_integrity = current[field]
            if not isinstance(old_integrity, Mapping) or not isinstance(new_integrity, Mapping):
                if old_integrity != new_integrity: _issue(issues, "seal", f"{workload}: pre-confirmation integrity changed")
            else:
                for name, value in old_integrity.items():
                    if new_integrity.get(name) != value: _issue(issues, "seal", f"{workload}: pre-confirmation integrity field changed: {name}")
        elif _canonical(development[field]) != _canonical(current[field]):
            _issue(issues, "seal", f"{workload}: sealed pre-confirmation field changed: {field}")

def _evaluate_workload(result: Any, root: Path, workload: str, issues: dict[str, list[str]], development: Any = None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        _issue(issues, "schema", f"{workload}: result must be an object"); return {"workload_id": workload, "valid": False}
    if result.get("workload_id") != workload: _issue(issues, "matrix", f"{workload}: workload_id mismatch")
    if development is not None: _compare_development_snapshot(development, result, workload, issues)
    family = "detection" if workload == DETECTION_WORKLOAD else "classification"
    if result.get("family") != family: _issue(issues, "matrix", f"{workload}: family must be {family}")
    for key in ("manifests", "provenance", "baselines", "arms", "ensemble", "development_selection", "confirmation", "integrity", "leakage_counters", "resource_ledger", "artifact_hashes"):
        if key not in result: _issue(issues, "schema", f"{workload}: missing top-level {key}")
    _config_checks(result.get("config"), issues, workload); _check_leakage(result, issues, workload); _verify_hashes(result, root, issues, workload)
    accounting = _verify_matrix(result, workload, issues); _verify_selection(result, workload, issues); _require_confirmation_methods(result, workload, family, root, issues)
    if result.get("integrity", {}).get("confirmed") is False if isinstance(result.get("integrity"), Mapping) else False:
        _issue(issues, "seal", f"{workload}: integrity declares confirmation failure")
    prediction_sets = _find_prediction_sets(result.get("confirmation", {}), root)
    recomputed: dict[str, Any] = {}
    for name, records in prediction_sets.items():
        try: recomputed[name] = classification_metrics(records) if family == "classification" else detection_metrics(records, 20)
        except (ValueError, ZeroDivisionError) as exc: _issue(issues, "metrics", f"{workload}: invalid stored predictions at {name}: {exc}")
    # Check every explicitly stored metric that has a corresponding recomputation.
    for name, metric in recomputed.items():
        if "pso" in name.lower() and isinstance(metric, Mapping):
            stored = _metric_from_record(result.get("confirmation"), family)
            if stored and family == "classification" and not (_same(stored.get("nll"), metric.get("nll"), 1e-7) and _same(stored.get("accuracy"), metric.get("accuracy"), 1e-7)):
                _issue(issues, "metrics", f"{workload}: stored classification metric disagrees with probabilities at {name}")
    plateau = _plateau_flags(result, workload, family, issues)
    gates = _workload_gates(result, workload, family, root, issues)
    objective = _number(result.get("development_selection"), "objective_improvement", "relative_objective_improvement")
    selection_metric = _number(result.get("development_selection"), "selection_metric", "selection_nll", "selection_map50_95")
    overfit = bool(objective is not None and objective >= 0.01 and selection_metric is not None and ((family == "classification" and selection_metric > 0.01) or (family == "detection" and selection_metric < -0.005)))
    if overfit: _issue(issues, "overfit", f"{workload}: objective improvement conflicts with held-out/selection metric")
    return {"workload_id": workload, "family": family, "valid": True, "accounting": accounting, "recomputed": recomputed, "plateau": plateau, "gates": gates, "overfit_signal": overfit}


def evaluate_run(run_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Evaluate one frozen run, returning findings even when artifacts are malformed."""
    root = Path(run_root); issues = {key: [] for key in ISSUE_CATEGORIES}; workloads: dict[str, Any] = {}
    manifest: Any = None
    try: manifest = _json(root / "frozen_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc: _issue(issues, "schema", f"cannot load frozen_manifest.json: {exc}")
    if isinstance(manifest, Mapping):
        for location in _walk_nonfinite(manifest): _issue(issues, "finite", f"non-finite value in frozen manifest at {location}")
        _hash_manifest(root, manifest, issues); _config_checks(manifest.get("config"), issues)
        frozen_config = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
        if sorted(frozen_config.get("workload_ids", ())) != sorted(WORKLOADS): _issue(issues, "matrix", "frozen manifest does not seal all three workloads")
    for workload in WORKLOADS:
        path = root / "workloads" / workload / "result.json"
        development_path = root / "workloads" / workload / "development_result.json"
        if isinstance(manifest, Mapping):
            manifest_artifacts = manifest.get("artifacts", {})
            expected_development = f"workloads/{workload}/development_result.json"
            if not isinstance(manifest_artifacts, Mapping) or expected_development not in manifest_artifacts:
                _issue(issues, "seal", f"{workload}: frozen manifest must seal {expected_development}")
        try: result = _json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _issue(issues, "schema", f"{workload}: cannot load result.json: {exc}"); continue
        try: development = _json(development_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _issue(issues, "seal", f"{workload}: cannot load development_result.json: {exc}"); development = None
        for location in _walk_nonfinite(result): _issue(issues, "finite", f"{workload}: non-finite value at {location}")
        if development is not None:
            for location in _walk_nonfinite(development): _issue(issues, "finite", f"{workload}: non-finite development value at {location}")
        workloads[workload] = _evaluate_workload(result, root, workload, issues, development)
    # Cross-workload exact accounting is intentionally independent of stored totals.
    totals = {key: sum(int(w.get("accounting", {}).get(key, 0)) for w in workloads.values()) for key in ("primary_queries", "primary_random_queries", "ensemble_queries", "primary_samples", "primary_random_samples", "ensemble_samples")}
    totals["pso_queries"] = totals["primary_queries"] + totals["ensemble_queries"]
    totals["candidate_samples"] = totals["primary_samples"] + totals["ensemble_samples"]
    if totals["pso_queries"] != TOTAL_PSO_QUERIES: _issue(issues, "accounting", f"total PSO queries must be {TOTAL_PSO_QUERIES}, got {totals['pso_queries']}")
    if totals["candidate_samples"] != TOTAL_CANDIDATE_SAMPLES: _issue(issues, "accounting", f"total candidate-sample evaluations must be {TOTAL_CANDIDATE_SAMPLES}, got {totals['candidate_samples']}")
    # A success flag is never consumed; it is checked against independently observed integrity.
    integrity_ok = not any(issues[key] for key in ("schema", "provenance", "matrix", "accounting", "seal", "selection", "finite", "leakage", "metrics"))
    payload = {"evaluator_version": EVALUATOR_VERSION, "protocol_version": PROTOCOL_VERSION, "run_root": str(root), "pass": integrity_ok, "integrity_pass": integrity_ok, "workloads": workloads, "accounting": {**totals, "expected_pso_queries": TOTAL_PSO_QUERIES, "expected_candidate_samples": TOTAL_CANDIDATE_SAMPLES}, "issues": issues, "issue_counts": {key: len(value) for key, value in issues.items()}}
    return payload


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen post-training model-convergence artifacts")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv); payload = evaluate_run(args.run_root)
    destination = args.output or args.run_root / "evaluation.json"
    try: _atomic_json(destination, payload)
    except OSError as exc:
        print(f"evaluator output failed: {exc}", file=sys.stderr); return 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload["pass"] else 1


__all__ = [
    "BASE_SEEDS", "BOOTSTRAP_ALPHA", "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "CLASSIFICATION_WORKLOADS",
    "DETECTION_WORKLOAD", "EVALUATOR_VERSION", "ENSEMBLE_QUERIES", "TOTAL_CANDIDATE_SAMPLES", "TOTAL_PSO_QUERIES",
    "WORKLOADS", "classification_metrics", "detection_metrics", "evaluate_run", "main", "build_cli_parser",
]

if __name__ == "__main__":
    raise SystemExit(main())
