"""Behavioral tests for the post-training convergence protocol.

These tests intentionally use tiny local modules and synthetic artifacts.  They
exercise protocol boundaries (rather than implementation details) while
keeping the production data/model paths completely offline.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import post_training_model_convergence as study


class TinyStatefulModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Linear(3, 2, bias=False)
        self.bn = nn.BatchNorm1d(2)
        self.head = nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(self.feature(value)))


def _tiny_model() -> TinyStatefulModel:
    torch.manual_seed(17)
    model = TinyStatefulModel()
    model.train()
    return model


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", "post-training-model-convergence-drift"),
        ("split_seed", study.SPLIT_SEED + 1),
        ("base_seeds", (501, 502, 504)),
        ("swarm_seeds", (601, 602, 604)),
        ("projection_seed", study.PROJECTION_SEED + 1),
        ("bootstrap_seed", study.BOOTSTRAP_SEED + 1),
        ("particle_count", study.PARTICLE_COUNT - 1),
        ("pso_generations", study.PSO_GENERATIONS - 1),
        ("residual_dimension", study.RESIDUAL_DIMENSION - 1),
        ("residual_bound", study.RESIDUAL_BOUND / 2),
        ("initial_radius", study.INITIAL_RADIUS / 2),
        ("objective_checkpoints", (0, 1)),
    ],
)
def test_study_config_rejects_protocol_constant_drift(field: str, value: object) -> None:
    with pytest.raises(study.ProtocolError):
        study.StudyConfig(**{field: value})


def test_study_config_round_trip_and_matrix_order_boundary(tmp_path: Path) -> None:
    config = study.StudyConfig()
    assert study.StudyConfig.from_dict(config.to_dict()) == config
    assert config.base_seeds == (501, 502, 503)
    assert config.swarm_seeds == (601, 602, 603)
    assert config.objective_checkpoints == (0, 10, 20, 30, 40, 50, 60)
    with pytest.raises(study.ProtocolError, match="fixed order"):
        study._make_adapters(
            study.StudyConfig(workload_ids=(study.DEFAULT_WORKLOAD_IDS[0],)),
            tmp_path / "run",
            tmp_path / "data",
            False,
        )

def test_selected_codec_is_deterministic_and_preserves_nonselected_state() -> None:
    model_a = _tiny_model()
    model_b = copy.deepcopy(model_a)
    names = ("feature.weight",)
    codec_a = study.SelectedResidualCodec(model_a, names, projection_seed=12345)
    codec_b = study.SelectedResidualCodec(model_b, names, projection_seed=12345)
    residual = torch.linspace(-0.75, 0.75, study.RESIDUAL_DIMENSION)

    assert codec_a.names == names
    assert torch.equal(codec_a.projection_indices, codec_b.projection_indices)
    assert torch.equal(codec_a.decode(residual), codec_b.decode(residual))
    assert torch.equal(codec_a.decode(torch.zeros(study.RESIDUAL_DIMENSION)), model_a.feature.weight.detach().flatten())
    assert codec_a.scales == codec_b.scales
    first_indices = codec_a.projection_indices
    second_indices = codec_a.projection_indices
    assert first_indices.data_ptr() != second_indices.data_ptr()

    before = {name: value.detach().clone() for name, value in model_a.named_parameters()}
    before_buffers = {name: value.detach().clone() for name, value in model_a.named_buffers()}
    original_modes = {name: child.training for name, child in model_a.named_modules()}
    with codec_a.applied(model_a, residual):
        assert not torch.equal(model_a.feature.weight.detach(), before["feature.weight"])
        assert torch.equal(model_a.head.weight.detach(), before["head.weight"])
        assert torch.equal(model_a.bn.running_mean, before_buffers["bn.running_mean"])
        assert model_a.training is False
    assert {name: child.training for name, child in model_a.named_modules()} == original_modes
    for name, value in model_a.named_parameters():
        assert torch.equal(value, before[name])
    for name, value in model_a.named_buffers():
        assert torch.equal(value, before_buffers[name])


def test_selected_codec_restores_state_after_exception_and_rejects_nonselected_mutation() -> None:
    model = _tiny_model()
    codec = study.SelectedResidualCodec(model, ("feature.weight",))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with pytest.raises(RuntimeError, match="callback failure"):
        with codec.applied(model, codec.zero_residual()):
            model.bn.running_mean.add_(1.0)
            raise RuntimeError("callback failure")
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())

    with pytest.raises(study.ProtocolError, match="non-selected"):
        with codec.applied(model, torch.ones(study.RESIDUAL_DIMENSION)):
            with torch.no_grad():
                model.head.bias.add_(1.0)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_state_neutral_audit_restores_model_and_rng() -> None:
    model = _tiny_model()
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_modes = {name: child.training for name, child in model.named_modules()}
    before_rng = torch.get_rng_state().clone()

    def callback() -> float:
        model.eval()
        with torch.no_grad():
            model.feature.weight.add_(3.0)
            model.bn.running_var.mul_(2.0)
        torch.manual_seed(999)
        return 1.25

    assert study.run_state_neutral_audit(model, callback) == 1.25
    assert {name: child.training for name, child in model.named_modules()} == before_modes
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before_state.items())


def _objective(residual: torch.Tensor) -> study.ObjectiveResult:
    # Deliberately use every coordinate so a candidate is not a mock echo.
    return study.ObjectiveResult(
        loss=float(torch.sum(residual.square()).item()),
        samples=3,
        forward_passes=1,
        backward_passes=1,
    )


def test_pso_and_random_have_exact_equal_query_and_sample_budgets() -> None:
    validation_calls: list[torch.Tensor] = []

    def validation(residual: torch.Tensor) -> study.AuditResult:
        validation_calls.append(residual.detach().clone())
        return study.AuditResult(loss=float(residual.abs().mean()), samples=2)

    pso = study.run_residual_pso(_objective, seed=study.SWARM_SEEDS[0], validation=validation)
    random = study.run_equal_budget_random(_objective, seed=study.SWARM_SEEDS[0])

    for result in (pso, random):
        assert result.objective_queries == study.PARTICLE_COUNT * study.PSO_GENERATIONS == 720
        assert result.counters.objective_samples == 720 * 3
        assert result.counters.objective_forward_passes == 720
        assert result.counters.objective_backward_passes == 720
        assert result.counters.objective_failures == 0
        assert len(result.endpoints) == study.PSO_GENERATIONS
        assert len(result.trajectory) == study.PSO_GENERATIONS
        assert result.best_objective is not None
        assert result.best_residual is not None
        assert result.best_residual.shape == (study.RESIDUAL_DIMENSION,)

    assert len(validation_calls) == len(study.OBJECTIVE_CHECKPOINTS)
    assert pso.counters.validation_evaluations == len(study.OBJECTIVE_CHECKPOINTS)
    assert pso.counters.validation_samples == len(study.OBJECTIVE_CHECKPOINTS) * 2
    assert random.method == "feature_random"
    assert pso.method == "feature_pso"


def test_state_machine_and_confirmation_seal_boundaries(tmp_path: Path) -> None:
    config = study.StudyConfig()
    state = study.prepare_run(tmp_path, config)
    with pytest.raises(study.StateTransitionError):
        state.transition(study.StudyState.FROZEN)
    state.transition(study.StudyState.DEVELOPING)
    artifact = tmp_path / "evidence.json"
    study.atomic_write_json(artifact, {"metric": 1.0})

    manifest = study.freeze_run(tmp_path, config, ["evidence.json"], state)
    assert state.state is study.StudyState.FROZEN
    assert study.load_frozen_manifest(tmp_path).manifest_hash == manifest.manifest_hash
    with pytest.raises(study.StateTransitionError):
        state.transition(study.StudyState.DEVELOPING)

    study.begin_confirmation(tmp_path, state)
    assert state.state is study.StudyState.CONFIRMING
    study.finish_confirmation(state, success=True)
    assert state.state is study.StudyState.COMPLETED
    with pytest.raises(study.StateTransitionError):
        study.finish_confirmation(state, success=True)


def test_frozen_manifest_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    config = study.StudyConfig()
    state = study.prepare_run(tmp_path, config)
    state.transition(study.StudyState.DEVELOPING)
    artifact = tmp_path / "checkpoint.bin"
    artifact.write_bytes(b"original")
    study.freeze_run(tmp_path, config, [artifact.name], state)
    assert study.verify_frozen_manifest(tmp_path).artifacts[artifact.name]

    artifact.write_bytes(b"tampered")
    with pytest.raises(study.SealError, match="hash mismatch"):
        study.verify_frozen_manifest(tmp_path)


def _completed_record() -> dict[str, float]:
    return {"loss": 1.0, "queries": 1}


def _matrix_result(workload_id: str, artifact_name: str, artifact_hash: str) -> dict[str, object]:
    cell_tree = {
        str(base): {str(swarm): _completed_record() for swarm in study.SWARM_SEEDS}
        for base in study.BASE_SEEDS
    }
    return {
        "workload_id": workload_id,
        "family": "classification",
        "config": {},
        "manifests": {},
        "provenance": {},
        "baselines": {str(seed): _completed_record() for seed in study.BASE_SEEDS},
        "arms": {
            "feature_pso": cell_tree,
            "feature_random": copy.deepcopy(cell_tree),
            "feature_adam": {str(seed): _completed_record() for seed in study.BASE_SEEDS},
            "head_adam": {str(seed): _completed_record() for seed in study.BASE_SEEDS},
        },
        "ensemble": {
            "uniform": _completed_record(),
            "uniform_temperature": _completed_record(),
            "slsqp_weights": _completed_record(),
            "ensemble_pso": [_completed_record() for _ in study.SWARM_SEEDS],
        },
        "development_selection": {},
        "confirmation": {},
        "integrity": {"official_test_opened": False},
        "leakage_counters": {},
        "resource_ledger": {},
        "artifact_hashes": {artifact_name: artifact_hash},
    }


def test_strict_matrix_validation_accepts_complete_matrix_and_rejects_missing_cell(tmp_path: Path) -> None:
    for workload_id in study.DEFAULT_WORKLOAD_IDS:
        workload_root = tmp_path / "workloads" / workload_id
        workload_root.mkdir(parents=True)
        evidence = workload_root / "evidence.bin"
        evidence.write_bytes(workload_id.encode())
        relative = str(evidence.relative_to(tmp_path))
        result = _matrix_result(workload_id, relative, study.fingerprint_file(evidence))
        (workload_root / "result.json").write_text(json.dumps(result), encoding="utf-8")

    validated = study._validate_matrix_results(tmp_path, strict_development=True)
    assert set(validated) == set(study.DEFAULT_WORKLOAD_IDS)

    path = tmp_path / "workloads" / study.DEFAULT_WORKLOAD_IDS[0] / "result.json"
    broken = json.loads(path.read_text(encoding="utf-8"))
    del broken["arms"]["feature_pso"]["501"]["601"]
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(study.SealError, match="feature_pso matrix is incomplete"):
        study._validate_matrix_results(tmp_path, strict_development=True)


def test_development_reuse_requires_complete_hash_verified_artifacts(
    tmp_path: Path,
) -> None:
    workload_root = tmp_path / "workloads" / "synthetic"
    workload_root.mkdir(parents=True)
    artifact = workload_root / "evidence.bin"
    artifact.write_bytes(b"complete")
    relative = str(artifact.relative_to(tmp_path))
    result = _matrix_result(
        "synthetic",
        relative,
        study.fingerprint_file(artifact),
    )
    (workload_root / "result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    (workload_root / "development_reuse.json").write_text(
        json.dumps(
            {
                "protocol_version": study.PROTOCOL_VERSION,
                "source_run": "failed-but-preserved",
            }
        ),
        encoding="utf-8",
    )

    class ReusedAdapter:
        workload_id = "synthetic"

        def run_phase(self, phase: str) -> object:
            raise AssertionError(f"unexpected phase: {phase}")

    assert study._run_adapter_development(
        [ReusedAdapter()],
        tmp_path,
    ) == [result]
    artifact.write_bytes(b"drift")
    with pytest.raises(study.SealError, match="hash drift"):
        study._run_adapter_development(
            [ReusedAdapter()],
            tmp_path,
        )
