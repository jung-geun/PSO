"""Pinned Ultralytics YOLO11n/VOC adapter for the convergence protocol.

The module deliberately keeps Ultralytics, torchvision and ensemble-boxes imports
inside the operations that need them.  Importing this module is consequently safe
in the normal (non-detection) installation.  All persistent writes go through the
common protocol's atomic helpers and every phase is guarded by the run state.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import gc
import json
import math
import os
import random
import shutil
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from pso.optimizer import _RandomSource
from pso.plugins import ConstrictionMovement, IterationContext, SwarmState

from test.post_training_model_convergence import (
    BASE_SEEDS,
    PROJECTION_SEED,
    PSO_GENERATIONS,
    PARTICLE_COUNT,
    RESIDUAL_DIMENSION,
    RESIDUAL_BOUND,
    SWARM_SEEDS,
    AuditResult,
    ObjectiveResult,
    PROTOCOL_VERSION,
    ProtocolError,
    ResourceCounters,
    SealError,
    SelectedResidualCodec,
    StudyConfig,
    StudyState,
    StudyStateMachine,
    atomic_write_bytes,
    atomic_write_json,
    begin_confirmation,
    canonical_json,
    fingerprint_file,
    fingerprint_module,
    fingerprint_nonselected_state,
    finish_confirmation,
    freeze_run,
    load_frozen_manifest,
    load_state,
    prepare_run,
    run_equal_budget_random,
    run_residual_pso,
    run_state_neutral_audit,
    select_endpoint,
    sha256_bytes,
    verify_frozen_manifest,
)

WORKLOAD_ID = "voc_yolo11n"
FAMILY = "detection"
ULTRALYTICS_VERSION = "8.4.142"
ENSEMBLE_BOXES_VERSION = "1.0.9"
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
VOC_CLASS_TO_ID = {name: index for index, name in enumerate(VOC_CLASSES)}
BP_COUNT, REFINE_COUNT, SELECTION_COUNT, OBJECTIVE_COUNT = 11551, 2500, 2500, 512
IMG_SIZE = 640
EXPECTED_BLOCK_INDEX = 22
EXPECTED_DETECT_INDEX = 23
EXPECTED_HEAD_BIAS_COUNT = 252
VOC_YEARS = ("2007", "2012")

WBF_PARTICLE_COUNT = 12
WBF_GENERATIONS = 20

class YoloProtocolError(ProtocolError):
    """A detection-specific protocol violation."""


@dataclasses.dataclass(frozen=True)
class VOCRecord:
    year: str
    image_id: str
    image_path: str
    annotation_path: str
    width: int
    height: int
    labels: tuple[tuple[int, float, float, float, float], ...]
    difficult_excluded: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class VOCManifest:
    bp_train: tuple[VOCRecord, ...]
    refine_search: tuple[VOCRecord, ...]
    selection_val: tuple[VOCRecord, ...]
    permutation_seed: int
    duplicate_groups: Mapping[str, tuple[str, ...]]
    counts: Mapping[str, int]
    objective_keys: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bp_train": [r.to_dict() for r in self.bp_train],
            "refine_search": [r.to_dict() for r in self.refine_search],
            "selection_val": [r.to_dict() for r in self.selection_val],
            "permutation_seed": self.permutation_seed,
            "duplicate_groups": {k: list(v) for k, v in self.duplicate_groups.items()},
            "counts": dict(self.counts),
            "objective_keys": [list(x) for x in self.objective_keys],
        }


@dataclasses.dataclass(frozen=True)
class VOCTestGuard:
    run_root: str
    frozen_manifest_hash: str
    confirmation_started: bool

    def require_open(self) -> None:
        if not self.confirmation_started or not self.frozen_manifest_hash:
            raise SealError("VOC2007 test is sealed until frozen confirmation begins")


def _ultralytics() -> Any:
    """Import the pinned optional package only when a detection operation runs."""
    try:
        import ultralytics  # type: ignore
    except ImportError as exc:
        raise YoloProtocolError(
            "Ultralytics is required for voc_yolo11n; install ultralytics==8.4.142"
        ) from exc
    version = str(getattr(ultralytics, "__version__", ""))
    if version != ULTRALYTICS_VERSION:
        raise YoloProtocolError(
            f"Ultralytics version mismatch: expected {ULTRALYTICS_VERSION}, got {version or 'unknown'}"
        )
    return ultralytics


def _torchvision_voc() -> Any:
    try:
        from torchvision.datasets import VOCDetection  # type: ignore
    except ImportError as exc:
        raise YoloProtocolError("torchvision with VOCDetection is required for VOC preparation") from exc
    return VOCDetection


def _wbf() -> Callable[..., Any]:
    try:
        from ensemble_boxes import weighted_boxes_fusion  # type: ignore
    except ImportError as exc:
        raise YoloProtocolError(
            "ensemble-boxes is required for YOLO ensemble arms; install ensemble-boxes==1.0.9"
        ) from exc
    module = __import__("ensemble_boxes")
    version = str(getattr(module, "__version__", ""))
    if version and version != ENSEMBLE_BOXES_VERSION:
        raise YoloProtocolError(
            f"ensemble-boxes version mismatch: expected {ENSEMBLE_BOXES_VERSION}, got {version}"
        )
    return weighted_boxes_fusion


def pinned_preflight() -> dict[str, str]:
    """Check optional package pins without importing them at module import time."""
    ultra = _ultralytics()
    # Importing ensemble-boxes here verifies the package before a run starts.
    _wbf()
    return {
        "ultralytics": str(getattr(ultra, "__version__", "")),
        "ensemble_boxes": ENSEMBLE_BOXES_VERSION,
        "voc_classes": str(len(VOC_CLASSES)),
    }


def _canonical_pixels(image: Any) -> tuple[int, int, bytes]:
    """Return the label-free duplicate key required by the protocol."""
    try:
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        rgb = image.convert("RGB")
        width, height = rgb.size
        return width, height, np.asarray(rgb, dtype=np.uint8).tobytes(order="C")
    except ImportError as exc:
        raise YoloProtocolError("Pillow is required for VOC duplicate fingerprinting") from exc


def image_fingerprint(image: Any) -> str:
    width, height, pixels = _canonical_pixels(image)
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "little", signed=False))
    digest.update(height.to_bytes(8, "little", signed=False))
    digest.update(pixels)
    return digest.hexdigest()


def _parse_int(node: ET.Element, tag: str) -> int:
    child = node.find(tag)
    if child is None or child.text is None:
        raise YoloProtocolError(f"VOC annotation missing {tag}")
    try:
        return int(child.text)
    except ValueError as exc:
        raise YoloProtocolError(f"invalid integer in VOC annotation {tag}") from exc


def parse_voc_xml(annotation_path: str | os.PathLike[str], image_path: str | os.PathLike[str], *, year: str, image_id: str) -> VOCRecord:
    """Parse one XML and convert non-difficult objects to normalized xywh labels."""
    path = Path(annotation_path)
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise YoloProtocolError(f"VOC annotation has no size: {path}")
    width, height = _parse_int(size, "width"), _parse_int(size, "height")
    if width <= 0 or height <= 0:
        raise YoloProtocolError(f"invalid VOC dimensions in {path}")
    labels: list[tuple[int, float, float, float, float]] = []
    excluded = 0
    for object_node in root.findall("object"):
        name_node = object_node.find("name")
        if name_node is None or not name_node.text:
            raise YoloProtocolError(f"VOC object has no class in {path}")
        class_name = name_node.text.strip().lower()
        if class_name not in VOC_CLASS_TO_ID:
            raise YoloProtocolError(f"unknown VOC class {class_name!r} in {path}")
        difficult_node = object_node.find("difficult")
        difficult = difficult_node is not None and (difficult_node.text or "0").strip() == "1"
        if difficult:
            excluded += 1
            continue
        box = object_node.find("bndbox")
        if box is None:
            raise YoloProtocolError(f"VOC object has no bndbox in {path}")
        xmin, ymin = _parse_int(box, "xmin"), _parse_int(box, "ymin")
        xmax, ymax = _parse_int(box, "xmax"), _parse_int(box, "ymax")
        if xmax < xmin or ymax < ymin:
            raise YoloProtocolError(f"inverted VOC box in {path}")
        # This is the pinned Ultralytics VOC convention: center uses -1, while
        # width and height are the XML extent without an additional correction.
        center_x = ((xmin + xmax) / 2.0 - 1.0) / width
        center_y = ((ymin + ymax) / 2.0 - 1.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        values = (center_x, center_y, box_width, box_height)
        if not all(math.isfinite(value) for value in values):
            raise YoloProtocolError(f"non-finite VOC box in {path}")
        labels.append((VOC_CLASS_TO_ID[class_name], *values))
    try:
        fingerprint = image_fingerprint(image_path)
    except (OSError, ValueError) as exc:
        raise YoloProtocolError(f"cannot fingerprint VOC image {image_path}") from exc
    return VOCRecord(year, image_id, str(image_path), str(annotation_path), width, height, tuple(labels), excluded, fingerprint)


def write_yolo_label(record: VOCRecord, path: str | os.PathLike[str]) -> Path:
    lines = ["%d %.10f %.10f %.10f %.10f" % label for label in record.labels]
    return atomic_write_bytes(path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def _voc_roots(data_root: Path, year: str, *, split: str = "trainval") -> tuple[Path, Path, Path]:
    if split not in {"trainval", "test"}:
        raise YoloProtocolError(f"unsupported VOC split: {split}")
    root = data_root / "VOCdevkit" / f"VOC{year}"
    return root / "JPEGImages", root / "Annotations", root / "ImageSets" / "Main" / f"{split}.txt"
def _records_from_voc(data_root: Path, *, allow_download: bool) -> list[VOCRecord]:
    records: list[VOCRecord] = []
    for year in VOC_YEARS:
        image_root, annotation_root, split_path = _voc_roots(data_root, year)
        if not split_path.is_file():
            if not allow_download:
                raise YoloProtocolError(f"VOC{year} is unavailable; rerun preparation with --allow-download")
            VOCDetection = _torchvision_voc()
            VOCDetection(root=str(data_root), year=year, image_set="trainval", download=True)
        if not split_path.is_file():
            raise YoloProtocolError(f"torchvision did not create VOC{year} trainval manifest")
        ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for image_id in ids:
            image_path = image_root / f"{image_id}.jpg"
            annotation_path = annotation_root / f"{image_id}.xml"
            if not image_path.is_file() or not annotation_path.is_file():
                raise YoloProtocolError(f"incomplete VOC{year} item: {image_id}")
            records.append(parse_voc_xml(annotation_path, image_path, year=year, image_id=image_id))
    return records


def _assign_duplicate_groups(records: Sequence[VOCRecord], *, seed: int) -> tuple[list[VOCRecord], dict[str, tuple[str, ...]]]:
    groups: dict[str, list[VOCRecord]] = {}
    for record in records:
        groups.setdefault(record.fingerprint, []).append(record)
    rng = random.Random(seed)
    order = list(records)
    rng.shuffle(order)
    order_position = {f"{record.year}:{record.image_id}": index for index, record in enumerate(order)}
    group_map: dict[str, tuple[str, ...]] = {}
    for fingerprint, members in groups.items():
        members.sort(key=lambda record: order_position[f"{record.year}:{record.image_id}"])
        keys = tuple(f"{record.year}:{record.image_id}" for record in members)
        group_map[fingerprint] = keys
    grouped_order: list[VOCRecord] = []
    seen: set[str] = set()
    for record in order:
        if record.fingerprint in seen:
            continue
        seen.add(record.fingerprint)
        grouped_order.extend(groups[record.fingerprint])
    return grouped_order, group_map


def make_voc_manifests(records: Sequence[VOCRecord], *, seed: int = 20260908) -> VOCManifest:
    if len(records) != 16551:
        raise YoloProtocolError(f"expected 16,551 VOC trainval records, found {len(records)}")
    ordered, groups = _assign_duplicate_groups(records, seed=seed)
    by_key = {
        f"{record.year}:{record.image_id}": record for record in ordered
    }
    ordered_groups: list[tuple[VOCRecord, ...]] = []
    seen_fingerprints: set[str] = set()
    for record in ordered:
        if record.fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(record.fingerprint)
        ordered_groups.append(
            tuple(by_key[key] for key in groups[record.fingerprint])
        )

    def take_groups(
        available: Sequence[tuple[VOCRecord, ...]],
        count: int,
    ) -> tuple[tuple[VOCRecord, ...], list[tuple[VOCRecord, ...]]]:
        chosen: list[VOCRecord] = []
        deferred: list[tuple[VOCRecord, ...]] = []
        for group in available:
            if len(chosen) + len(group) <= count:
                chosen.extend(group)
            else:
                deferred.append(group)
        if len(chosen) != count:
            raise YoloProtocolError(
                f"duplicate-safe partition cannot satisfy exact size {count}"
            )
        return tuple(chosen), deferred

    bp, remaining = take_groups(ordered_groups, BP_COUNT)
    refine, remaining = take_groups(remaining, REFINE_COUNT)
    selection = tuple(record for group in remaining for record in group)
    if len(bp) != BP_COUNT or len(refine) != REFINE_COUNT or len(selection) != SELECTION_COUNT:
        raise YoloProtocolError("VOC duplicate grouping did not produce exact manifest sizes")
    all_keys = {f"{r.year}:{r.image_id}" for r in bp + refine + selection}
    if len(all_keys) != len(bp) + len(refine) + len(selection):
        raise YoloProtocolError("VOC manifests overlap")
    for split in (bp, refine, selection):
        if not set(range(20)).issubset({label[0] for record in split for label in record.labels}):
            raise YoloProtocolError("VOC manifest does not contain all 20 classes")
    objective = tuple((r.year, r.image_id) for r in refine[:OBJECTIVE_COUNT])
    return VOCManifest(bp, refine, selection, seed, groups, {
        "bp_train": len(bp), "refine_search": len(refine), "selection_val": len(selection),
    }, objective)


def prepare_voc(
    data_root: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    *,
    allow_download: bool,
    seed: int = 20260908,
) -> VOCManifest:
    data = Path(data_root)
    root = Path(run_root) / "workloads" / WORKLOAD_ID
    root.mkdir(parents=True, exist_ok=True)
    records = _records_from_voc(data, allow_download=allow_download)
    manifest = make_voc_manifests(records, seed=seed)
    atomic_write_json(root / "voc_manifest.json", manifest.to_dict())
    labels_root = root / "labels"
    images_root = root / "images"
    labels_root.mkdir(parents=True, exist_ok=True)
    for split_name, split in (
        ("bp_train", manifest.bp_train),
        ("refine_search", manifest.refine_search),
        ("selection_val", manifest.selection_val),
    ):
        for record in split:
            write_yolo_label(
                record,
                labels_root / split_name / f"{record.year}_{record.image_id}.txt",
            )
            destination = images_root / split_name / f"{record.year}_{record.image_id}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.symlink_to(Path(record.image_path).resolve())
            except FileExistsError:
                if not destination.exists():
                    raise YoloProtocolError(f"stale image link: {destination}")
            except OSError:
                shutil.copy2(record.image_path, destination)
    return manifest


def guarded_voc_test_loader(
    data_root: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    *,
    confirmation: bool = False,
) -> Any:
    """Open VOC2007 test only after a valid frozen manifest and confirmation."""
    root = Path(run_root)
    state = load_state(root)
    if state.state != StudyState.CONFIRMING or not confirmation:
        raise SealError("VOC2007 official test is sealed until confirm phase")
    frozen = verify_frozen_manifest(root)
    image_root, annotation_root, split_path = _voc_roots(Path(data_root), "2007", split="test")
    if not split_path.is_file():
        raise YoloProtocolError("VOC2007 test manifest is unavailable")
    records = []
    for image_id in (
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ):
        image_path = image_root / f"{image_id}.jpg"
        annotation_path = annotation_root / f"{image_id}.xml"
        records.append(
            parse_voc_xml(annotation_path, image_path, year="2007", image_id=image_id)
        )
    return records, VOCTestGuard(str(root), frozen.manifest_hash, True)


def write_study_yaml(
    manifest: VOCManifest,
    path: str | os.PathLike[str],
    *,
    labels_root: str | os.PathLike[str],
) -> Path:
    """Write a study-only YAML with train/selection images and no test/download."""
    labels = Path(labels_root).resolve()
    dataset_root = labels.parent
    yaml = (
        f"path: {dataset_root}\n"
        f"train: {dataset_root / 'images' / 'bp_train'}\n"
        f"val: {dataset_root / 'images' / 'selection_val'}\n"
        "names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(VOC_CLASSES))
        + "\n"
    )
    if "download:" in yaml or "\ntest:" in yaml:
        raise YoloProtocolError("study YAML cannot contain download or test entries")
    if manifest.counts.get("bp_train") != BP_COUNT:
        raise YoloProtocolError("study YAML manifest is not the pinned bp_train split")
    return atomic_write_bytes(path, yaml.encode("utf-8"))


def _model_yaml_path() -> str:
    try:
        import ultralytics
    except ImportError as exc:
        raise YoloProtocolError("Ultralytics is required to create YOLO11n") from exc
    path = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "11" / "yolo11.yaml"
    if not path.is_file():
        raise YoloProtocolError(f"pinned yolo11.yaml is missing: {path}")
    return str(path)


def make_yolo11n(*, device: str | torch.device = "cpu", nc: int = 20) -> Any:
    """Create a scratch YOLO11n with a rebuilt 20-class Detect head."""
    ultra = _ultralytics()
    if nc != 20:
        raise YoloProtocolError("VOC YOLO11n must have exactly 20 classes")
    wrapper = ultra.YOLO(_model_yaml_path(), task="detect")
    try:
        from ultralytics.nn.tasks import DetectionModel  # type: ignore
        wrapper.model = DetectionModel(_model_yaml_path(), ch=3, nc=nc, verbose=False)
        from ultralytics.cfg import get_cfg  # type: ignore
        wrapper.model.args = get_cfg()
    except (ImportError, TypeError) as exc:
        raise YoloProtocolError("pinned Ultralytics cannot construct 20-class DetectionModel") from exc
    wrapper.model.to(device)
    assert_yolo_topology(wrapper.model)
    return wrapper


def assert_yolo_topology(model: nn.Module) -> None:
    layers = getattr(model, "model", None)
    if layers is None or len(layers) <= EXPECTED_DETECT_INDEX:
        raise YoloProtocolError("YOLO11n graph is shorter than the pinned block22/Detect graph")
    block = layers[EXPECTED_BLOCK_INDEX]
    detect = layers[EXPECTED_DETECT_INDEX]
    if block.__class__.__name__ != "C3k2":
        raise YoloProtocolError(f"expected model.22 C3k2, found {block.__class__.__name__}")
    if detect.__class__.__name__ != "Detect":
        raise YoloProtocolError(f"expected model.23 Detect, found {detect.__class__.__name__}")
    if int(getattr(detect, "nc", -1)) != 20:
        raise YoloProtocolError(f"expected Detect.nc=20, found {getattr(detect, 'nc', None)}")
    if not hasattr(detect, "cv2") or not hasattr(detect, "cv3") or len(detect.cv2) != 3 or len(detect.cv3) != 3:
        raise YoloProtocolError("pinned Detect head must expose three cv2 and cv3 branches")
    bias = [*list(detect.cv2[i][-1].bias for i in range(3)), *list(detect.cv3[i][-1].bias for i in range(3))]
    if any(value is None for value in bias):
        raise YoloProtocolError("all Detect terminal heads must expose biases")
    if sum(int(value.numel()) for value in bias) != EXPECTED_HEAD_BIAS_COUNT:
        raise YoloProtocolError("Detect output bias dimension does not equal pinned 252")


def selected_block_names(model: nn.Module) -> tuple[str, ...]:
    names = tuple(name for name, _ in model.named_parameters() if name.startswith("model.22."))
    if not names:
        raise YoloProtocolError("no model.22 floating parameters found")
    if any(not dict(model.named_parameters())[name].is_floating_point() for name in names):
        raise YoloProtocolError("model.22 contains a non-floating selected parameter")
    return names


def selected_head_bias_names(model: nn.Module) -> tuple[str, ...]:
    names = tuple(
        name
        for name, _ in model.named_parameters()
        if len(name.split(".")) == 6
        and name.split(".")[0:2] == ["model", "23"]
        and name.split(".")[2] in {"cv2", "cv3"}
        and name.split(".")[3] in {"0", "1", "2"}
        and name.split(".")[4:] == ["2", "bias"]
    )
    if len(names) != 6 or sum(dict(model.named_parameters())[name].numel() for name in names) != EXPECTED_HEAD_BIAS_COUNT:
        raise YoloProtocolError("Detect bias selection does not match six tensors and 252 scalars")
    return names


def _detect_inputs(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the frozen prefix and return the three Detect inputs."""
    outputs: dict[int, torch.Tensor] = {}
    x = images
    layers = model.model
    for index, module in enumerate(layers[:EXPECTED_DETECT_INDEX]):
        source = getattr(module, "f", -1)
        if isinstance(source, int):
            x = x if source == -1 else outputs[source]
        else:
            x = [x if item == -1 else outputs[item] for item in source]
        x = module(x)
        outputs[index] = x
    try:
        return outputs[16].detach(), outputs[19].detach(), outputs[21].detach()
    except KeyError as exc:
        raise YoloProtocolError(
            "pinned YOLO graph did not produce cached tensors 16,19,21"
        ) from exc


def _letterbox_record(record: VOCRecord, *, size: int = IMG_SIZE) -> tuple[torch.Tensor, tuple[float, tuple[float, float]]]:
    """Decode one VOC image and apply the pinned fixed 640 letterbox."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise YoloProtocolError("Pillow is required for VOC image decoding") from exc
    image = Image.open(record.image_path).convert("RGB")
    width, height = image.size
    gain = min(size / width, size / height)
    resized = image.resize((max(1, round(width * gain)), max(1, round(height * gain))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - resized.width) / 2
    pad_y = (size - resized.height) / 2
    canvas.paste(resized, (round(pad_x), round(pad_y)))
    value = (
        torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy())
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
    )
    return value, (gain, (pad_x, pad_y))

def _native_target(
    record: VOCRecord,
    *,
    index: int = 0,
    ratio_pad: tuple[float, tuple[float, float]],
    size: int = IMG_SIZE,
) -> dict[str, torch.Tensor]:
    gain, (pad_x, pad_y) = ratio_pad
    transformed = []
    for label, center_x, center_y, width, height in record.labels:
        transformed.append(
            (
                (
                    center_x * record.width * gain + pad_x
                ) / size,
                (
                    center_y * record.height * gain + pad_y
                ) / size,
                width * record.width * gain / size,
                height * record.height * gain / size,
            )
        )
    cls = torch.tensor(
        [label[0] for label in record.labels],
        dtype=torch.float32,
    )
    boxes = torch.tensor(transformed, dtype=torch.float32)
    return {
        "batch_idx": torch.full(
            (len(record.labels),),
            index,
            dtype=torch.int64,
        ),
        "cls": cls.reshape(-1, 1),
        "bboxes": boxes.reshape(-1, 4),
    }

def native_batch(records: Sequence[VOCRecord], *, device: torch.device | str) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[tuple[float, tuple[float, float]], ...]]:
    images, targets, ratio_pad = [], [], []
    for index, record in enumerate(records):
        image, padding = _letterbox_record(record)
        images.append(image)
        targets.append(
            _native_target(record, index=index, ratio_pad=padding)
        )
        ratio_pad.append(padding)
    if not images:
        raise YoloProtocolError("native batch cannot be empty")
    batch = {key: torch.cat([target[key] for target in targets], dim=0).to(device) for key in ("batch_idx", "cls", "bboxes")}
    batch["img"] = torch.stack(images).to(device)
    return batch["img"], batch, tuple(ratio_pad)

@dataclasses.dataclass
class DetectionCache:
    images: torch.Tensor
    detect_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    targets: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.images = self.images.detach().clone()
        self.detect_inputs = tuple(value.detach().clone() for value in self.detect_inputs)  # type: ignore[assignment]

    def to(self, device: torch.device | str) -> "DetectionCache":
        return DetectionCache(self.images.to(device), tuple(value.to(device) for value in self.detect_inputs), self.targets, self.provenance)


def build_detection_cache(model: nn.Module, batches: Iterable[tuple[torch.Tensor, Mapping[str, Any]]], *, provenance: Mapping[str, Any], device: torch.device | str) -> DetectionCache:
    images_out: list[torch.Tensor] = []
    inputs_out = [[], [], []]
    targets: list[Mapping[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for images, batch in batches:
            images = images.to(device=device, dtype=torch.float32)
            cached = _detect_inputs(model, images)
            images_out.append(images.cpu())
            for index, value in enumerate(cached):
                inputs_out[index].append(value.cpu())
            targets.append(
                {
                    key: value.detach().cpu()
                    for key, value in batch.items()
                    if key != "img" and torch.is_tensor(value)
                }
                | {"_image_count": int(images.shape[0])}
            )
    if not images_out:
        raise YoloProtocolError("cannot create a cache from zero batches")
    return DetectionCache(
        torch.cat(images_out),
        tuple(torch.cat(values) for values in inputs_out),
        tuple(targets),
        dict(provenance),
    )


def _loss_callable(model: nn.Module) -> Any:
    try:
        from ultralytics.utils.loss import v8DetectionLoss  # type: ignore
    except ImportError as exc:
        raise YoloProtocolError(
            "pinned v8DetectionLoss is unavailable"
        ) from exc
    if not hasattr(model, "args") or not hasattr(model, "model"):
        raise YoloProtocolError(
            "native detection loss requires an Ultralytics DetectionModel"
        )
    return v8DetectionLoss(model)


def _loss_value(loss: Any) -> torch.Tensor:
    value = loss[0] if isinstance(loss, tuple) else loss
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.sum()


def _cache_target_batch(
    target: Mapping[str, Any],
    images: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    batch = {
        key: torch.as_tensor(target[key], device=device)
        for key in ("batch_idx", "cls", "bboxes")
        if key in target
    }
    batch["img"] = images
    return batch


def cached_detection_loss_tensor(
    model: nn.Module,
    cache: DetectionCache,
    *,
    model_device: torch.device | str = "cpu",
    backward: bool = False,
) -> torch.Tensor:
    """Evaluate cached loss in source-sized chunks.

    When ``backward`` is true, gradients are accumulated per chunk so the
    complete 2,500-image objective never materializes one CUDA graph.
    """
    model.eval()
    device = torch.device(model_device)
    loss_fn = _loss_callable(model)
    total = torch.zeros((), device=device)
    cursor = 0
    for target in cache.targets:
        image_count = int(target["_image_count"])
        stop = cursor + image_count
        images = cache.images[cursor:stop].to(device)
        inputs = tuple(
            value[cursor:stop].to(device)
            for value in cache.detect_inputs
        )
        batch = _cache_target_batch(target, images, device)
        with torch.set_grad_enabled(backward):
            predictions = model.model[EXPECTED_BLOCK_INDEX](inputs[2])
            outputs = model.model[EXPECTED_DETECT_INDEX](
                [inputs[0], inputs[1], predictions]
            )
            chunk = _loss_value(loss_fn(outputs, batch))
        if backward:
            (chunk / int(cache.images.shape[0])).backward()
        total = total + chunk.detach()
        cursor = stop
    if cursor != int(cache.images.shape[0]):
        raise YoloProtocolError("cached target/image counts disagree")
    return total / max(cursor, 1)


def cached_detection_objective(
    model: nn.Module,
    cache: DetectionCache,
    *,
    codec: SelectedResidualCodec | None = None,
    residual: torch.Tensor | None = None,
    model_device: torch.device | str = "cpu",
    backward: bool = False,
) -> ObjectiveResult:
    """Evaluate cached block22+Detect outputs through native v8DetectionLoss."""
    if codec is not None and residual is not None:
        codec.apply_residual(model, residual)
    value = cached_detection_loss_tensor(
        model,
        cache,
        model_device=model_device,
        backward=backward,
    )
    if backward:
        value.backward()
    count = int(cache.images.shape[0])
    return ObjectiveResult(
        float(value.detach().cpu()),
        count,
        1,
        int(backward),
    )


def cached_full_parity(model: nn.Module, cache: DetectionCache, *, model_device: torch.device | str = "cpu", atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
    """Compare full-prefix block22+Detect outputs against cached suffix execution."""
    device = torch.device(model_device)
    model.eval()
    with torch.no_grad():
        cached_inputs = tuple(value.to(device) for value in cache.detect_inputs)
        suffix = model.model[EXPECTED_BLOCK_INDEX](cached_inputs[2])
        cached = model.model[EXPECTED_DETECT_INDEX]([cached_inputs[0], cached_inputs[1], suffix])
        full_inputs = _detect_inputs(model, cache.images.to(device))
        full_suffix = model.model[EXPECTED_BLOCK_INDEX](full_inputs[2])
        full = model.model[EXPECTED_DETECT_INDEX]([full_inputs[0], full_inputs[1], full_suffix])
    def max_error(left: Any, right: Any) -> float:
        if torch.is_tensor(left) and torch.is_tensor(right):
            return float((left - right).abs().max().cpu())
        if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
            return max((max_error(a, b) for a, b in zip(left, right)), default=0.0)
        return 0.0
    error = max_error(cached, full)
    reference = cached[0] if isinstance(cached, (tuple, list)) else cached
    scale = float(reference.detach().abs().max().cpu()) if torch.is_tensor(reference) else 1.0
    allowed = atol + rtol * max(scale, 1.0)
    if error > allowed:
        raise YoloProtocolError(f"cached/full parity failed: max_error={error}, allowed={allowed}")
    return {"max_abs_error": error, "allowed": allowed, "passed": True}


def native_detection_metrics(predictions: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]], *, iou_thresholds: Sequence[float] = tuple(np.arange(0.5, 0.96, 0.05))) -> dict[str, Any]:
    """Compute dataset-level detection metrics from native boxes, not image means."""
    if len(predictions) != len(targets):
        raise YoloProtocolError("prediction/target image counts differ")
    # The pinned validator is the authority for production metrics.  This helper
    # is intentionally strict about shape and delegates matching when available.
    try:
        from ultralytics.utils.metrics import ap_per_class  # type: ignore
    except ImportError as exc:
        raise YoloProtocolError("pinned Ultralytics metric implementation unavailable") from exc
    stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for prediction, target in zip(predictions, targets):
        stats.append((
            np.asarray(prediction.get("correct", []), dtype=bool),
            np.asarray(prediction.get("conf", []), dtype=np.float32),
            np.asarray(prediction.get("pred_cls", []), dtype=np.float32),
            np.asarray(target.get("target_cls", []), dtype=np.float32),
        ))
    if not stats:
        return {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0, "per_class_ap": []}
    correct, conf, pred_cls, target_cls = (np.concatenate(parts) if any(parts) else np.empty((0,)) for parts in zip(*stats))
    if correct.ndim == 1:
        correct = correct[:, None]
    if correct.size == 0:
        return {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0, "per_class_ap": [0.0] * 20}
    result = ap_per_class(correct, conf, pred_cls, target_cls, plot=False, names={i: n for i, n in enumerate(VOC_CLASSES)})
    # Ultralytics has changed tuple ordering across versions; pinned 8.4.142 is
    # checked here rather than silently publishing an incorrectly labelled metric.
    if len(result) < 4:
        raise YoloProtocolError("unexpected pinned ap_per_class return shape")
    tp, fp, p, r, f1, ap, unique = result[:7]
    ap = np.asarray(ap)
    if ap.ndim == 2:
        per_class = ap.mean(axis=1)
        map50 = float(ap[:, 0].mean()) if ap.shape[1] else 0.0
        map5095 = float(ap.mean())
    else:
        per_class, map50, map5095 = ap, float(ap.mean()), float(ap.mean())
    return {"map50": map50, "map50_95": map5095, "precision": float(np.asarray(p).mean()), "recall": float(np.asarray(r).mean()), "per_class_ap": per_class.tolist()}


def transform_boxes_to_original(boxes: np.ndarray, *, ratio_pad: tuple[float, tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] < 4:
        raise YoloProtocolError("boxes must have shape (N,4+) in letterbox pixels")
    gain, pad = float(ratio_pad[0]), ratio_pad[1]
    if gain <= 0:
        raise YoloProtocolError("letterbox gain must be positive")
    values[:, [0, 2]] = (values[:, [0, 2]] - float(pad[0])) / gain
    values[:, [1, 3]] = (values[:, [1, 3]] - float(pad[1])) / gain
    height, width = shape
    values[:, [0, 2]] = np.clip(values[:, [0, 2]], 0, width)
    values[:, [1, 3]] = np.clip(values[:, [1, 3]], 0, height)
    return values


def transform_boxes_to_letterbox(boxes: np.ndarray, *, ratio_pad: tuple[float, tuple[float, float]]) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64).copy()
    gain, pad = float(ratio_pad[0]), ratio_pad[1]
    if gain <= 0:
        raise YoloProtocolError("letterbox gain must be positive")
    values[:, [0, 2]] = values[:, [0, 2]] * gain + float(pad[0])
    values[:, [1, 3]] = values[:, [1, 3]] * gain + float(pad[1])
    return values


def weighted_box_fusion(images: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> Mapping[str, np.ndarray]:
    if len(images) != len(weights) or not images:
        raise YoloProtocolError("WBF requires one image prediction per model and one weight per model")
    normalized_weights = np.asarray(weights, dtype=np.float64)
    if (
        not np.isfinite(normalized_weights).all()
        or (normalized_weights < 0).any()
        or float(normalized_weights.sum()) <= 0
    ):
        raise YoloProtocolError(
            "WBF weights must be finite, nonnegative, and have positive sum"
        )
    normalized_weights /= normalized_weights.sum()
    fuse = _wbf()
    boxes_list, scores_list, labels_list = [], [], []
    for image in images:
        boxes = np.asarray(image.get("boxes", []), dtype=np.float64)
        scores = np.asarray(image.get("scores", []), dtype=np.float64)
        labels = np.asarray(image.get("labels", []), dtype=np.int64)
        if boxes.size:
            boxes = boxes.reshape(-1, 4)
            if (boxes < 0).any() or (boxes > 1).any():
                raise YoloProtocolError("WBF boxes must be normalized original-coordinate xyxy")
        boxes_list.append(boxes.tolist())
        scores_list.append(scores.tolist())
        labels_list.append(labels.tolist())
    boxes, scores, labels = fuse(
        boxes_list, scores_list, labels_list, weights=normalized_weights.tolist(),
        iou_thr=.55, skip_box_thr=.001, conf_type="avg", allows_overflow=False,
    )
    order = np.argsort(-np.asarray(scores))[:300]
    return {"boxes": np.asarray(boxes)[order], "scores": np.asarray(scores)[order], "labels": np.asarray(labels, dtype=np.int64)[order]}



def _wbf_dataset_metrics(
    member_predictions: Sequence[Sequence[Mapping[str, Any]]],
    targets: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> dict[str, Any]:
    from test.evaluate_post_training_model_convergence import detection_metrics

    records: list[dict[str, Any]] = []
    for image_index, target in enumerate(targets):
        fused = weighted_box_fusion(
            [member_predictions[member][image_index] for member in range(3)],
            weights,
        )
        predictions = [
            {
                "box": [float(value) for value in box],
                "class_id": int(label),
                "score": float(score),
            }
            for box, score, label in zip(
                fused["boxes"], fused["scores"], fused["labels"]
            )
        ]
        boxes = np.asarray(target.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(target.get("labels", []), dtype=np.int64)
        ground_truth = [
            {
                "box": [float(value) for value in box],
                "class_id": int(label),
            }
            for box, label in zip(boxes, labels)
        ]
        records.append(
            {
                "image_id": str(target.get("image_id", image_index)),
                "predictions": predictions,
                "ground_truth": ground_truth,
            }
        )
    return detection_metrics(records, class_count=len(VOC_CLASSES))


def run_wbf_weight_search(
    member_predictions: Sequence[Sequence[Mapping[str, Any]]],
    targets: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    random_mode: bool,
) -> dict[str, Any]:
    if (
        len(member_predictions) != 3
        or any(len(rows) != len(targets) for rows in member_predictions)
        or seed not in SWARM_SEEDS
    ):
        raise YoloProtocolError(
            "WBF search requires three aligned member sets and a fixed swarm seed"
        )
    rng = _RandomSource(seed=seed, device="cpu")
    positions = [torch.zeros(3, dtype=torch.float32)]
    for _ in range(5):
        value = rng.uniform((3,), -0.25, 0.25, device="cpu")
        positions.extend((value, -value))
    positions.append(rng.uniform((3,), -0.25, 0.25, device="cpu"))
    velocities = [torch.zeros_like(position) for position in positions]
    pbest = [position.clone() for position in positions]
    pbest_scores = [math.inf] * len(positions)
    best = positions[0].clone()
    best_score = math.inf
    best_metrics: dict[str, Any] | None = None
    trajectory: list[dict[str, Any]] = []
    movement = ConstrictionMovement(c0=2.05, c1=2.05)
    evaluations = 0

    for generation in range(1, WBF_GENERATIONS + 1):
        for index, position in enumerate(positions):
            weights = torch.softmax(position, dim=0).tolist()
            metrics = _wbf_dataset_metrics(member_predictions, targets, weights)
            score = -float(metrics["map50_95"])
            evaluations += 1
            if score < pbest_scores[index]:
                pbest_scores[index] = score
                pbest[index] = position.clone()
            if score < best_score:
                best_score = score
                best = position.clone()
                best_metrics = metrics
        trajectory.append(
            {
                "generation": generation,
                "objective": best_score,
                "map50_95": -best_score,
            }
        )
        if generation == 20:
            break
        if random_mode:
            positions = [
                rng.uniform((3,), -5.0, 5.0, device="cpu")
                for _ in positions
            ]
            velocities = [torch.zeros_like(position) for position in positions]
            continue
        state = SwarmState(
            positions=tuple(position.clone() for position in positions),
            velocities=tuple(velocity.clone() for velocity in velocities),
            pbest_positions=tuple(position.clone() for position in pbest),
            pbest_scores=tuple((score, 0.0, 0.0) for score in pbest_scores),
            gbest_position=best.clone(),
            gbest_score=(best_score, 0.0, 0.0),
            pbest_improved=tuple(False for _ in positions),
        )
        next_positions: list[torch.Tensor] = []
        next_velocities: list[torch.Tensor] = []
        for index, position in enumerate(positions):
            context = IterationContext(
                epoch=generation + 1,
                total_epochs=20,
                w=1.0,
                particle_idx=index,
                is_negative=False,
                rng=rng,
                optimizer=None,
            )
            _, velocity = movement.propose(index, state, context)
            candidate = position + velocity
            outside = (candidate < -5.0) | (candidate > 5.0)
            next_positions.append(torch.clamp(candidate, -5.0, 5.0))
            next_velocities.append(
                torch.where(outside, torch.zeros_like(velocity), velocity)
            )
        positions, velocities = next_positions, next_velocities
    if (
        best_metrics is None
        or evaluations != WBF_PARTICLE_COUNT * WBF_GENERATIONS
    ):
        raise YoloProtocolError("WBF search did not complete exactly 240 evaluations")
    return {
        "seed": seed,
        "method": "ensemble_random" if random_mode else "ensemble_pso",
        "queries": evaluations,
        "sample_evaluations": evaluations * len(targets),
        "logits": best.tolist(),
        "weights": torch.softmax(best, dim=0).tolist(),
        "metrics": best_metrics,
        "trajectory": trajectory,
    }
class StrictScratchTrainer:
    """Native training boundary: OOM, NaN and invalid checkpoints are fatal."""
    def __init__(
        self,
        *,
        device: str,
        batch: int = 16,
        epochs: int = 100,
    ) -> None:
        if device not in {"cpu", "mps", "cuda"}:
            raise YoloProtocolError(
                "device must remain cpu, mps, or cuda for the "
                "complete run"
            )
        if epochs not in {2, 100}:
            raise YoloProtocolError("trainer epochs must be smoke 2 or production 100")
        self.device, self.batch, self.epochs = device, batch, epochs
        self.ema_capture: dict[str, torch.Tensor] | None = None
        self.telemetry: list[dict[str, Any]] = []

    @property
    def overrides(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs, "optimizer": "SGD", "lr0": .01, "lrf": .01,
            "momentum": .937, "weight_decay": .0005, "cos_lr": True,
            "warmup_epochs": 3, "batch": self.batch, "imgsz": IMG_SIZE,
            "amp": False, "workers": 0, "deterministic": True, "patience": 0,
            "pretrained": False, "close_mosaic": 10, "device": self.device,
            "val": True, "plots": False, "save": True,
        }

    def capture_live_ema(self, trainer: Any, epoch: int) -> None:
        metrics = getattr(trainer, "metrics", None)
        if self.epochs == 2 or epoch >= self.epochs - 11:
            row = {"epoch": epoch + 1}
            if isinstance(metrics, Mapping):
                row.update(
                    {
                        str(key): float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    }
                )
            self.telemetry.append(row)
        if epoch != self.epochs - 1:
            return
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema is None:
            raise YoloProtocolError("live fp32 EMA is unavailable at final epoch")
        self.ema_capture = {name: value.detach().float().cpu().clone() for name, value in ema.state_dict().items()}
        if not all(bool(torch.isfinite(value).all()) for value in self.ema_capture.values()):
            raise YoloProtocolError("live EMA contains non-finite values")

    def refusal(self, error: BaseException) -> None:
        message = str(error).lower()
        if "out of memory" in message or "nan" in message or "checkpoint" in message:
            raise YoloProtocolError(f"native training failed without recovery: {error}") from error
        raise error


def _reused_native_baseline(
    run_path: Path,
    trainer: StrictScratchTrainer,
    base_seed: int,
) -> dict[str, Any] | None:
    baseline_root = (
        run_path
        / "workloads"
        / WORKLOAD_ID
        / "baselines"
        / str(base_seed)
    )
    marker_path = baseline_root / "baseline_reuse.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    output = baseline_root / "ema_fp32.pt"
    results_path = (
        run_path
        / "ultralytics"
        / f"base-{base_seed}-{trainer.epochs}e"
        / "results.csv"
    )
    if (
        marker.get("protocol_version") != PROTOCOL_VERSION
        or marker.get("checkpoint_hash") != fingerprint_file(output)
        or marker.get("results_hash") != fingerprint_file(results_path)
    ):
        raise YoloProtocolError(
            f"invalid reused baseline marker: {marker_path}"
        )
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (
        len(rows) != trainer.epochs
        or int(float(rows[-1]["epoch"])) != trainer.epochs
    ):
        raise YoloProtocolError(
            "reused native baseline does not contain every epoch"
        )
    state = torch.load(output, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
        torch.is_tensor(value)
        and bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise YoloProtocolError(
            "reused native baseline checkpoint is invalid"
        )
    telemetry = [
        {
            key.strip(): float(value)
            for key, value in row.items()
            if key is not None
            and value is not None
            and value.strip()
        }
        for row in rows[-11:]
    ]
    return {
        "seed": base_seed,
        "checkpoint": str(output.relative_to(run_path)),
        "telemetry": telemetry,
        "resolved": trainer.overrides,
        "checkpoint_hash": fingerprint_file(output),
        "result": f"reused:{marker['source_run']}",
        "reused": True,
    }


def train_baseline(
    *,
    model: Any,
    yaml_path: str,
    trainer: StrictScratchTrainer,
    run_root: str | os.PathLike[str],
    base_seed: int,
) -> dict[str, Any]:
    """Run one native baseline, or reuse an explicitly hash-verified run."""
    _ultralytics()
    if not isinstance(base_seed, int) or base_seed not in BASE_SEEDS:
        raise YoloProtocolError("base seed must be one of 501, 502, 503")
    run_path = Path(run_root)
    os.environ["YOLO_CONFIG_DIR"] = str(run_path / "yolo_config")
    os.environ["ULTRALYTICS_HUB"] = "0"
    os.environ["ULTRALYTICS_SETTINGS_YAML"] = str(run_path / "ultralytics_settings.yaml")
    reused = _reused_native_baseline(
        run_path,
        trainer,
        base_seed,
    )
    if reused is not None:
        return reused
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    callback = lambda tr: trainer.capture_live_ema(tr, int(getattr(tr, "epoch", -1)))
    add_callback = getattr(model, "add_callback", None)
    if not callable(add_callback):
        raise YoloProtocolError("Ultralytics model does not expose add_callback")
    add_callback("on_train_epoch_end", callback)
    try:
        results = model.train(
            data=yaml_path,
            seed=base_seed,
            project=str(run_path.resolve() / "ultralytics"),
            name=f"base-{base_seed}-{trainer.epochs}e",
            exist_ok=False,
            **trainer.overrides,
        )
    except BaseException as exc:
        trainer.refusal(exc)
    if trainer.ema_capture is None:
        raise YoloProtocolError("training did not capture final live fp32 EMA")
    output = (
        run_path
        / "workloads"
        / WORKLOAD_ID
        / "baselines"
        / str(base_seed)
        / "ema_fp32.pt"
    )
    atomic_write_bytes(
        output,
        _torch_save_bytes(trainer.ema_capture),
    )
    return {
        "seed": base_seed,
        "checkpoint": str(output.relative_to(run_path)),
        "checkpoint_hash": fingerprint_file(output),
        "telemetry": trainer.telemetry,
        "resolved": trainer.overrides,
        "result": str(results),
    }


def _resolve_run_path(
    run_root: str | os.PathLike[str],
    value: str | os.PathLike[str],
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(run_root) / path

def _checkpoint_record_valid(
    run_root: str | os.PathLike[str],
    record: Any,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    checkpoint = record.get("checkpoint")
    expected = record.get("checkpoint_hash")
    if not isinstance(checkpoint, str) or not isinstance(expected, str):
        return False
    path = _resolve_run_path(run_root, checkpoint)
    return path.is_file() and fingerprint_file(path) == expected


def _torch_save_bytes(value: Any) -> bytes:
    import io
    stream = io.BytesIO(); torch.save(value, stream); return stream.getvalue()


def _manifest_from_json(path: str | os.PathLike[str]) -> VOCManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    def records(key: str) -> tuple[VOCRecord, ...]:
        rows = []
        for row in value[key]:
            item = dict(row)
            item["labels"] = tuple(tuple(label) for label in item["labels"])
            rows.append(VOCRecord(**item))
        return tuple(rows)
    return VOCManifest(
        records("bp_train"), records("refine_search"), records("selection_val"),
        int(value["permutation_seed"]),
        {str(k): tuple(v) for k, v in value["duplicate_groups"].items()},
        {str(k): int(v) for k, v in value["counts"].items()},
        tuple(tuple(item) for item in value["objective_keys"]),
    )


def _native_loss_tensor(detector: nn.Module, images: torch.Tensor, batch: Mapping[str, Any]) -> torch.Tensor:
    detector.train()
    prediction = detector(images)
    loss = detector.loss(dict(batch), prediction) if callable(getattr(detector, "loss", None)) else _loss_callable(detector)(prediction, dict(batch))
    return _loss_value(loss) / max(int(images.shape[0]), 1)


def _accumulated_native_loss(detector: nn.Module, batches: Sequence[tuple[torch.Tensor, Mapping[str, Any]]]) -> torch.Tensor:
    if not batches:
        raise YoloProtocolError("native objective requires at least one batch")
    total: torch.Tensor | None = None
    count = 0
    for images, batch in batches:
        value = _native_loss_tensor(detector, images, batch)
        weight = int(images.shape[0])
        total = value * weight if total is None else total + value * weight
        count += weight
    if total is None or count == 0:
        raise YoloProtocolError("native objective contains zero images")
    return total / count


def _native_predictions(detector: nn.Module, images: torch.Tensor) -> list[Any]:
    from ultralytics.utils.nms import non_max_suppression  # type: ignore
    detector.eval()
    with torch.no_grad():
        raw = detector(images)
        return non_max_suppression(raw, conf_thres=.001, iou_thres=.7, max_det=300, multi_label=True, agnostic=False)




def evaluate_detection_records(
    detector: nn.Module,
    records: Sequence[VOCRecord],
    *,
    device: torch.device | str,
    batch_size: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from test.evaluate_post_training_model_convergence import detection_metrics

    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        subset = records[start : start + batch_size]
        images, _, ratio_pad = native_batch(subset, device=device)
        outputs = _native_predictions(detector, images)
        for record, output, padding in zip(subset, outputs, ratio_pad):
            boxes = (
                output[:, :4].detach().cpu().numpy()
                if output.numel()
                else np.empty((0, 4))
            )
            boxes = transform_boxes_to_original(
                boxes,
                ratio_pad=padding,
                shape=(record.height, record.width),
            )
            predictions = [
                {
                    "box": [float(value) for value in box],
                    "class_id": int(label),
                    "score": float(score),
                }
                for box, score, label in zip(
                    boxes,
                    output[:, 4].detach().cpu().numpy()
                    if output.numel()
                    else np.empty((0,)),
                    output[:, 5].detach().cpu().numpy()
                    if output.numel()
                    else np.empty((0,)),
                )
            ]
            ground_truth = []
            for label in record.labels:
                _, cx, cy, width, height = label
                ground_truth.append(
                    {
                        "box": [
                            float((cx - width / 2) * record.width),
                            float((cy - height / 2) * record.height),
                            float((cx + width / 2) * record.width),
                            float((cy + height / 2) * record.height),
                        ],
                        "class_id": int(label[0]),
                    }
                )
            rows.append(
                {
                    "image_id": f"{record.year}:{record.image_id}",
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                }
            )
    return detection_metrics(rows, class_count=len(VOC_CLASSES)), rows

def run_smoke_feature_pso(detector: nn.Module, cache: DetectionCache, *, device: torch.device | str) -> dict[str, Any]:
    """Run the real two-generation smoke PSO over cached native loss."""
    codec = SelectedResidualCodec(detector, selected_block_names(detector), projection_seed=PROJECTION_SEED)
    generator = torch.Generator(device="cpu").manual_seed(601)
    positions = [torch.zeros(RESIDUAL_DIMENSION, device=device)]
    for _ in range(11):
        positions.append(torch.rand(RESIDUAL_DIMENSION, generator=generator).to(device).mul_(.5).sub_(.25))
    velocities = [torch.zeros_like(position) for position in positions]
    pbest = [position.clone() for position in positions]
    scores: list[float | None] = [None] * len(positions)
    gbest: torch.Tensor | None = None
    gscore = math.inf
    trajectory = []
    for generation in range(1, 3):
        for index, position in enumerate(positions):
            result = cached_detection_objective(detector, cache, codec=codec, residual=position, model_device=device)
            if scores[index] is None or result.loss < scores[index]:
                scores[index] = result.loss
                pbest[index] = position.clone()
            if result.loss < gscore:
                gscore, gbest = result.loss, position.clone()
        if gbest is None:
            raise YoloProtocolError("smoke PSO did not produce an incumbent")
        trajectory.append({"generation": generation, "objective_best": gscore})
        if generation == 2:
            break
        for index in range(len(positions)):
            r1 = torch.rand(RESIDUAL_DIMENSION, generator=generator).to(device)
            r2 = torch.rand(RESIDUAL_DIMENSION, generator=generator).to(device)
            velocity = .7 * velocities[index] + 1.49445 * r1 * (pbest[index] - positions[index]) + 1.49445 * r2 * (gbest - positions[index])
            proposal = torch.clamp(positions[index] + velocity, -1.0, 1.0)
            velocities[index] = torch.where((proposal == -1.0) | (proposal == 1.0), torch.zeros_like(velocity), velocity)
            positions[index] = proposal
    return {"generations": 2, "queries": 24, "best_objective": gscore, "trajectory": trajectory}
def run_feature_search(
    model: nn.Module,
    objective: Callable[[torch.Tensor], ObjectiveResult | float],
    *,
    base_seed: int,
    swarm_seed: int,
    device: torch.device | str,
    validation: Callable[[torch.Tensor], AuditResult] | None = None,
) -> dict[str, Any]:
    """Run fixed feature PSO and equal-query random arms for one base."""
    names = selected_block_names(model)
    codec = SelectedResidualCodec(
        model,
        names,
        projection_seed=PROJECTION_SEED,
    )
    if base_seed not in BASE_SEEDS or swarm_seed not in SWARM_SEEDS:
        raise YoloProtocolError(
            "feature arms require the fixed base and swarm seed sets"
        )
    pso = run_residual_pso(
        objective,
        codec,
        seed=swarm_seed,
        device=device,
        model=model,
        validation=validation,
        objective_samples=OBJECTIVE_COUNT,
    )
    random_result = run_equal_budget_random(
        objective,
        codec,
        seed=swarm_seed,
        device=device,
        model=model,
        validation=validation,
        objective_samples=OBJECTIVE_COUNT,
    )
    return {
        "base_seed": base_seed,
        "swarm_seed": swarm_seed,
        "selected_names": list(names),
        "projection_seed": PROJECTION_SEED,
        "feature_pso": pso.to_dict(include_vectors=True),
        "feature_random": random_result.to_dict(include_vectors=True),
    }


def run_bounded_adam(
    model: nn.Module,
    parameters: Sequence[str],
    objective: Callable[[bool], torch.Tensor],
    *,
    updates: int = 40,
    lr: float = 1e-3,
    bounds: float | Mapping[str, float] = 0.25,
) -> dict[str, Any]:
    """Run the fixed 40-step full-objective AdamW arm from the base state."""
    if updates != 40 or lr != 1e-3:
        raise YoloProtocolError("AdamW arm requires exactly 40 updates at lr=1e-3")
    named = dict(model.named_parameters())
    selected = [named[name] for name in parameters if name in named]
    if len(selected) != len(parameters):
        raise YoloProtocolError("AdamW arm includes an unknown parameter")
    base = {name: value.detach().clone() for name, value in zip(parameters, selected)}
    optimizer = torch.optim.AdamW(selected, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.0)
    trajectory: list[dict[str, float]] = []
    try:
        for update in range(updates + 1):
            should_update = update < updates
            if should_update:
                optimizer.zero_grad(set_to_none=True)
            value = objective(should_update)
            if not torch.is_tensor(value) or value.ndim != 0 or not bool(torch.isfinite(value).item()):
                raise YoloProtocolError("AdamW objective must return one finite scalar tensor")
            trajectory.append({"update": update, "objective": float(value.detach().cpu())})
            if not should_update:
                break
            optimizer.step()
            with torch.no_grad():
                for name, parameter in zip(parameters, selected):
                    limit = float(bounds[name] if isinstance(bounds, Mapping) else bounds)
                    parameter.copy_(torch.clamp(parameter, base[name] - limit, base[name] + limit))
    finally:
        optimizer.zero_grad(set_to_none=True)
    return {
        "method": "feature_adam" if any(name.startswith("model.22.") for name in parameters) else "head_adam",
        "parameters": list(parameters),
        "updates": updates,
        "trajectory": trajectory,
        "final_objective": trajectory[-1]["objective"],
    }


def run_head_adam(
    model: nn.Module,
    objective: Callable[[bool], torch.Tensor],
) -> dict[str, Any]:
    """Run the six Detect-terminal-bias control with its declared ±0.25 box."""
    return run_bounded_adam(
        model,
        selected_head_bias_names(model),
        objective,
        bounds=0.25,
    )


class YoloConvergenceAdapter:
    def __init__(self, *, workload_id: str, config: StudyConfig, run_root: str | os.PathLike[str], data_root: str | os.PathLike[str], device: str | torch.device, allow_download: bool) -> None:
        if workload_id != WORKLOAD_ID:
            raise YoloProtocolError(f"unsupported workload id: {workload_id}")
        if str(device) not in {"cpu", "mps", "cuda"}:
            raise YoloProtocolError(
                "device must be cpu, mps, or cuda"
            )
        self.workload_id, self.config = workload_id, config
        self.run_root, self.data_root, self.device = Path(run_root), Path(data_root), str(device)
        self.allow_download = bool(allow_download)
        self.root = self.run_root / "workloads" / WORKLOAD_ID
        self.result_path = self.root / "result.json"
        self.result: dict[str, Any] = {"workload_id": WORKLOAD_ID, "family": FAMILY, "config": config.to_dict(), "manifests": {}, "provenance": {}, "baselines": {}, "arms": {}, "ensemble": {}, "development_selection": {}, "confirmation": {}, "integrity": {}, "leakage_counters": {"official_test_data_loaded_before_freeze": False, "official_test_evaluations_before_freeze": 0, "official_test_construction": 0, "official_test_forward_passes": 0}, "resource_ledger": {}, "artifact_hashes": {}}
        if self.result_path.is_file():
            persisted = json.loads(self.result_path.read_text(encoding="utf-8"))
            if persisted.get("workload_id") != WORKLOAD_ID:
                raise YoloProtocolError("persisted workload id mismatch")
            if StudyConfig.from_dict(persisted.get("config", {})) != config:
                raise YoloProtocolError("persisted configuration mismatch")
            self.result = persisted

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.result_path, self.result)

    def prepare(self) -> dict[str, Any]:
        if not (self.run_root / "state.json").is_file():
            prepare_run(self.run_root, self.config)
        package = pinned_preflight()
        manifest = prepare_voc(self.data_root, self.run_root, allow_download=self.allow_download, seed=self.config.split_seed)
        yaml_path = write_study_yaml(manifest, self.root / "study.yaml", labels_root=self.root / "labels")
        self.result["manifests"] = {"voc": str(self.root / "voc_manifest.json"), "study_yaml": str(yaml_path), "counts": dict(manifest.counts), "objective_count": len(manifest.objective_keys)}
        self.result["provenance"] = {"packages": package, "projection_seed": PROJECTION_SEED, "data_root": str(self.data_root)}
        self.result["integrity"] = {"prepared": True, "classes": list(VOC_CLASSES), "test_sealed": True, "official_test_opened": False}
        self._save()
        return self.result

    def smoke(self) -> dict[str, Any]:
        if not (self.root / "voc_manifest.json").is_file():
            self.prepare()
        pinned_preflight()
        manifest = _manifest_from_json(self.root / "voc_manifest.json")
        smoke_train = manifest.bp_train[:32]
        smoke_val = manifest.selection_val[:16]
        train_list = self.root / "smoke_train.txt"
        val_list = self.root / "smoke_val.txt"
        atomic_write_bytes(
            train_list,
            (
                "\n".join(
                    str(
                        (
                            self.root
                            / "images"
                            / "bp_train"
                            / f"{record.year}_{record.image_id}.jpg"
                        ).absolute()
                    )
                    for record in smoke_train
                )
                + "\n"
            ).encode("utf-8"),
        )
        atomic_write_bytes(
            val_list,
            (
                "\n".join(
                    str(
                        (
                            self.root
                            / "images"
                            / "selection_val"
                            / f"{record.year}_{record.image_id}.jpg"
                        ).absolute()
                    )
                    for record in smoke_val
                )
                + "\n"
            ).encode("utf-8"),
        )
        smoke_yaml = self.root / "smoke.yaml"
        atomic_write_bytes(
            smoke_yaml,
            (
                f"path: {self.root.resolve()}\n"
                f"train: {train_list.resolve()}\n"
                f"val: {val_list.resolve()}\n"
                "names:\n"
                + "\n".join(
                    f"  {index}: {name}"
                    for index, name in enumerate(VOC_CLASSES)
                )
                + "\n"
            ).encode("utf-8"),
        )
        smoke_wrapper = make_yolo11n(device=self.device)
        smoke_training = train_baseline(
            model=smoke_wrapper,
            yaml_path=str(smoke_yaml),
            trainer=StrictScratchTrainer(
                device=self.device,
                batch=4,
                epochs=2,
            ),
            run_root=self.run_root,
            base_seed=BASE_SEEDS[0],
        )
        images, batch, _ = native_batch(
            manifest.refine_search[:2],
            device=self.device,
        )
        detector = make_yolo11n(device=self.device).model
        detector.load_state_dict(
            torch.load(
                _resolve_run_path(
                    self.run_root,
                    smoke_training["checkpoint"],
                ),
                map_location=self.device,
                weights_only=True,
            ),
            strict=True,
        )
        detector.train()
        loss = _native_loss_tensor(detector, images, batch)
        if not bool(torch.isfinite(loss).item()):
            raise YoloProtocolError("smoke native detection loss is non-finite")
        detector.zero_grad(set_to_none=True)
        loss.backward()
        detector.zero_grad(set_to_none=True)
        detector.eval()
        predictions = _native_predictions(detector, images)
        cache = build_detection_cache(detector, [(images, batch)], provenance={"phase": "smoke"}, device=self.device)
        parity_zero = cached_full_parity(detector, cache, model_device=self.device)
        names = selected_block_names(detector)
        codec = SelectedResidualCodec(detector, names, projection_seed=PROJECTION_SEED)
        zero = codec.zero_residual(device=self.device)
        zero_loss = cached_detection_objective(detector, cache, codec=codec, residual=zero, model_device=self.device)
        smoke_pso = run_smoke_feature_pso(detector, cache, device=self.device)
        nonzero = torch.full((RESIDUAL_DIMENSION,), .1, device=self.device)
        nonzero_loss = cached_detection_objective(detector, cache, codec=codec, residual=nonzero, model_device=self.device)
        nonzero_parity = cached_full_parity(detector, cache, model_device=self.device)
        codec.restore_base(detector)
        if not math.isfinite(nonzero_loss.loss):
            raise YoloProtocolError("smoke cached native loss is non-finite")
        checkpoint = self.root / "smoke_checkpoint.pt"
        atomic_write_bytes(checkpoint, _torch_save_bytes(detector.state_dict()))
        reloaded = make_yolo11n(device=self.device).model
        reloaded.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True), strict=True)
        self.result["integrity"].update({
            "smoke": True,
            "selected_block_names": list(names),
            "selected_head_bias_names": list(selected_head_bias_names(detector)),
            "topology": "model.22 C3k2 -> model.23 Detect",
            "smoke_native_loss": zero_loss.to_dict(),
            "smoke_nonzero_loss": nonzero_loss.to_dict(),
            "smoke_cached_full_parity": parity_zero,
            "smoke_nonzero_full_parity": nonzero_parity,
            "smoke_pso": smoke_pso,
            "smoke_training": {
                "epochs": 2,
                "batch": 4,
                "telemetry": smoke_training["telemetry"],
                "checkpoint": str(checkpoint.relative_to(self.run_root)),
            },
            "smoke_nms_images": len(predictions),
            "smoke_checkpoint": str(checkpoint),
        })
        self._save()
        return self.result

    def develop(self) -> dict[str, Any]:
        if not (self.root / "voc_manifest.json").is_file():
            self.prepare()
        pinned_preflight()
        manifest = _manifest_from_json(self.root / "voc_manifest.json")
        yaml_path = self.root / "study.yaml"
        if not yaml_path.is_file():
            write_study_yaml(manifest, yaml_path, labels_root=self.root / "labels")
        objective_records = manifest.refine_search[:OBJECTIVE_COUNT]
        objective_batches = []
        for start in range(0, len(objective_records), 8):
            subset = objective_records[start:start + 8]
            images, batch, _ = native_batch(subset, device="cpu")
            objective_batches.append((images, batch))
        selection_records = manifest.selection_val
        baselines: dict[str, Any] = dict(
            self.result.get("baselines", {})
        )
        for base_seed in BASE_SEEDS:
            trainer = StrictScratchTrainer(device=self.device, batch=16)
            baseline = baselines.get(str(base_seed))
            if not _checkpoint_record_valid(self.run_root, baseline):
                training_model = make_yolo11n(device=self.device)
                baseline = train_baseline(
                    model=training_model,
                    yaml_path=str(yaml_path),
                    trainer=trainer,
                    run_root=self.run_root,
                    base_seed=base_seed,
                )
                baselines[str(base_seed)] = baseline
                self.result["baselines"] = baselines
                self._save()
                del training_model
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            detector = make_yolo11n(device=self.device).model
            state = torch.load(
                _resolve_run_path(
                    self.run_root,
                    baseline["checkpoint"],
                ),
                map_location=self.device,
                weights_only=True,
            )
            detector.load_state_dict(state, strict=True)
            detector.to(self.device)
            detector.eval()
            cache = build_detection_cache(
                detector, objective_batches,
                provenance={"base_seed": base_seed, "split": "refine_search", "count": OBJECTIVE_COUNT},
                device=self.device,
            )
            objective = lambda residual, detector=detector, cache=cache: cached_detection_objective(detector, cache, codec=None, residual=None, model_device=self.device)
            pristine_state = {
                name: value.detach().cpu().clone()
                for name, value in detector.state_dict().items()
            }
            self.result["arms"].setdefault("feature_pso", {}).setdefault(
                str(base_seed), {}
            )
            self.result["arms"].setdefault("feature_random", {}).setdefault(
                str(base_seed), {}
            )
            arm_record: dict[str, Any] = {}
            for swarm_seed in SWARM_SEEDS:
                existing = [
                    self.result["arms"][method][str(base_seed)].get(
                        str(swarm_seed)
                    )
                    for method in ("feature_pso", "feature_random")
                ]
                if all(
                    _checkpoint_record_valid(self.run_root, record)
                    for record in existing
                ):
                    for method, record in zip(
                        ("feature_pso", "feature_random"),
                        existing,
                    ):
                        arm_record[f"{method}:{swarm_seed}"] = record
                    continue
                codec = SelectedResidualCodec(
                    detector,
                    selected_block_names(detector),
                    projection_seed=PROJECTION_SEED,
                )
                arm_objective = (
                    lambda residual, detector=detector, cache=cache:
                    cached_detection_objective(
                        detector,
                        cache,
                        codec=None,
                        residual=None,
                        model_device=self.device,
                    )
                )
                def selection_audit(
                    residual: torch.Tensor,
                    detector: nn.Module = detector,
                    codec: SelectedResidualCodec = codec,
                ) -> AuditResult:
                    with codec.applied(detector, residual):
                        metrics, _ = evaluate_detection_records(
                            detector,
                            selection_records,
                            device=self.device,
                        )
                    return AuditResult(
                        loss=-float(metrics["map50_95"]),
                        primary_metric=float(metrics["map50_95"]),
                        samples=len(selection_records),
                        metadata=metrics,
                    )

                combined = run_feature_search(
                    detector,
                    arm_objective,
                    base_seed=base_seed,
                    swarm_seed=swarm_seed,
                    device=self.device,
                    validation=selection_audit,
                )
                for method in ("feature_pso", "feature_random"):
                    record = dict(combined[method])
                    record.update(
                        {
                            "base_seed": base_seed,
                            "swarm_seed": swarm_seed,
                            "projection_seed": PROJECTION_SEED,
                        }
                    )
                    endpoint_by_generation = {
                        int(endpoint["generation"]): endpoint["residual"]
                        for endpoint in record.get("endpoints", [])
                    }
                    ranked_checkpoints: list[
                        tuple[float, int, list[float], Mapping[str, Any]]
                    ] = []
                    trajectory = record.get("trajectory", [])
                    if trajectory:
                        initial = trajectory[0].get("initial_validation")
                        if isinstance(initial, Mapping):
                            ranked_checkpoints.append(
                                (
                                    float(initial["loss"]),
                                    0,
                                    [0.0] * RESIDUAL_DIMENSION,
                                    initial,
                                )
                            )
                    for row in trajectory:
                        audit = row.get("validation")
                        generation = int(row.get("generation", -1))
                        if isinstance(audit, Mapping):
                            ranked_checkpoints.append(
                                (
                                    float(audit["loss"]),
                                    generation,
                                    list(endpoint_by_generation[generation]),
                                    audit,
                                )
                            )
                    if not ranked_checkpoints:
                        raise YoloProtocolError(
                            "feature arm has no selection checkpoints"
                        )
                    ranked_checkpoints.sort(key=lambda item: item[:2])
                    _, selected_generation, selected_residual, selected_audit = (
                        ranked_checkpoints[0]
                    )
                    selection_metrics = dict(
                        selected_audit.get("metadata", {})
                    )
                    candidate_model = make_yolo11n(device=self.device).model
                    candidate_model.load_state_dict(pristine_state, strict=True)
                    candidate_codec = SelectedResidualCodec(
                        candidate_model,
                        selected_block_names(candidate_model),
                        projection_seed=PROJECTION_SEED,
                    )
                    residual = torch.as_tensor(
                        selected_residual,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    candidate_codec.apply_residual(candidate_model, residual)
                    checkpoint = (
                        self.root
                        / "arms"
                        / str(base_seed)
                        / f"{method}-{swarm_seed}.pt"
                    )
                    atomic_write_bytes(
                        checkpoint,
                        _torch_save_bytes(candidate_model.state_dict()),
                    )
                    record["selection_metrics"] = selection_metrics
                    record["selected_generation"] = selected_generation
                    record["selected_residual"] = residual.detach().cpu().tolist()
                    record["checkpoint"] = str(
                        checkpoint.relative_to(self.run_root)
                    )
                    record["checkpoint_hash"] = fingerprint_file(checkpoint)
                    self.result["arms"][method][str(base_seed)][
                        str(swarm_seed)
                    ] = record
                    arm_record[f"{method}:{swarm_seed}"] = record
                    self._save()
            for method in ("feature_adam", "head_adam"):
                existing = self.result["arms"].get(method, {}).get(
                    str(base_seed)
                )
                if _checkpoint_record_valid(self.run_root, existing):
                    arm_record[method] = existing
                    continue
                control_model = make_yolo11n(device=self.device).model
                control_model.load_state_dict(pristine_state, strict=True)
                if method == "feature_adam":
                    parameter_names = selected_block_names(control_model)
                    feature_codec = SelectedResidualCodec(
                        control_model,
                        parameter_names,
                        projection_seed=PROJECTION_SEED,
                    )
                    feature_bounds = {
                        name: RESIDUAL_BOUND * scale
                        for name, scale in zip(
                            feature_codec.names,
                            feature_codec.scales,
                        )
                    }
                    record = run_bounded_adam(
                        control_model,
                        parameter_names,
                        lambda backward, detector=control_model, cache=cache:
                        cached_detection_loss_tensor(
                            detector,
                            cache,
                            model_device=self.device,
                            backward=backward,
                        ),
                        bounds=feature_bounds,
                    )
                else:
                    record = run_head_adam(
                        control_model,
                        lambda backward, detector=control_model, cache=cache:
                        cached_detection_loss_tensor(
                            detector,
                            cache,
                            model_device=self.device,
                            backward=backward,
                        ),
                    )
                selection_metrics, _ = evaluate_detection_records(
                    control_model,
                    selection_records,
                    device=self.device,
                )
                checkpoint = (
                    self.root / "arms" / str(base_seed) / f"{method}.pt"
                )
                atomic_write_bytes(
                    checkpoint,
                    _torch_save_bytes(control_model.state_dict()),
                )
                record.update(
                    {
                        "base_seed": base_seed,
                        "selection_metrics": selection_metrics,
                        "checkpoint": str(checkpoint.relative_to(self.run_root)),
                        "checkpoint_hash": fingerprint_file(checkpoint),
                    }
                )
                self.result["arms"].setdefault(method, {})[
                    str(base_seed)
                ] = record
                arm_record[method] = record
                self._save()
                del control_model
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            selected: dict[str, Any] = {}
            for method in ("feature_pso", "feature_random"):
                records = self.result["arms"][method][str(base_seed)]
                winner = max(
                    records.values(),
                    key=lambda record: (
                        float(record["selection_metrics"]["map50_95"]),
                        -int(record["swarm_seed"]),
                    ),
                )
                selected[method] = {
                    "swarm_seed": int(winner["swarm_seed"]),
                    "selection_map50_95": float(
                        winner["selection_metrics"]["map50_95"]
                    ),
                    "best_residual": list(winner["selected_residual"]),
                    "generation": int(winner["selected_generation"]),
                    "checkpoint": winner["checkpoint"],
                    "checkpoint_hash": winner["checkpoint_hash"],
                }
            self.result["development_selection"][str(base_seed)] = selected
            arm_path = self.root / "arms" / str(base_seed) / "record.json"
            atomic_write_json(arm_path, arm_record)
            del detector, cache, objective
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        pool_records = objective_records + selection_records
        member_rows = [[] for _ in pool_records]
        for baseline in baselines.values():
            detector = make_yolo11n(device=self.device).model
            detector.load_state_dict(
                torch.load(
                    _resolve_run_path(
                        self.run_root,
                        baseline["checkpoint"],
                    ),
                    map_location=self.device,
                    weights_only=True,
                ),
                strict=True,
            )
            detector.eval()
            for start in range(0, len(pool_records), 4):
                subset = pool_records[start : start + 4]
                images, _, ratio_pad = native_batch(
                    subset,
                    device=self.device,
                )
                outputs = _native_predictions(detector, images)
                for offset, (record, output, padding) in enumerate(
                    zip(subset, outputs, ratio_pad)
                ):
                    boxes = (
                        output[:, :4].detach().cpu().numpy()
                        if output.numel()
                        else np.empty((0, 4))
                    )
                    boxes = transform_boxes_to_original(
                        boxes,
                        ratio_pad=padding,
                        shape=(record.height, record.width),
                    )
                    boxes[:, [0, 2]] /= record.width
                    boxes[:, [1, 3]] /= record.height
                    member_rows[start + offset].append(
                        {
                            "boxes": boxes,
                            "scores": (
                                output[:, 4].detach().cpu().numpy()
                                if output.numel()
                                else np.empty((0,))
                            ),
                            "labels": (
                                output[:, 5]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.int64)
                                if output.numel()
                                else np.empty((0,), dtype=np.int64)
                            ),
                        }
                    )
        member_predictions = [
            [member_rows[index][member] for index in range(len(pool_records))]
            for member in range(len(baselines))
        ]
        target_rows = [
            {
                "image_id": f"{record.year}:{record.image_id}",
                "labels": [int(label[0]) for label in record.labels],
                "boxes": [
                    [
                        float(label[1] - label[3] / 2),
                        float(label[2] - label[4] / 2),
                        float(label[1] + label[3] / 2),
                        float(label[2] + label[4] / 2),
                    ]
                    for label in record.labels
                ],
            }
            for record in pool_records
        ]
        objective_members = [
            rows[:OBJECTIVE_COUNT] for rows in member_predictions
        ]
        selection_members = [
            rows[OBJECTIVE_COUNT:] for rows in member_predictions
        ]
        objective_targets = target_rows[:OBJECTIVE_COUNT]
        selection_targets = target_rows[OBJECTIVE_COUNT:]
        uniform_weights = [1.0 / len(baselines)] * len(baselines)
        uniform_objective = _wbf_dataset_metrics(
            objective_members, objective_targets, uniform_weights
        )
        uniform_selection = _wbf_dataset_metrics(
            selection_members, selection_targets, uniform_weights
        )
        ensemble_pso = [
            run_wbf_weight_search(
                objective_members,
                objective_targets,
                seed=seed,
                random_mode=False,
            )
            for seed in SWARM_SEEDS
        ]
        ensemble_random = [
            run_wbf_weight_search(
                objective_members,
                objective_targets,
                seed=seed,
                random_mode=True,
            )
            for seed in SWARM_SEEDS
        ]
        for record in ensemble_pso + ensemble_random:
            record["selection_metrics"] = _wbf_dataset_metrics(
                selection_members,
                selection_targets,
                record["weights"],
            )
        selected_pso = max(
            ensemble_pso,
            key=lambda record: (
                float(record["selection_metrics"]["map50_95"]),
                -int(record["seed"]),
            ),
        )
        selected_random = max(
            ensemble_random,
            key=lambda record: (
                float(record["selection_metrics"]["map50_95"]),
                -int(record["seed"]),
            ),
        )
        fused = [
            weighted_box_fusion(rows, uniform_weights)
            for rows in member_rows
        ]
        ensemble_path = self.root / "ensemble_uniform_wbf.json"
        atomic_write_json(
            ensemble_path,
            [{key: value.tolist() for key, value in row.items()} for row in fused],
        )
        self.result["ensemble"] = {
            "uniform_wbf": {
                "path": str(ensemble_path.relative_to(self.run_root)),
                "weights": uniform_weights,
                "objective_metrics": uniform_objective,
                "selection_metrics": uniform_selection,
            },
            "ensemble_pso": ensemble_pso,
            "ensemble_random": ensemble_random,
        }
        self.result.setdefault("development_selection", {})["ensemble"] = {
            "ensemble_pso": {
                "seed": int(selected_pso["seed"]),
                "weights": list(selected_pso["weights"]),
                "selection_map50_95": float(
                    selected_pso["selection_metrics"]["map50_95"]
                ),
            },
            "ensemble_random": {
                "seed": int(selected_random["seed"]),
                "weights": list(selected_random["weights"]),
                "selection_map50_95": float(
                    selected_random["selection_metrics"]["map50_95"]
                ),
            },
        }
        self.result["baselines"] = baselines
        artifact_paths = [
            _resolve_run_path(self.run_root, item["checkpoint"])
            for item in baselines.values()
        ] + [
            ensemble_path,
            self.root / "voc_manifest.json",
            self.root / "voc_study.yaml",
        ]
        artifact_paths.extend(
            path for path in (self.root / "arms").rglob("*") if path.is_file()
        )
        self.result["artifact_hashes"] = {
            str(path.relative_to(self.run_root)): fingerprint_file(path)
            for path in artifact_paths
            if path.is_file()
        }
        primary_queries = (
            len(BASE_SEEDS)
            * len(SWARM_SEEDS)
            * PARTICLE_COUNT
            * PSO_GENERATIONS
        )
        ensemble_queries = (
            len(SWARM_SEEDS)
            * WBF_PARTICLE_COUNT
            * WBF_GENERATIONS
        )
        self.result["resource_ledger"] = {
            "baseline_count": len(baselines),
            "objective_samples": len(objective_records),
            "selection_samples": len(selection_records),
            "primary_pso_queries": primary_queries,
            "primary_random_queries": primary_queries,
            "ensemble_pso_queries": ensemble_queries,
            "ensemble_random_queries": ensemble_queries,
            "pso_candidate_samples": (
                primary_queries + ensemble_queries
            ) * len(objective_records),
            "random_candidate_samples": (
                primary_queries + ensemble_queries
            ) * len(objective_records),
            "pso_cells": len(BASE_SEEDS) * len(SWARM_SEEDS),
        }
        self.result["integrity"].update(
            {"developed": True, "official_test_opened": False}
        )
        self._save()
        return self.result
    def confirm(self) -> dict[str, Any]:
        state = load_state(self.run_root)
        if state.state != StudyState.FROZEN:
            raise SealError("confirm requires a frozen complete development matrix")
        begin_confirmation(self.run_root, state)
        atomic_write_json(self.run_root / "state.json", state.to_dict())
        records, guard = guarded_voc_test_loader(
            self.data_root,
            self.run_root,
            confirmation=True,
        )
        self.result["leakage_counters"]["official_test_construction"] += 1
        predictions_root = self.root / "confirmation_predictions"
        predictions_root.mkdir(parents=True, exist_ok=True)
        confirmation: dict[str, Any] = {
            "test_records": len(records),
            "manifest_hash": guard.frozen_manifest_hash,
        }
        base_rows: dict[str, list[dict[str, Any]]] = {}

        def evaluate_checkpoint(
            method: str,
            base_seed: str,
            checkpoint: str,
        ) -> list[dict[str, Any]]:
            detector = make_yolo11n(device=self.device).model
            detector.load_state_dict(
                torch.load(
                    self.run_root / checkpoint
                    if not Path(checkpoint).is_absolute()
                    else checkpoint,
                    map_location=self.device,
                    weights_only=True,
                ),
                strict=True,
            )
            metrics, rows = evaluate_detection_records(
                detector,
                records,
                device=self.device,
            )
            self.result["leakage_counters"][
                "official_test_forward_passes"
            ] += 1
            destination = predictions_root / f"{method}_{base_seed}.json"
            atomic_write_json(destination, rows)
            confirmation.setdefault(method, {})[base_seed] = {
                "predictions": str(destination.relative_to(self.run_root)),
                "metrics": metrics,
                "images": len(rows),
            }
            return rows

        for base_seed, baseline in sorted(
            self.result.get("baselines", {}).items()
        ):
            base_rows[base_seed] = evaluate_checkpoint(
                "base",
                base_seed,
                baseline["checkpoint"],
            )
            selected = self.result["development_selection"][base_seed]
            for method in ("feature_pso", "feature_random"):
                swarm_seed = str(selected[method]["swarm_seed"])
                checkpoint = self.result["arms"][method][base_seed][
                    swarm_seed
                ]["checkpoint"]
                evaluate_checkpoint(method, base_seed, checkpoint)
            for method in ("feature_adam", "head_adam"):
                checkpoint = self.result["arms"][method][base_seed][
                    "checkpoint"
                ]
                evaluate_checkpoint(method, base_seed, checkpoint)

        base_seed_order = [str(seed) for seed in BASE_SEEDS]
        member_predictions: list[list[dict[str, np.ndarray]]] = []
        for base_seed in base_seed_order:
            member: list[dict[str, np.ndarray]] = []
            for record, row in zip(records, base_rows[base_seed]):
                boxes = np.asarray(
                    [item["box"] for item in row["predictions"]],
                    dtype=np.float64,
                ).reshape(-1, 4)
                if boxes.size:
                    boxes[:, [0, 2]] /= record.width
                    boxes[:, [1, 3]] /= record.height
                member.append(
                    {
                        "boxes": boxes,
                        "scores": np.asarray(
                            [item["score"] for item in row["predictions"]],
                            dtype=np.float64,
                        ),
                        "labels": np.asarray(
                            [item["class_id"] for item in row["predictions"]],
                            dtype=np.int64,
                        ),
                    }
                )
            member_predictions.append(member)
        target_rows = []
        for record, row in zip(records, base_rows[base_seed_order[0]]):
            boxes = np.asarray(
                [item["box"] for item in row["ground_truth"]],
                dtype=np.float64,
            ).reshape(-1, 4)
            if boxes.size:
                boxes[:, [0, 2]] /= record.width
                boxes[:, [1, 3]] /= record.height
            target_rows.append(
                {
                    "image_id": row["image_id"],
                    "boxes": boxes.tolist(),
                    "labels": [
                        int(item["class_id"])
                        for item in row["ground_truth"]
                    ],
                }
            )

        weights_by_method = {
            "uniform_wbf": [1.0 / len(BASE_SEEDS)] * len(BASE_SEEDS),
            "ensemble_pso": self.result["development_selection"]["ensemble"][
                "ensemble_pso"
            ]["weights"],
            "ensemble_random": self.result["development_selection"][
                "ensemble"
            ]["ensemble_random"]["weights"],
        }
        for method, weights in weights_by_method.items():
            fused_rows = []
            for index, record in enumerate(records):
                fused = weighted_box_fusion(
                    [member[index] for member in member_predictions],
                    weights,
                )
                boxes = fused["boxes"].copy()
                if boxes.size:
                    boxes[:, [0, 2]] *= record.width
                    boxes[:, [1, 3]] *= record.height
                fused_rows.append(
                    {
                        "image_id": target_rows[index]["image_id"],
                        "predictions": [
                            {
                                "box": [float(value) for value in box],
                                "class_id": int(label),
                                "score": float(score),
                            }
                            for box, label, score in zip(
                                boxes.tolist(),
                                fused["labels"].tolist(),
                                fused["scores"].tolist(),
                            )
                        ],
                        "ground_truth": base_rows[
                            base_seed_order[0]
                        ][index]["ground_truth"],
                    }
                )
            destination = predictions_root / f"{method}.json"
            atomic_write_json(destination, fused_rows)
            confirmation[method] = {
                "predictions": str(destination.relative_to(self.run_root)),
                "metrics": _wbf_dataset_metrics(
                    member_predictions,
                    target_rows,
                    weights,
                ),
                "weights": [float(weight) for weight in weights],
                "images": len(fused_rows),
            }
        self.result["confirmation"] = confirmation
        self.result["artifact_hashes"].update(
            {
                str(path.relative_to(self.run_root)): fingerprint_file(path)
                for path in predictions_root.glob("*.json")
            }
        )
        finish_confirmation(state, success=True)
        atomic_write_json(self.run_root / "state.json", state.to_dict())
        self._save()
        return self.result
    def run_phase(self, phase: str) -> dict[str, Any]:
        if phase == "prepare": return self.prepare()
        if phase == "smoke": return self.smoke()
        if phase == "develop": return self.develop()
        if phase == "confirm": return self.confirm()
        if phase == "publish":
            if not self.result_path.is_file(): raise SealError("cannot publish without adapter result")
            return json.loads(self.result_path.read_text(encoding="utf-8"))
        raise YoloProtocolError(f"unsupported adapter phase: {phase}")


def create_adapter(*, workload_id: str = WORKLOAD_ID, config: StudyConfig, run_root: str | os.PathLike[str], data_root: str | os.PathLike[str], device: str | torch.device, allow_download: bool = False) -> YoloConvergenceAdapter:
    return YoloConvergenceAdapter(workload_id=workload_id, config=config, run_root=run_root, data_root=data_root, device=device, allow_download=allow_download)


__all__ = [
    "BASE_SEEDS", "DetectionCache", "FAMILY", "IMG_SIZE", "OBJECTIVE_COUNT",
    "PROJECTION_SEED", "StrictScratchTrainer", "ULTRALYTICS_VERSION",
    "VOC_CLASSES", "VOCManifest", "VOCRecord", "VOCTestGuard", "WORKLOAD_ID",
    "YoloConvergenceAdapter", "YoloProtocolError", "assert_yolo_topology",
    "build_detection_cache", "cached_detection_objective", "cached_full_parity",
    "create_adapter", "guarded_voc_test_loader", "image_fingerprint",
    "make_voc_manifests", "make_yolo11n", "native_detection_metrics",
    "parse_voc_xml", "pinned_preflight", "prepare_voc", "run_bounded_adam",
    "run_feature_search", "run_head_adam", "selected_block_names",
    "selected_head_bias_names", "train_baseline", "transform_boxes_to_letterbox",
    "transform_boxes_to_original", "weighted_box_fusion", "write_study_yaml",
    "write_yolo_label",
]
