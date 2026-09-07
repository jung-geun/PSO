"""Offline behavioral tests for the pinned VOC/YOLO convergence adapter."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test import post_training_yolo_convergence as study
from test.post_training_model_convergence import SealError, StudyConfig, prepare_run


def _record(index: int, *, fingerprint: str | None = None) -> study.VOCRecord:
    """Build a cheap, label-complete synthetic record for manifest tests."""
    labels = tuple((class_id, 0.5, 0.5, 0.25, 0.25) for class_id in range(20))
    return study.VOCRecord(
        year="2007" if index % 2 == 0 else "2012",
        image_id=f"item-{index:05d}",
        image_path=f"/synthetic/{index}.jpg",
        annotation_path=f"/synthetic/{index}.xml",
        width=640,
        height=480,
        labels=labels,
        difficult_excluded=0,
        fingerprint=fingerprint or f"{index:064x}",
    )


def test_optional_detection_imports_are_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the adapter stays safe when optional detection packages are absent."""
    real_import = builtins.__import__

    def block_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("blocked optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_ultralytics)
    with pytest.raises(study.YoloProtocolError, match="Ultralytics is required"):
        study._ultralytics()

    def block_ensemble_boxes(name, *args, **kwargs):
        if name == "ensemble_boxes":
            raise ImportError("blocked optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_ensemble_boxes)
    with pytest.raises(study.YoloProtocolError, match="ensemble-boxes is required"):
        study._wbf()


def test_parse_voc_xml_excludes_difficult_and_uses_pinned_coordinates(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(image_path)
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        """<annotation>
        <size><width>20</width><height>20</height><depth>3</depth></size>
        <object><name>cat</name><difficult>0</difficult>
          <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>9</xmax><ymax>10</ymax></bndbox>
        </object>
        <object><name>dog</name><difficult>1</difficult>
          <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>19</xmax><ymax>19</ymax></bndbox>
        </object>
        </annotation>""",
        encoding="utf-8",
    )

    record = study.parse_voc_xml(xml_path, image_path, year="2007", image_id="sample")

    assert record.width == 20 and record.height == 20
    assert record.difficult_excluded == 1
    assert record.labels == ((7, 0.2, 0.25, 0.4, 0.4),)
    assert record.fingerprint == study.image_fingerprint(image_path)


def test_duplicate_grouping_keeps_group_members_together() -> None:
    duplicate_a = _record(0, fingerprint="same")
    duplicate_b = _record(1, fingerprint="same")
    unique = _record(2, fingerprint="unique")

    ordered, groups = study._assign_duplicate_groups(
        [duplicate_a, duplicate_b, unique], seed=20260908
    )

    assert groups["same"] == tuple(f"{item.year}:{item.image_id}" for item in ordered if item.fingerprint == "same")
    same_positions = [index for index, item in enumerate(ordered) if item.fingerprint == "same"]
    assert same_positions == list(range(min(same_positions), max(same_positions) + 1))
    assert {item.image_id for item in ordered} == {"item-00000", "item-00001", "item-00002"}


def test_manifest_has_exact_disjoint_partitions_and_objective_prefix() -> None:
    records = [_record(index) for index in range(16551)]
    manifest = study.make_voc_manifests(records, seed=20260908)

    assert manifest.counts == {
        "bp_train": study.BP_COUNT,
        "refine_search": study.REFINE_COUNT,
        "selection_val": study.SELECTION_COUNT,
    }
    partitions = (manifest.bp_train, manifest.refine_search, manifest.selection_val)
    keys = [
        {f"{item.year}:{item.image_id}" for item in partition}
        for partition in partitions
    ]
    assert [len(partition) for partition in partitions] == [11551, 2500, 2500]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert manifest.objective_keys == tuple(
        (item.year, item.image_id) for item in manifest.refine_search[: study.OBJECTIVE_COUNT]
    )
    for partition in partitions:
        assert {label[0] for item in partition for label in item.labels} == set(range(20))


def test_letterbox_box_round_trip_preserves_original_coordinates() -> None:
    original = np.array([[10.0, 5.0, 190.0, 95.0, 0.87]], dtype=np.float64)
    ratio_pad = (3.2, (0.0, 160.0))  # 200x100 image letterboxed to 640x640

    letterboxed = study.transform_boxes_to_letterbox(original, ratio_pad=ratio_pad)
    restored = study.transform_boxes_to_original(
        letterboxed, ratio_pad=ratio_pad, shape=(100, 200)
    )

    assert np.allclose(restored, original, atol=1e-12)
    assert np.allclose(letterboxed[0, :4], [32.0, 176.0, 608.0, 464.0])
    clipped = study.transform_boxes_to_original(
        np.array([[-10.0, 150.0, 650.0, 500.0]]),
        ratio_pad=ratio_pad,
        shape=(100, 200),
    )
    assert np.array_equal(clipped, np.array([[0.0, 0.0, 200.0, 100.0]]))

def test_native_target_uses_non_square_letterbox_geometry() -> None:
    record = study.VOCRecord(
        year="2007",
        image_id="wide",
        image_path="/synthetic/wide.jpg",
        annotation_path="/synthetic/wide.xml",
        width=200,
        height=100,
        labels=((0, 0.5, 0.5, 0.5, 0.5),),
        difficult_excluded=0,
        fingerprint="a" * 64,
    )
    target = study._native_target(
        record,
        index=3,
        ratio_pad=(3.2, (0.0, 160.0)),
    )
    assert target["batch_idx"].tolist() == [3]
    assert target["cls"].tolist() == [[0.0]]
    assert np.allclose(
        target["bboxes"].numpy(),
        np.array([[0.5, 0.5, 0.5, 0.25]]),
    )


def test_wbf_uses_normalized_weights_and_stable_score_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_fusion(boxes, scores, labels, **kwargs):
        captured["weights"] = kwargs["weights"]
        captured["kwargs"] = kwargs
        return (
            [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]],
            [0.20, 0.90, 0.50],
            [2, 1, 0],
        )

    monkeypatch.setattr(study, "_wbf", lambda: fake_fusion)
    result = study.weighted_box_fusion(
        [
            {"boxes": np.array([[0.1, 0.1, 0.2, 0.2]]), "scores": [0.8], "labels": [2]},
            {"boxes": np.array([[0.3, 0.3, 0.4, 0.4]]), "scores": [0.7], "labels": [1]},
        ],
        [2.0, 6.0],
    )

    assert captured["weights"] == pytest.approx([0.25, 0.75])
    assert sum(captured["weights"]) == pytest.approx(1.0)
    assert captured["kwargs"]["iou_thr"] == 0.55
    assert result["scores"].tolist() == [0.90, 0.50, 0.20]
    assert result["labels"].tolist() == [1, 0, 2]


def test_wbf_pso_and_random_have_exact_12x20_query_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, ...]] = []

    def tiny_metric(member_predictions, targets, weights):
        values = tuple(float(value) for value in weights)
        calls.append(values)
        assert sum(values) == pytest.approx(1.0)
        return {"map50_95": values[0]}

    monkeypatch.setattr(study, "_wbf_dataset_metrics", tiny_metric)
    targets = [{"image_id": "a"}, {"image_id": "b"}]
    members = [[{} for _ in targets] for _ in range(3)]

    pso = study.run_wbf_weight_search(members, targets, seed=601, random_mode=False)
    random_result = study.run_wbf_weight_search(members, targets, seed=601, random_mode=True)

    assert pso["method"] == "ensemble_pso"
    assert random_result["method"] == "ensemble_random"
    for result in (pso, random_result):
        assert result["queries"] == 12 * 20
        assert result["sample_evaluations"] == 12 * 20 * len(targets)
        assert len(result["trajectory"]) == 20
        assert sum(result["weights"]) == pytest.approx(1.0)
        assert all(0.0 <= weight <= 1.0 for weight in result["weights"])
    assert len(calls) == 2 * 12 * 20


class C3k2(nn.Module):
    pass


class Detect(nn.Module):
    def __init__(self, nc: int = 20) -> None:
        super().__init__()
        self.nc = nc
        self.cv2 = nn.ModuleList([nn.Sequential(nn.Linear(1, 42)) for _ in range(3)])
        self.cv3 = nn.ModuleList([nn.Sequential(nn.Linear(1, 42)) for _ in range(3)])


class WrongDetect(Detect):
    pass


class WrongBlock(nn.Module):
    pass


def _tiny_graph(*, block: nn.Module | None = None, detect: nn.Module | None = None) -> nn.Module:
    graph = nn.Module()
    graph.model = nn.ModuleList([nn.Identity() for _ in range(22)] + [block or C3k2(), detect or Detect()])
    return graph


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (nn.Module(), "shorter than"),
        (_tiny_graph(block=WrongBlock()), "expected model.22 C3k2"),
        (_tiny_graph(detect=WrongDetect()), "expected model.23 Detect"),
        (_tiny_graph(detect=Detect(nc=19)), "expected Detect.nc=20"),
    ],
)
def test_topology_guard_rejects_tiny_mismatched_graphs(graph: nn.Module, message: str) -> None:
    if not hasattr(graph, "model"):
        graph.model = nn.ModuleList([nn.Identity(), nn.Identity()])
    with pytest.raises(study.YoloProtocolError, match=message):
        study.assert_yolo_topology(graph)


def test_official_test_loader_refuses_before_frozen_confirmation(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    data_root = tmp_path / "data"
    state = prepare_run(run_root, StudyConfig(device="cpu"))

    with pytest.raises(SealError, match="sealed until confirm phase"):
        study.guarded_voc_test_loader(data_root, run_root, confirmation=False)
    assert state.state.value == "prepared"
    assert not (data_root / "VOCdevkit").exists()

    with pytest.raises(SealError, match="sealed until frozen confirmation"):
        study.VOCTestGuard(str(run_root), "", False).require_open()


def test_native_baseline_reuse_verifies_complete_epoch_artifacts(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    baseline_root = (
        run_root
        / "workloads"
        / study.WORKLOAD_ID
        / "baselines"
        / "501"
    )
    baseline_root.mkdir(parents=True)
    checkpoint = baseline_root / "ema_fp32.pt"
    torch.save({"weight": torch.ones(2)}, checkpoint)
    results = (
        run_root
        / "ultralytics"
        / "base-501-100e"
        / "results.csv"
    )
    results.parent.mkdir(parents=True)
    results.write_text(
        "epoch,train/loss\n"
        + "".join(f"{epoch},{1 / epoch}\n" for epoch in range(1, 101)),
        encoding="utf-8",
    )
    marker = {
        "protocol_version": study.PROTOCOL_VERSION,
        "source_run": "source",
        "checkpoint_hash": study.fingerprint_file(checkpoint),
        "results_hash": study.fingerprint_file(results),
    }
    (baseline_root / "baseline_reuse.json").write_text(
        json.dumps(marker),
        encoding="utf-8",
    )

    reused = study._reused_native_baseline(
        run_root,
        study.StrictScratchTrainer(device="cpu"),
        501,
    )
    assert reused is not None
    assert reused["reused"] is True
    assert len(reused["telemetry"]) == 11
    results.write_text("epoch,train/loss\n1,1\n", encoding="utf-8")
    with pytest.raises(study.YoloProtocolError, match="reused baseline marker"):
        study._reused_native_baseline(
            run_root,
            study.StrictScratchTrainer(device="cpu"),
            501,
        )
