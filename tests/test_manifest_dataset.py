from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image
import pytest

from rs_hmcomm.datasets.manifest import ManifestDatasetError, iter_manifest_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color).save(path)


def _write_manifest(path: Path, rows: list[dict[str, object] | str]) -> None:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(sample_id: str, image: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "image": image,
        "question": f"question-{sample_id}",
        "answer": f"answer-{sample_id}",
    }


def test_manifest_preserves_order_and_loads_detached_images(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "dataset"
    manifest_dir.mkdir()
    _write_image(manifest_dir / "images" / "one.png", (255, 0, 0))
    _write_image(manifest_dir / "images" / "two.png", (0, 255, 0))
    manifest = manifest_dir / "samples.jsonl"
    first = _row("one", "images/one.png")
    first.update(
        {
            "task_type": "counting",
            "bbox": [0, 1, 2, 3],
            "metadata": {"split": "tiny"},
        }
    )
    _write_manifest(manifest, [first, _row("two", "images/two.png")])

    samples = list(iter_manifest_dataset(manifest))

    assert [sample.sample_id for sample in samples] == ["one", "two"]
    assert samples[0].task_type == "counting"
    assert samples[0].bbox == (0.0, 1.0, 2.0, 3.0)
    assert samples[0].metadata == {"split": "tiny"}
    assert samples[1].task_type == "vqa"
    assert samples[1].metadata is None
    assert samples[0].image.mode == "RGB"
    assert samples[0].image.getpixel((0, 0)) == (255, 0, 0)

    # This succeeds on Windows only when Pillow has released the source handle.
    (manifest_dir / "images" / "one.png").unlink()
    assert samples[0].image.getpixel((0, 0)) == (255, 0, 0)


def test_manifest_limit_is_positive_and_preserves_prefix(tmp_path: Path) -> None:
    for index in range(3):
        _write_image(tmp_path / f"{index}.png", (index, index, index))
    manifest = tmp_path / "samples.jsonl"
    _write_manifest(manifest, [_row(str(index), f"{index}.png") for index in range(3)])

    assert [sample.sample_id for sample in iter_manifest_dataset(manifest, limit=2)] == ["0", "1"]
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_manifest_dataset(manifest, limit=0))
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_manifest_dataset(manifest, limit=True))


@pytest.mark.parametrize(
    "bad_row, expected",
    [
        ("{not-json}", "invalid JSON"),
        ({"sample_id": "x", "image": "x.png", "question": "", "answer": "a"}, "question"),
        (
            {"sample_id": "x", "image": "x.png", "question": "q", "answer": "a", "extra": 1},
            "unknown fields",
        ),
    ],
)
def test_manifest_rejects_bad_lines(tmp_path: Path, bad_row: dict[str, object] | str, expected: str) -> None:
    _write_image(tmp_path / "x.png", (0, 0, 0))
    manifest = tmp_path / "samples.jsonl"
    _write_manifest(manifest, [bad_row])

    with pytest.raises(ManifestDatasetError, match=expected):
        list(iter_manifest_dataset(manifest))


def test_manifest_rejects_parent_path_escape(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "dataset"
    manifest_dir.mkdir()
    _write_image(tmp_path / "outside.png", (0, 0, 0))
    manifest = manifest_dir / "samples.jsonl"
    _write_manifest(manifest, [_row("escape", "../outside.png")])

    with pytest.raises(ManifestDatasetError, match="must not contain"):
        list(iter_manifest_dataset(manifest))


def test_manifest_rejects_symlink_escape(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "dataset"
    manifest_dir.mkdir()
    outside = tmp_path / "outside.png"
    _write_image(outside, (0, 0, 0))
    link = manifest_dir / "linked.png"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted on this platform")
    manifest = manifest_dir / "samples.jsonl"
    _write_manifest(manifest, [_row("escape", "linked.png")])

    with pytest.raises(ManifestDatasetError, match="outside the manifest directory"):
        list(iter_manifest_dataset(manifest))


def test_manifest_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    _write_image(tmp_path / "one.png", (0, 0, 0))
    _write_image(tmp_path / "two.png", (0, 0, 0))
    manifest = tmp_path / "samples.jsonl"
    _write_manifest(manifest, [_row("duplicate", "one.png"), _row("duplicate", "two.png")])

    with pytest.raises(ManifestDatasetError, match="duplicate sample_id"):
        list(iter_manifest_dataset(manifest))
