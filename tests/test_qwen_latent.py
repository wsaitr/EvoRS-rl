"""Offline tests for QwenMmprojLatentEncoder (mock HTTP, no server needed)."""
from __future__ import annotations

import io
import json
import math
import struct
from typing import Any

import pytest
from PIL import Image

from rs_hmcomm.artifacts import InMemoryArtifactStore
from rs_hmcomm.artifacts.qwen_latent import (
    LatentConsumer,
    QwenMmprojLatentEncoder,
)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def _install_fake_server(monkeypatch: Any, dim: int = 16) -> list[dict]:
    """Patch urllib to return a fixed embedding; record requests."""
    requests: list[dict] = []

    def fake_urlopen(request, timeout=None):
        payload = json.loads(request.data)
        requests.append(payload)
        vector = [0.5] * dim
        response = FakeResponse()
        response.body = [{"embedding": [vector]}]
        return response

    import rs_hmcomm.artifacts.qwen_latent as module

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    return requests


class TestQwenMmprojLatentEncoder:
    def test_encode_full_image(self, monkeypatch):
        store = InMemoryArtifactStore()
        requests = _install_fake_server(monkeypatch)
        encoder = QwenMmprojLatentEncoder(store, max_side=64, expected_dim=16)

        image = Image.new("RGB", (128, 128), (10, 20, 30))
        artifact = encoder.encode(
            image,
            artifact_id="lat-1",
            source_image_ref="test://img",
            compatible_consumers=("reasoning",),
        )

        assert artifact.shape == (16,)
        assert artifact.byte_size == 16 * 4
        assert artifact.roi is None
        assert len(requests) == 1
        assert "image_data" in requests[0]

    def test_encode_roi_crops_region(self, monkeypatch):
        store = InMemoryArtifactStore()
        _install_fake_server(monkeypatch)
        encoder = QwenMmprojLatentEncoder(store, max_side=64, expected_dim=16)

        image = Image.new("RGB", (100, 100))
        artifact = encoder.encode(
            image,
            artifact_id="lat-roi",
            source_image_ref="test://img",
            roi=(0.25, 0.25, 0.75, 0.75),
        )
        assert artifact.roi == (0.25, 0.25, 0.75, 0.75)

    def test_invalid_roi_rejected(self, monkeypatch):
        store = InMemoryArtifactStore()
        _install_fake_server(monkeypatch)
        encoder = QwenMmprojLatentEncoder(store)
        with pytest.raises(ValueError):
            encoder.encode(
                Image.new("RGB", (32, 32)),
                artifact_id="x",
                source_image_ref="test://",
                roi=(0.5, 0.5, 0.5, 0.5),
            )

    def test_unexpected_dimension_raises(self, monkeypatch):
        store = InMemoryArtifactStore()
        _install_fake_server(monkeypatch, dim=8)  # wrong on purpose
        encoder = QwenMmprojLatentEncoder(store)
        with pytest.raises(RuntimeError, match="dimension"):
            encoder.encode(
                Image.new("RGB", (32, 32)),
                artifact_id="y",
                source_image_ref="test://",
            )


class TestLatentConsumer:
    def _put_small(self, store, consumers=("reasoning",)):
        from rs_hmcomm.artifacts.store import ArtifactKind, VisualArtifact

        vector = [0.1] * 8
        data = struct.pack("<8f", *vector)
        store.put(
            VisualArtifact(
                artifact_id="small",
                kind=ArtifactKind.LATENT,
                media_type="application/x-evors-float32-le",
                shape=(8,),
                dtype="float32",
                byte_size=len(data),
                source_image_ref="t://",
                roi=None,
                encoder="e",
                layer="l",
                projector="p",
                compatible_consumers=tuple(consumers),
                sha256=__import__("hashlib").sha256(data).hexdigest(),
            ),
            data,
        )

    def test_roundtrip_values(self):
        store = InMemoryArtifactStore()
        self._put_small(store)
        values = LatentConsumer(store, "reasoning").consume("small")
        assert len(values) == 8
        assert all(abs(v - 0.1) < 1e-6 for v in values)

    def test_incompatible_consumer_blocked(self):
        store = InMemoryArtifactStore()
        self._put_small(store, consumers=("reasoning",))
        with pytest.raises(PermissionError):
            LatentConsumer(store, "verification").consume("small")
