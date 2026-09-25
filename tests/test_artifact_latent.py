from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from PIL import Image
import pytest

from rs_hmcomm.artifacts import (
    ArtifactIntegrityError,
    ArtifactKind,
    InMemoryArtifactStore,
    IncompatibleLatentConsumerError,
    LatentConsumer,
    LatentFeatureConfig,
    LatentFeatureDisabledError,
    PIXEL_STATS_FEATURE_COUNT,
    PixelStatsLatentEncoder,
    VisualArtifact,
)


ENABLED = LatentFeatureConfig(enabled=True)


def test_latent_feature_is_disabled_by_default() -> None:
    store = InMemoryArtifactStore()
    encoder = PixelStatsLatentEncoder(store)

    with pytest.raises(LatentFeatureDisabledError):
        encoder.encode(
            Image.new("RGB", (4, 4), "red"),
            artifact_id="latent-disabled",
            source_image_ref="task-image",
        )

    assert len(store) == 0


def test_real_pixels_round_trip_through_store_and_compatible_consumer() -> None:
    store = InMemoryArtifactStore()
    encoder = PixelStatsLatentEncoder(store, ENABLED)
    artifact = encoder.encode(
        Image.new("RGB", (8, 6), (255, 0, 0)),
        artifact_id="latent-red",
        source_image_ref="task://red-image",
        compatible_consumers=("reasoning", "verification"),
    )

    assert artifact.kind == ArtifactKind.LATENT
    assert artifact.shape == (PIXEL_STATS_FEATURE_COUNT,)
    assert artifact.dtype == "float32"
    assert artifact.byte_size == PIXEL_STATS_FEATURE_COUNT * 4
    assert artifact.sha256 == hashlib.sha256(store.get("latent-red").data).hexdigest()

    descriptor = artifact.descriptor()
    encoded_descriptor = json.dumps(descriptor, sort_keys=True)
    assert set(descriptor) == {
        "artifact_id",
        "kind",
        "media_type",
        "shape",
        "dtype",
        "byte_size",
        "source_image_ref",
        "roi",
        "encoder",
        "layer",
        "projector",
        "compatible_consumers",
        "sha256",
    }
    assert "data" not in descriptor
    assert "payload" not in descriptor
    assert "255, 0, 0" not in encoded_descriptor

    result = LatentConsumer(store, "reasoning", ENABLED).consume("latent-red")
    assert result.bytes_loaded == artifact.byte_size
    assert len(result.values) == PIXEL_STATS_FEATURE_COUNT
    assert result.values[:3] == pytest.approx((1.0, 0.0, 0.0))


def test_latent_values_and_hash_change_when_image_pixels_change() -> None:
    store = InMemoryArtifactStore()
    encoder = PixelStatsLatentEncoder(store, ENABLED)
    red = encoder.encode(
        Image.new("RGB", (4, 4), (255, 0, 0)),
        artifact_id="latent-red",
        source_image_ref="task://counterfactual",
        compatible_consumers=("reasoning",),
    )
    blue = encoder.encode(
        Image.new("RGB", (4, 4), (0, 0, 255)),
        artifact_id="latent-blue",
        source_image_ref="task://counterfactual",
        compatible_consumers=("reasoning",),
    )
    red_repeat = encoder.encode(
        Image.new("RGB", (4, 4), (255, 0, 0)),
        artifact_id="latent-red-repeat",
        source_image_ref="task://counterfactual",
        compatible_consumers=("reasoning",),
    )

    consumer = LatentConsumer(store, "reasoning", ENABLED)
    red_values = consumer.consume(red.artifact_id).values
    blue_values = consumer.consume(blue.artifact_id).values

    assert red_values != blue_values
    assert red.sha256 != blue.sha256
    assert consumer.consume(red_repeat.artifact_id).values == red_values
    assert red_repeat.sha256 == red.sha256


def test_roi_latent_uses_only_the_requested_image_region() -> None:
    image = Image.new("RGB", (8, 4), "red")
    for x in range(4, 8):
        for y in range(4):
            image.putpixel((x, y), (0, 0, 255))

    store = InMemoryArtifactStore()
    encoder = PixelStatsLatentEncoder(store, ENABLED)
    left = encoder.encode(
        image,
        artifact_id="latent-left",
        source_image_ref="crop-source",
        roi=(0.0, 0.0, 0.5, 1.0),
        compatible_consumers=("reasoning",),
    )
    right = encoder.encode(
        image,
        artifact_id="latent-right",
        source_image_ref="crop-source",
        roi=(0.5, 0.0, 1.0, 1.0),
        compatible_consumers=("reasoning",),
    )

    consumer = LatentConsumer(store, "reasoning", ENABLED)
    assert consumer.consume(left.artifact_id).values[:3] == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert consumer.consume(right.artifact_id).values[:3] == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert left.roi == (0.0, 0.0, 0.5, 1.0)
    assert right.roi == (0.5, 0.0, 1.0, 1.0)


def test_latent_consumer_enforces_compatibility_and_feature_flag() -> None:
    store = InMemoryArtifactStore()
    artifact = PixelStatsLatentEncoder(store, ENABLED).encode(
        Image.new("RGB", (2, 2), "white"),
        artifact_id="latent-private",
        source_image_ref="task-image",
        compatible_consumers=("verification",),
    )

    with pytest.raises(IncompatibleLatentConsumerError):
        LatentConsumer(store, "reasoning", ENABLED).consume(artifact.artifact_id)
    with pytest.raises(LatentFeatureDisabledError):
        LatentConsumer(store, "verification").consume(artifact.artifact_id)


def test_store_rejects_byte_size_hash_mismatch_and_duplicate_ids() -> None:
    payload = b"four"
    base = VisualArtifact(
        artifact_id="image-bytes",
        kind=ArtifactKind.IMAGE,
        media_type="application/octet-stream",
        shape=(len(payload),),
        dtype="uint8",
        byte_size=len(payload),
        source_image_ref="task-image",
        roi=None,
        encoder=None,
        layer=None,
        projector=None,
        compatible_consumers=(),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    store = InMemoryArtifactStore()
    with pytest.raises(ArtifactIntegrityError, match="byte_size"):
        store.put(replace(base, artifact_id="wrong-size", byte_size=5), payload)
    with pytest.raises(ArtifactIntegrityError, match="sha256"):
        store.put(replace(base, artifact_id="wrong-hash", sha256="0" * 64), payload)

    store.put(base, bytearray(payload))
    assert store.get(base.artifact_id).data == payload
    with pytest.raises(ValueError, match="already exists"):
        store.put(base, payload)
