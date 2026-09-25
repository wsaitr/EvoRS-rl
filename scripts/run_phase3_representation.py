"""Run the Phase 3 Workspace/spatial/artifact/latent representation smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from rs_hmcomm.artifacts import (
    InMemoryArtifactStore,
    LatentConsumer,
    LatentFeatureConfig,
    PixelStatsLatentEncoder,
)
from rs_hmcomm.contracts import AnswerCandidate, Claim, CostVector, Evidence, TaskContext
from rs_hmcomm.datasets.manifest import iter_manifest_dataset
from rs_hmcomm.spatial import SpatialLevel, SpatialNode
from rs_hmcomm.workspace import (
    WorkspaceDelta,
    WorkspaceReducer,
    WorkspaceState,
    build_scoped_view,
)


DEFAULT_CONFIG = Path("configs/experiments/phase3_representation.yaml")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = _mapping(yaml.safe_load(handle), "config")
    unknown = set(values) - {"phase", "dataset", "latent", "logging"}
    if unknown:
        raise ValueError(f"unknown Phase 3 config fields: {sorted(unknown)}")
    if values.get("phase") != 3:
        raise ValueError("Phase 3 runner requires phase: 3")
    return values


def _write_summary(path: Path, summary: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("Phase 3 output must be a file")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(
    config_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
    latent_enabled_override: bool | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    dataset_values = _mapping(config.get("dataset"), "dataset")
    manifest_value = dataset_values.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("dataset.manifest must be a non-empty path")
    samples = tuple(
        iter_manifest_dataset(manifest_value, limit=dataset_values.get("limit"))
    )
    if not samples:
        raise ValueError("Phase 3 manifest contains no samples")

    latent_values = _mapping(config.get("latent"), "latent")
    enabled_value = latent_values.get("enabled", False)
    if not isinstance(enabled_value, bool):
        raise TypeError("latent.enabled must be a boolean")
    latent_enabled = (
        enabled_value
        if latent_enabled_override is None
        else latent_enabled_override
    )
    consumer_id = latent_values.get("consumer_id", "reasoning")
    if not isinstance(consumer_id, str) or not consumer_id.strip():
        raise ValueError("latent.consumer_id must be a non-empty string")
    roi_value = latent_values.get("roi")
    roi = tuple(roi_value) if roi_value is not None else None

    results: list[dict[str, Any]] = []
    for sample in samples:
        image_ref = f"manifest://{Path(manifest_value).name}/{sample.sample_id}"
        task = TaskContext(
            task_id=f"task-{sample.sample_id}",
            image_ref=image_ref,
            question=sample.question,
            task_type=sample.task_type,
            answer_format="short_text",
            metadata={"fixture": True},
        )
        store = InMemoryArtifactStore()
        feature_config = LatentFeatureConfig(enabled=latent_enabled)
        artifacts = ()
        artifact_refs: tuple[str, ...] = ()
        if latent_enabled:
            descriptor = PixelStatsLatentEncoder(store, feature_config).encode(
                sample.image,
                artifact_id=f"latent-{sample.sample_id}",
                source_image_ref=image_ref,
                roi=roi,
                compatible_consumers=(consumer_id,),
            )
            artifacts = (descriptor,)
            artifact_refs = (descriptor.artifact_id,)

        scene = SpatialNode(
            node_id=f"scene-{sample.sample_id}",
            level=SpatialLevel.SCENE,
            semantic="remote_sensing_scene",
            confidence=1.0,
            provenance={"question_conditioned": True},
        )
        object_node = SpatialNode(
            node_id=f"object-{sample.sample_id}",
            level=SpatialLevel.OBJECT,
            semantic="question_relevant_object_group",
            bbox=roi or (0.0, 0.0, 1.0, 1.0),
            parent_id=scene.node_id,
            confidence=0.8,
            uncertainty=0.2,
            evidence_ids=(f"evidence-{sample.sample_id}",),
            visual_artifact_refs=artifact_refs,
            provenance={"created_by": "phase3-generalist-smoke"},
        )
        evidence = Evidence(
            evidence_id=f"evidence-{sample.sample_id}",
            kind="spatial_observation",
            summary="Question-relevant objects are localized in the scene.",
            created_by="generalist",
            confidence=0.8,
            spatial_node_ids=(object_node.node_id,),
            artifact_refs=artifact_refs,
        )
        hidden_evidence = Evidence(
            evidence_id=f"unrelated-{sample.sample_id}",
            kind="control_record",
            summary="Independent record used to verify scoped reads.",
            created_by="runtime",
        )
        claim = Claim(
            claim_id=f"claim-{sample.sample_id}",
            statement="The answer is supported by localized visual evidence.",
            created_by="generalist",
            evidence_ids=(evidence.evidence_id,),
            confidence=0.8,
        )
        answer = AnswerCandidate(
            candidate_id=f"answer-{sample.sample_id}",
            answer=sample.answer,
            created_by="generalist",
            claim_ids=(claim.claim_id,),
            confidence=0.8,
        )

        initial = WorkspaceState.initial(f"episode-{sample.sample_id}", task)
        state = WorkspaceReducer.apply(
            initial,
            WorkspaceDelta(
                delta_id=f"delta-{sample.sample_id}",
                episode_id=initial.episode_id,
                base_version=initial.version,
                actor_id="generalist",
                evidence=(evidence, hidden_evidence),
                claims=(claim,),
                answer_candidates=(answer,),
                spatial_nodes=(scene, object_node),
                artifacts=artifacts,
            ),
        )
        view = build_scoped_view(state, (answer.candidate_id,))
        if hidden_evidence.evidence_id in view.visible_record_ids:
            raise RuntimeError("scoped view leaked an unrelated record")

        bytes_loaded = 0
        latent_value_count = 0
        latent_value_digest = None
        if latent_enabled:
            visible_artifact_ids = {item.artifact_id for item in view.artifacts}
            if artifact_refs[0] not in visible_artifact_ids:
                raise RuntimeError("authorized latent descriptor is absent from scoped view")
            read = LatentConsumer(store, consumer_id, feature_config).consume(
                artifact_refs[0]
            )
            bytes_loaded = read.bytes_loaded
            latent_value_count = len(read.values)
            latent_value_digest = hashlib.sha256(repr(read.values).encode("ascii")).hexdigest()

        workspace_json = json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True)
        results.append(
            {
                "sample_id": sample.sample_id,
                "workspace_version": state.version,
                "workspace_record_count": len(state.record_ids),
                "scoped_visible_record_ids": sorted(view.visible_record_ids),
                "unrelated_record_hidden": hidden_evidence.evidence_id
                not in view.visible_record_ids,
                "spatial_levels": [item.level.value for item in view.spatial_nodes],
                "object_parent_level": scene.level.value,
                "artifact_descriptor_count": len(view.artifacts),
                "workspace_contains_binary_payload": "artifact_payload" in workspace_json,
                "latent_enabled": latent_enabled,
                "latent_consumed": latent_enabled and bytes_loaded > 0,
                "latent_value_count": latent_value_count,
                "latent_value_sha256": latent_value_digest,
                "latent_bytes_loaded": bytes_loaded,
                "cost": CostVector(message_latent_bytes=bytes_loaded).to_dict(),
            }
        )

    passed = all(
        item["workspace_version"] == 1
        and item["unrelated_record_hidden"]
        and not item["workspace_contains_binary_payload"]
        and (
            (item["latent_consumed"] and item["latent_bytes_loaded"] > 0)
            if latent_enabled
            else (
                not item["latent_consumed"]
                and item["artifact_descriptor_count"] == 0
            )
        )
        for item in results
    )
    summary = {
        "schema_version": "evors.phase3-representation-summary.v1",
        "phase": 3,
        "status": "passed" if passed else "failed",
        "validation_scope": "representation_mechanism_smoke",
        "scientific_claim": False,
        "latent_encoder": (
            "pixel_stats_smoke_v1" if latent_enabled else None
        ),
        "learned_visual_representation": False,
        "sample_count": len(results),
        "results": results,
    }
    if output_path is None:
        logging_values = _mapping(config.get("logging"), "logging")
        raw_output = logging_values.get("output")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ValueError("logging.output must be a non-empty path")
        output_path = Path(raw_output)
    summary["output"] = str(output_path)
    _write_summary(output_path, summary, overwrite=overwrite)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-latent", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        args.config,
        output_path=args.output,
        overwrite=args.overwrite,
        latent_enabled_override=False if args.no_latent else None,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
