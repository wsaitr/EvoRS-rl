"""VLM-backed Cognitive Expert mechanisms producing real hierarchical state.

Unlike ``mechanisms.py`` (deterministic rule fixtures), these experts call a
real vision-language model through a ``ModelBackend`` and write question-
conditioned Scene/Object spatial nodes with model-proposed bounding boxes into
the shared Workspace.  Every record keeps ``provenance`` marking the model,
so downstream consumers can distinguish measured output from fixtures.

Scientific claim: the mechanisms are real (model-driven), but any accuracy
claim still requires the experiment harness; nothing here reads labels.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from rs_hmcomm.backends.base import BackendRequest, ModelBackend, VisualInput
from rs_hmcomm.contracts import (
    AnswerCandidate,
    Claim,
    Evidence,
    RecordStatus,
)
from rs_hmcomm.spatial import SpatialLevel, SpatialNode
from rs_hmcomm.workspace import ScopedWorkspaceView, WorkspaceDelta

from .contracts import ExpertObservation, ExpertResult
from .image_store import ImageStore


_BBOX_RE = re.compile(
    r"\[\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\]"
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_bboxes(
    text: str,
    image_size: tuple[int, int] | None = None,
) -> list[tuple[float, float, float, float]]:
    """Parse [x1, y1, x2, y2] boxes from VLM JSON output as normalized coords.

    Qwen-VL models emit pixel coordinates on their internal (smart-resized)
    canvas, which may exceed the file dimensions.  Strategy: values within
    [0, 1] pass through; otherwise the coordinate canvas is inferred from the
    largest value seen across all boxes in this response (rounded up to a
    multiple of the model's patch grid), and every box is divided by it.
    Boxes still outside [0, 1] after normalization are clamped.
    """
    raw: list[tuple[float, float, float, float]] = []
    for match in _BBOX_RE.finditer(text):
        x1, y1, x2, y2 = (float(group) for group in match.groups())
        if x2 <= x1 or y2 <= y1:
            continue
        raw.append((x1, y1, x2, y2))
    if not raw:
        return []

    max_value = max(value for box in raw for value in box)
    if max_value <= 1.0:
        return [box for box in raw if all(0.0 <= v <= 1.0 for v in box)]

    # Infer the internal canvas from the largest observed coordinate.
    canvas = math.ceil(max_value / 28.0) * 28.0

    def normalize(box: tuple[float, float, float, float]) -> tuple[float, ...]:
        return tuple(_clamp01(value / canvas) for value in box)

    return [tuple(normalize(box)) for box in raw]  # type: ignore[return-value]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object in free-form text."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start >= 0:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


DEFAULT_IMAGE_STORE = ImageStore()


class _VLMMechanism:
    expert_id: str

    def __init__(self, backend: ModelBackend, *, image_store: ImageStore | None = None) -> None:
        self.backend = backend
        self.image_store = image_store or DEFAULT_IMAGE_STORE

    def _require_observation(self, observation: ExpertObservation) -> None:
        if not isinstance(observation, ExpertObservation):
            raise TypeError("observation must be an ExpertObservation")
        if observation.expert_id != self.expert_id:
            raise ValueError(
                f"{self.expert_id} expert cannot execute an "
                f"{observation.expert_id} observation"
            )

    def _result(
        self,
        observation: ExpertObservation,
        delta: WorkspaceDelta,
    ) -> ExpertResult:
        result = ExpertResult(expert_id=self.expert_id, workspace_delta=delta)
        result.validate_against(observation)
        return result

    def _generate(
        self,
        observation: ExpertObservation,
        prompt: str,
        *,
        image: Any = None,
    ) -> tuple[str, int, int]:
        visuals: tuple[VisualInput, ...] = ()
        if image is not None:
            visuals = (
                VisualInput(visual_id="input_image", label="input_image", image=image),
            )
        request = BackendRequest(prompt=prompt, visuals=visuals)
        output = self.backend.generate_turn(request)
        usage = output.usage
        return (
            output.text,
            usage.input_tokens or 0,
            usage.output_tokens or 0,
        )


class VLMVisualExpert(_VLMMechanism):
    """Detect question-relevant objects and build a Scene→Object hierarchy.

    One VLM call returns candidate objects with normalized boxes.  Each parsed
    box becomes an OBJECT node under one SCENE node; unparsed responses fall
    back to a single whole-image object so the hierarchy contract still holds.
    """

    expert_id = "visual"

    DETECT_PROMPT = (
        "You are a remote-sensing image analyst. Locate objects relevant to the "
        "question. Respond with ONLY a JSON object of the form:\n"
        '{{"objects": [{{"label": "<short name>", "bbox": [x1, y1, x2, y2]}}]}}\n'
        "Coordinates are normalized floats in [0,1], origin top-left, format "
        "[x_min, y_min, x_max, y_max]. Include every distinct relevant object "
        "(up to 8). If none are visible use an empty list.\n"
        "Question: {question}"
    )

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        image = self.image_store.get(workspace.task.image_ref)
        image_size = getattr(image, "size", None)

        answer_text, tokens_in, tokens_out = self._generate(
            observation,
            self.DETECT_PROMPT.format(question=workspace.task.question),
            image=image,
        )

        scene_id = f"visual-scene-s{observation.step}"
        scene_node = SpatialNode(
            node_id=scene_id,
            level=SpatialLevel.SCENE,
            semantic="question-conditioned remote-sensing scene",
            provenance={
                "mechanism": "vlm_visual_expert_v1",
                "model": type(self.backend).__name__,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
        )

        parsed = _extract_json_object(answer_text)
        raw_objects = parsed.get("objects") if isinstance(parsed, dict) else None
        detections: list[tuple[str, tuple[float, float, float, float]]] = []
        if isinstance(raw_objects, list):
            for item in raw_objects:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "object")).strip() or "object"
                boxes = _parse_bboxes(json.dumps(item.get("bbox")), image_size)
                if boxes:
                    detections.append((label, boxes[0]))

        object_nodes: list[SpatialNode] = []
        if detections:
            for index, (label, bbox) in enumerate(detections[:8]):
                object_nodes.append(
                    SpatialNode(
                        node_id=f"visual-object-{index}-s{observation.step}",
                        level=SpatialLevel.OBJECT,
                        semantic=label,
                        bbox=bbox,
                        parent_id=scene_id,
                        confidence=0.6,
                        provenance={"source": "vlm_detection"},
                    )
                )
        else:
            object_nodes.append(
                SpatialNode(
                    node_id=f"visual-object-0-s{observation.step}",
                    level=SpatialLevel.OBJECT,
                    semantic="question_relevant_region",
                    bbox=(0.0, 0.0, 1.0, 1.0),
                    parent_id=scene_id,
                    confidence=0.3,
                    provenance={"source": "vlm_fallback_whole_image"},
                )
            )

        evidence = Evidence(
            evidence_id=f"visual-evidence-s{observation.step}",
            kind="visual",
            summary=(
                f"VLM detected {len(detections)} question-relevant object(s): "
                + ", ".join(label for label, _ in detections[:5])
                if detections
                else "VLM detection returned no parseable objects; fallback region used"
            ),
            created_by=self.expert_id,
            created_step=observation.step,
            confidence=0.6 if detections else 0.3,
            spatial_node_ids=tuple(node.node_id for node in object_nodes),
            payload={
                "mechanism": "vlm_visual_expert_v1",
                "model_used": True,
                "raw_response_chars": len(answer_text),
                "scientific_claim": False,
            },
        )
        delta = WorkspaceDelta(
            delta_id=f"visual-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            evidence=(evidence,),
            spatial_nodes=(scene_node, *object_nodes),
        )
        return self._result(observation, delta)


class VLMReasoningExpert(_VLMMechanism):
    """Ask the VLM for the final short answer over the scoped visual context."""

    expert_id = "reasoning"

    ANSWER_PROMPT = (
        "You are answering a remote-sensing question about the provided image.\n"
        "Question: {question}\n"
        "{context_block}"
        "Answer rules: counting questions - reply with a single number only. "
        "Yes/no questions - reply with exactly Yes or No. "
        "Other questions - reply with a short phrase."
    )

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        if not workspace.evidence:
            raise ValueError("reasoning expert requires at least one visible evidence record")
        image = self.image_store.get(workspace.task.image_ref)

        detection_lines = []
        for node in workspace.spatial_nodes:
            if node.level == SpatialLevel.OBJECT:
                bbox = ", ".join(f"{value:.2f}" for value in (node.bbox or ()))
                detection_lines.append(f"- {node.semantic} [{bbox}]")
        context_block = (
            ("Detected objects (normalized xy):\n" + "\n".join(detection_lines) + "\n")
            if detection_lines
            else ""
        )

        answer_text, tokens_in, tokens_out = self._generate(
            observation,
            self.ANSWER_PROMPT.format(question=workspace.task.question, context_block=context_block),
            image=image,
        )
        answer = answer_text.strip()
        if not answer:
            answer = "unanswerable"

        evidence_ids = tuple(item.evidence_id for item in workspace.evidence)
        cost_evidence = Evidence(
            evidence_id=f"reasoning-cost-s{observation.step}",
            kind="model_cost",
            summary=(
                f"VLM answer call: {tokens_in} input + {tokens_out} output tokens"
            ),
            created_by=self.expert_id,
            created_step=observation.step,
            payload={
                "mechanism": "vlm_reasoning_expert_v1",
                "model_input_tokens": tokens_in,
                "model_output_tokens": tokens_out,
                "scientific_claim": False,
            },
        )
        claim = Claim(
            claim_id=f"reasoning-claim-s{observation.step}",
            statement=(
                f"VLM answer derived from {len(evidence_ids)} evidence record(s)"
                + (f" and {len(detection_lines)} localized object node(s)" if detection_lines else "")
            ),
            created_by=self.expert_id,
            evidence_ids=evidence_ids,
            created_step=observation.step,
            status=RecordStatus.PROPOSED,
        )
        candidate = AnswerCandidate(
            candidate_id=f"reasoning-answer-s{observation.step}",
            answer=answer,
            created_by=self.expert_id,
            claim_ids=(claim.claim_id,),
            evidence_ids=evidence_ids,
            created_step=observation.step,
            status=RecordStatus.PROPOSED,
        )
        delta = WorkspaceDelta(
            delta_id=f"reasoning-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            evidence=(cost_evidence,),
            claims=(claim,),
            answer_candidates=(candidate,),
        )
        return self._result(observation, delta)


class VLMVerificationExpert(_VLMMechanism):
    """Have the VLM check the proposed candidate against the image."""

    expert_id = "verification"

    VERIFY_PROMPT = (
        "Verify this candidate answer against the image.\n"
        "Question: {question}\n"
        "Proposed answer: {candidate}\n"
        'Respond with ONLY JSON: {{"verdict": "accept"|"reject", "revised_answer": "<string>"}}\n'
        "If the proposed answer is wrong, provide the corrected answer."
    )

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        if not workspace.answer_candidates:
            raise ValueError("verification expert requires a visible answer candidate")
        image = self.image_store.get(workspace.task.image_ref)

        source = sorted(
            workspace.answer_candidates,
            key=lambda item: item.candidate_id,
        )[-1]
        answer_text, tokens_in, tokens_out = self._generate(
            observation,
            self.VERIFY_PROMPT.format(
                question=workspace.task.question,
                candidate=source.answer,
            ),
            image=image,
        )

        parsed = _extract_json_object(answer_text) or {}
        verdict = str(parsed.get("verdict", "")).strip().lower()
        revised = parsed.get("revised_answer")
        if verdict == "reject" and isinstance(revised, str) and revised.strip():
            final_answer = revised.strip()
        elif verdict == "accept":
            final_answer = source.answer
        else:
            # Unparseable verdict: keep the source answer, mark low confidence.
            final_answer = source.answer
        confidence = 0.7 if verdict in ("accept", "reject") else 0.4

        verified = AnswerCandidate(
            candidate_id=f"{source.candidate_id}-verified-s{observation.step}",
            answer=final_answer,
            created_by=self.expert_id,
            claim_ids=source.claim_ids,
            evidence_ids=source.evidence_ids,
            created_step=observation.step,
            confidence=confidence,
            status=RecordStatus.VERIFIED,
        )
        delta = WorkspaceDelta(
            delta_id=f"verification-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            answer_candidates=(verified,),
        )
        return self._result(observation, delta)


class VLMKnowledgeExpert(_VLMMechanism):
    """Text-only VLM call adding domain guidance conditioned on the question."""

    expert_id = "knowledge"

    KNOWLEDGE_PROMPT = (
        "Give at most two sentences of remote-sensing interpretation guidance "
        "for answering this question (no answer itself):\nQuestion: {question}"
    )

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        guidance, tokens_in, tokens_out = self._generate(
            observation,
            self.KNOWLEDGE_PROMPT.format(question=workspace.task.question),
        )
        evidence = Evidence(
            evidence_id=f"knowledge-evidence-s{observation.step}",
            kind="domain_knowledge",
            summary=guidance.strip() or "No guidance produced.",
            created_by=self.expert_id,
            created_step=observation.step,
            payload={
                "source": "vlm_knowledge_expert_v1",
                "model_used": True,
            },
        )
        delta = WorkspaceDelta(
            delta_id=f"knowledge-delta-s{observation.step}",
            episode_id=observation.episode_id,
            base_version=workspace.version,
            actor_id=self.expert_id,
            evidence=(evidence,),
        )
        return self._result(observation, delta)


def build_vlm_experts(
    backend: ModelBackend,
    *,
    image_store: ImageStore | None = None,
) -> dict[str, Any]:
    """Return the four VLM-driven experts keyed by expert_id."""
    return {
        "visual": VLMVisualExpert(backend, image_store=image_store),
        "knowledge": VLMKnowledgeExpert(backend, image_store=image_store),
        "reasoning": VLMReasoningExpert(backend, image_store=image_store),
        "verification": VLMVerificationExpert(backend, image_store=image_store),
    }


__all__ = [
    "ImageStore",
    "VLMKnowledgeExpert",
    "VLMReasoningExpert",
    "VLMVerificationExpert",
    "VLMVisualExpert",
    "build_vlm_experts",
]
