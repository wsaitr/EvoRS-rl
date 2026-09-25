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
    CostVector,
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

    # Qwen-VL models report coordinates on their internal smart-resize
    # canvas (probed at ~1024 for this server build), independent of the
    # file dimensions.  Calibrate with scripts/probe_bbox_canvas.py.
    canvas = 1024.0

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

    def __init__(
        self,
        backend: ModelBackend,
        *,
        image_store: ImageStore | None = None,
        artifact_store: Any = None,
        latent_consumer_factory: Any = None,
    ) -> None:
        self.backend = backend
        self.image_store = image_store or DEFAULT_IMAGE_STORE
        # Optional real-latent consumption: artifact_store holds binary
        # latents; latent_consumer_factory(store, consumer_id) must return an
        # object with ``consume(artifact_id) -> tuple[float, ...]``.
        self.artifact_store = artifact_store
        self.latent_consumer_factory = latent_consumer_factory

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
        *,
        cost_delta: CostVector | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> ExpertResult:
        result = ExpertResult(
            expert_id=self.expert_id,
            workspace_delta=delta,
            cost_delta=cost_delta or CostVector(),
            metrics=metrics or {},
        )
        result.validate_against(observation)
        return result

    def _generate(
        self,
        observation: ExpertObservation,
        prompt: str,
        *,
        image: Any = None,
        extra_visuals: tuple[VisualInput, ...] = (),
    ) -> tuple[str, int, int]:
        visuals: list[VisualInput] = []
        if image is not None:
            visuals.append(
                VisualInput(visual_id="input_image", label="input_image", image=image),
            )
        for extra in extra_visuals:
            if not isinstance(extra, VisualInput):
                raise TypeError("extra_visuals must contain VisualInput instances")
            visuals.append(extra)
        request = BackendRequest(prompt=prompt, visuals=tuple(visuals))
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

    With ``zoom_confirm=True`` each candidate bbox is deterministically
    cropped (and upscaled if small), then re-shown to the VLM for a yes/no
    confirmation pass; rejected candidates are dropped before the hierarchy
    is written.
    """

    expert_id = "visual"

    CONFIRM_PROMPT = (
        'This is a zoomed crop of a remote-sensing image. Does this crop '
        'clearly contain a "{label}"? Answer with exactly Yes or No.'
    )

    def __init__(
        self,
        backend: ModelBackend,
        *,
        image_store: ImageStore | None = None,
        zoom_confirm: bool = False,
        min_zoom_side: int = 112,
        use_contact_sheet: bool = False,
    ) -> None:
        super().__init__(backend, image_store=image_store)
        self.zoom_confirm = zoom_confirm
        self.min_zoom_side = min_zoom_side
        self.use_contact_sheet = use_contact_sheet

    def _crop_bbox(self, img, bbox):
        """Deterministic pixel crop; returns None on degenerate boxes."""
        width, height = img.size
        x1, y1, x2, y2 = bbox
        left = math.floor(x1 * width)
        top = math.floor(y1 * height)
        right = math.ceil(x2 * width)
        bottom = math.ceil(y2 * height)
        if right <= left or bottom <= top:
            return None
        cropped = img.crop((left, top, right, bottom))
        if cropped.width <= 0 or cropped.height <= 0:
            return None
        return cropped

    def _build_contact_sheet(
        self,
        image,
        detections: list[tuple[str, tuple[float, float, float, float]]],
        *,
        task_id: str,
    ) -> tuple[str | None, tuple]:
        """Assemble a contact sheet via ContactSheetTool and persist it to
        the image store under a task-scoped ref.  Returns ``(ref, panels)``.
        Panels is a tuple of (number, label) dicts for the instance table.
        """
        from rs_hmcomm.tools import ContactSheetTool
        from rs_hmcomm.agents.action_protocol import CallTool
        from rs_hmcomm.backends.base import VisualInput as _VI
        from rs_hmcomm.tools.base import ToolContext as _TC

        if not detections:
            return None, ()
        # Cap at what the tool renders to keep downstream consistent.
        capped = detections[:8]
        tool = ContactSheetTool()
        call = CallTool(
            call_id=f"contact-sheet-{task_id}-{hash(tuple((l, b) for l, b in capped)) & 0xFFFF:04x}",
            tool_name="contact_sheet",
            arguments={
                "image_ref": "input_image",
                "detections": [
                    {"label": label, "bbox": list(bbox)} for label, bbox in capped
                ],
            },
        )
        context = _TC(
            agent_id=self.expert_id,
            step=0,
            visuals_by_id={"input_image": _VI(visual_id="input_image", label="img", image=image)},
        )
        try:
            result = tool.execute(call, context)
        except Exception:
            # Tool failure must not break the expert pipeline; we fall back
            # to no contact sheet.
            return None, ()
        if not result.ok or not result.visuals:
            return None, ()
        sheet_image = result.visuals[0].image
        ref = f"contact_sheet_{task_id}"
        self.image_store.put(ref, sheet_image)
        panels = tuple(
            {"number": p["number"], "label": p["label"]}
            for p in result.metadata.get("panels", ())
        )
        return ref, panels

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
        confirm_tokens_in = confirm_tokens_out = 0
        confirm_calls = 0
        confirmed_count = 0

        def maybe_zoom_confirm(label, bbox):
            nonlocal confirm_tokens_in, confirm_tokens_out, confirmed_count, confirm_calls
            if not self.zoom_confirm or image is None:
                return True
            region = self._crop_bbox(image, bbox)
            if region is None:
                return True
            w, h = region.size
            scale = max(1.0, self.min_zoom_side / max(w, h))
            if scale > 1.0:
                region = region.resize((max(28, round(w * scale)), max(28, round(h * scale))))
            verdict_text, tin, tout = self._generate(
                observation,
                self.CONFIRM_PROMPT.format(label=label),
                image=region,
            )
            confirm_tokens_in += tin
            confirm_tokens_out += tout
            confirm_calls += 1
            ok = verdict_text.strip().lower().startswith("yes")
            if ok:
                confirmed_count += 1
            return ok

        scene_id = f"visual-scene-s{observation.step}"

        parsed = _extract_json_object(answer_text)
        raw_objects = parsed.get("objects") if isinstance(parsed, dict) else None
        detections: list[tuple[str, tuple[float, float, float, float]]] = []
        if isinstance(raw_objects, list):
            for item in raw_objects:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "object")).strip() or "object"
                boxes = _parse_bboxes(json.dumps(item.get("bbox")), image_size)
                if boxes and maybe_zoom_confirm(label, boxes[0]):
                    detections.append((label, boxes[0]))

        scene_node = SpatialNode(
            node_id=scene_id,
            level=SpatialLevel.SCENE,
            semantic="question-conditioned remote-sensing scene",
            provenance={
                "mechanism": "vlm_visual_expert_v1",
                "model": type(self.backend).__name__,
                "tokens_in": tokens_in + confirm_tokens_in,
                "tokens_out": tokens_out + confirm_tokens_out,
                "zoom_confirm": self.zoom_confirm,
                "zoom_confirmed_count": confirmed_count,
            },
        )

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

        # Assemble contact sheet: one overview panel + zoomed ROI panels.
        # The contact sheet is the key perceptual mechanism — reasoning
        # expert will read this image (not bbox text) to perceive the ROIs.
        contact_sheet_ref = None
        contact_sheet_panels = ()
        if image is not None and self.use_contact_sheet:
            contact_sheet_ref, contact_sheet_panels = self._build_contact_sheet(
                image, detections, task_id=workspace.task.task_id
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
                "contact_sheet_image_ref": contact_sheet_ref,
                "contact_sheet_panels": contact_sheet_panels,
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
        return self._result(
            observation,
            delta,
            cost_delta=CostVector(
                model_calls=1 + confirm_calls,
                model_input_tokens=tokens_in + confirm_tokens_in,
                model_output_tokens=tokens_out + confirm_tokens_out,
            ),
            metrics={
                "detections": len(detections),
                "confirmation_calls": confirm_calls,
                "confirmed_detections": confirmed_count,
            },
        )


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

    def __init__(
        self,
        backend: ModelBackend,
        *,
        image_store: ImageStore | None = None,
        artifact_store: InMemoryArtifactStore | None = None,
        latent_consumer_factory: Any | None = None,
        include_detection_context: bool = True,
    ):
        super().__init__(backend, image_store=image_store)
        self.artifact_store = artifact_store
        self.latent_consumer_factory = latent_consumer_factory
        self.include_detection_context = include_detection_context

    def _consume_visible_latents(
        self,
        workspace: ScopedWorkspaceView,
    ) -> dict[str, float]:
        """Actually read authorized latent artifacts and score them.

        Consumption = load each compatible latent vector this expert is
        granted, compute its L2 norm and pairwise cosine against the scene
        latent.  Returns per-artifact diagnostics injected into the answer
        prompt; empty when no latents are visible or no store is wired.
        """
        if self.artifact_store is None or self.latent_consumer_factory is None:
            return {}
        latent_artifacts = [
            artifact
            for artifact in workspace.artifacts
            if getattr(artifact, "kind", None) is not None
            and getattr(artifact.kind, "value", "") == "latent"
            and "reasoning" in artifact.compatible_consumers
        ]
        if not latent_artifacts:
            return {}
        consumer = self.latent_consumer_factory(self.artifact_store, "reasoning")
        vectors: dict[str, tuple[float, ...]] = {}
        for artifact in latent_artifacts:
            try:
                vectors[artifact.artifact_id] = tuple(consumer.consume(artifact.artifact_id))
            except (PermissionError, ValueError, KeyError):
                continue

        diagnostics: dict[str, float] = {}
        scene_id = next(
            (key for key in vectors if "scene" in key),
            "",
        )
        scene_vector = vectors.get(scene_id)
        for artifact_id, vector in vectors.items():
            norm = math.sqrt(sum(value * value for value in vector))
            diagnostics[artifact_id] = round(norm, 4)
            if scene_vector is not None and artifact_id != scene_id:
                dot = sum(a * b for a, b in zip(vector, scene_vector))
                denom = norm * math.sqrt(sum(v * v for v in scene_vector)) or 1.0
                diagnostics[f"{artifact_id}~scene_cos"] = round(dot / denom, 4)
        return diagnostics

    def run(self, observation: ExpertObservation) -> ExpertResult:
        self._require_observation(observation)
        workspace = observation.workspace
        if not workspace.evidence:
            raise ValueError("reasoning expert requires at least one visible evidence record")
        image = self.image_store.get(workspace.task.image_ref)

        # Contact sheet path (perceptual enhancement): if the visual expert
        # produced a contact sheet, reasoning sees it as a visual input —
        # each ROI is actually visible to the VLM, not summarized as text.
        contact_sheet_ref = f"contact_sheet_{workspace.task.task_id}"
        contact_sheet_image = self.image_store.get(contact_sheet_ref)

        # Instance table: minimal text "#1 roundabout, #2 roundabout, ..."
        # Pull panel metadata from the visual evidence's payload if present.
        panels: tuple = ()
        for evidence in workspace.evidence:
            payload = evidence.payload
            cs_panels = payload.get("contact_sheet_panels") if payload else None
            if cs_panels:
                panels = cs_panels
                break
        if panels:
            instance_table = "Detected instances (see contact sheet image for visuals):\n" + "\n".join(
                f"- #{p['number']} {p['label']}" for p in panels
            ) + "\n"
        else:
            instance_table = ""

        selected_evidence = tuple(
            item.summary
            for item in workspace.evidence
            if item.kind in {"active_perception_result", "domain_knowledge"}
        )
        evidence_block = ""
        if selected_evidence:
            evidence_block = "Disclosed expert evidence:\n" + "\n".join(
                f"- {summary}" for summary in selected_evidence[:6]
            ) + "\n"

        # Build context block: instance table only (no bbox text, no latent
        # scalars). Latent artifacts are physically consumed below; only the
        # number of authorized descriptors is exposed to the language prompt.
        context_block = (
            instance_table + evidence_block
            if self.include_detection_context
            else ""
        )
        latent_diagnostics = self._consume_visible_latents(workspace)
        if latent_diagnostics:
            latent_count = sum(
                1 for key in latent_diagnostics if not key.endswith("~scene_cos")
            )
            context_block += (
                f"Authorized visual latent descriptors consumed: {latent_count}.\n"
            )

        # If a contact sheet exists, pass it as a second visual so the VLM
        # can actually see each ROI at high resolution.
        extra_visuals: tuple[VisualInput, ...] = ()
        if contact_sheet_image is not None:
            extra_visuals = (
                VisualInput(
                    visual_id="contact_sheet",
                    label="contact_sheet",
                    image=contact_sheet_image,
                ),
            )

        answer_text, tokens_in, tokens_out = self._generate(
            observation,
            self.ANSWER_PROMPT.format(question=workspace.task.question, context_block=context_block),
            image=image,
            extra_visuals=extra_visuals,
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
                + (f" and contact sheet with {len(panels)} instance(s)" if panels else "")
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
        return self._result(
            observation,
            delta,
            cost_delta=CostVector(
                model_calls=1,
                model_input_tokens=tokens_in,
                model_output_tokens=tokens_out,
            ),
            metrics={
                "latent_diagnostic_count": len(latent_diagnostics),
                "contact_sheet_panels": len(panels),
            },
        )


class VLMVerificationExpert(_VLMMechanism):
    """Have the VLM check the proposed candidate against the image."""

    expert_id = "verification"

    VERIFY_PROMPT = (
        "Verify this candidate answer against the image.\n"
        "Question: {question}\n"
        "Proposed answer: {candidate}\n"
        "Disclosed supporting evidence:\n{evidence_block}\n"
        'Respond with ONLY JSON: {{"verdict": "accept"|"reject", '
        '"revised_answer": "<string>", "confidence": <0..1>}}\n'
        "Reject only when the image clearly contradicts the candidate. "
        "Multi-view evidence should not be overridden by an uncertain single-view count."
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
        supporting = tuple(
            item
            for item in workspace.evidence
            if item.kind in {"active_perception_result", "domain_knowledge"}
        )
        evidence_block = "\n".join(
            f"- {item.summary}" for item in supporting[:6]
        ) or "- No additional expert evidence disclosed."
        answer_text, tokens_in, tokens_out = self._generate(
            observation,
            self.VERIFY_PROMPT.format(
                question=workspace.task.question,
                candidate=source.answer,
                evidence_block=evidence_block,
            ),
            image=image,
        )

        parsed = _extract_json_object(answer_text) or {}
        verdict = str(parsed.get("verdict", "")).strip().lower()
        revised = parsed.get("revised_answer")
        raw_confidence = parsed.get("confidence")
        verifier_confidence = None
        if isinstance(raw_confidence, (int, float)) and not isinstance(
            raw_confidence, bool
        ):
            normalized_confidence = float(raw_confidence)
            if 0.0 <= normalized_confidence <= 1.0:
                verifier_confidence = normalized_confidence
        strongly_supported = any(
            str(item.payload.get("answer", "")).strip() == source.answer.strip()
            and item.kind == "active_perception_result"
            for item in supporting
        )
        object_count = sum(
            item.level == SpatialLevel.OBJECT for item in workspace.spatial_nodes
        )
        candidate_count = (
            int(source.answer.strip()) if source.answer.strip().isdigit() else None
        )
        structurally_supported = (
            candidate_count is not None
            and object_count > 0
            and candidate_count == object_count
        )
        # Self-reported VLM confidence is not independent evidence. A single
        # whole-image verification pass cannot override agreement between the
        # private multi-view result and the committed spatial object count.
        can_override = not (strongly_supported and structurally_supported)
        if (
            verdict == "reject"
            and isinstance(revised, str)
            and revised.strip()
            and can_override
        ):
            final_answer = revised.strip()
        elif verdict == "accept":
            final_answer = source.answer
        else:
            # Unparseable or weak contradiction: preserve the evidence-backed
            # candidate instead of allowing one extra VLM call to regress it.
            final_answer = source.answer
        confidence = (
            verifier_confidence
            if verifier_confidence is not None
            else (0.8 if strongly_supported else 0.4)
        )
        effective_verdict = (
            "reject_overruled_by_multiview_evidence"
            if verdict == "reject" and not can_override
            else (verdict or "unparseable")
        )

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
        return self._result(
            observation,
            delta,
            cost_delta=CostVector(
                model_calls=1,
                model_input_tokens=tokens_in,
                model_output_tokens=tokens_out,
            ),
            metrics={
                "verdict": effective_verdict,
                "strong_multiview_support": strongly_supported,
                "structural_count_support": structurally_supported,
                "visible_object_count": object_count,
                "verifier_confidence": verifier_confidence,
            },
        )


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
        # The backend protocol requires at least one visual per request, so
        # the knowledge expert also sees the whole image (guidance-only task).
        guidance, tokens_in, tokens_out = self._generate(
            observation,
            self.KNOWLEDGE_PROMPT.format(question=workspace.task.question),
            image=self.image_store.get(workspace.task.image_ref),
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
        return self._result(
            observation,
            delta,
            cost_delta=CostVector(
                model_calls=1,
                model_input_tokens=tokens_in,
                model_output_tokens=tokens_out,
            ),
        )


def build_vlm_experts(
    backend: ModelBackend,
    *,
    image_store: ImageStore | None = None,
    artifact_store: Any = None,
    latent_consumer_factory: Any = None,
    include_detection_context: bool = True,
    use_contact_sheet: bool = False,
) -> dict[str, Any]:
    """Build the four VLM-driven experts keyed by expert_id.

    ``artifact_store`` + ``latent_consumer_factory`` enable real latent
    consumption in the reasoning expert (granted latents are loaded and
    scored; diagnostics enter its answer prompt).

    ``include_detection_context`` controls whether the reasoning expert
    receives detection text and latent diagnostics in its prompt. Set to
    False for a "clean" mode where reasoning only sees the image + question.

    ``use_contact_sheet`` enables the contact-sheet perceptual mechanism:
    the visual expert assembles a contact sheet image (overview + zoomed
    ROI panels), reasoning expert receives it as a visual input and uses
    an instance table instead of bbox text.
    """
    return {
        "visual": VLMVisualExpert(
            backend,
            image_store=image_store,
            use_contact_sheet=use_contact_sheet,
        ),
        "knowledge": VLMKnowledgeExpert(backend, image_store=image_store),
        "reasoning": VLMReasoningExpert(
            backend,
            image_store=image_store,
            artifact_store=artifact_store,
            latent_consumer_factory=latent_consumer_factory,
            include_detection_context=include_detection_context,
        ),
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
