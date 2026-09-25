"""Workspace-aware reference router and scoped observation builder."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

from rs_hmcomm.artifacts import ArtifactKind, VisualArtifact
from rs_hmcomm.contracts import CostVector
from rs_hmcomm.contracts._validation import require_id, require_step, require_text
from rs_hmcomm.workspace import ScopedWorkspaceView, WorkspaceState, build_scoped_view

from .protocol import (
    AccessGrant,
    Inbox,
    MessageEnvelope,
    MessageModality,
    _modalities,
    _workspace_refs,
)


_TEXT_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _artifact_map(workspace: WorkspaceState) -> dict[str, VisualArtifact]:
    return {artifact.artifact_id: artifact for artifact in workspace.artifacts}


class ContextRouter:
    """Create one explicit disclosure and its matching access grant."""

    def __init__(self) -> None:
        self._sequence = 0

    @staticmethod
    def measure_cost(
        workspace: WorkspaceState,
        envelope: MessageEnvelope,
    ) -> CostVector:
        if not isinstance(workspace, WorkspaceState):
            raise TypeError("workspace must be a WorkspaceState")
        if not isinstance(envelope, MessageEnvelope):
            raise TypeError("envelope must be a MessageEnvelope")
        serialized = envelope.to_json()
        modalities = frozenset(envelope.modalities)
        text_tokens = (
            len(_TEXT_TOKEN_RE.findall(serialized))
            if MessageModality.TEXT in modalities
            else 0
        )
        structure_bytes = (
            len(serialized.encode("utf-8"))
            if MessageModality.STRUCTURE in modalities
            else 0
        )
        latent_bytes = 0
        if MessageModality.LATENT in modalities:
            artifacts = _artifact_map(workspace)
            latent_bytes = sum(
                artifacts[record_id].byte_size
                for record_id in envelope.workspace_refs
                if record_id in artifacts
                and artifacts[record_id].kind == ArtifactKind.LATENT
            )
        return CostVector(
            message_text_tokens=text_tokens,
            message_structure_bytes=structure_bytes,
            message_latent_bytes=latent_bytes,
        )

    def route(
        self,
        workspace: WorkspaceState,
        sender: str,
        receiver: str,
        refs: Iterable[str],
        modalities: Iterable[MessageModality | str],
        step: int,
        scope: str = "workspace_subset",
    ) -> tuple[MessageEnvelope, AccessGrant]:
        if not isinstance(workspace, WorkspaceState):
            raise TypeError("workspace must be a WorkspaceState")
        require_id(sender, "sender")
        require_id(receiver, "receiver")
        normalized_refs = _workspace_refs(refs)
        normalized_modalities = _modalities(modalities)
        require_step(step, "step")
        require_text(scope, "scope")
        unknown = set(normalized_refs) - set(workspace.record_ids)
        if unknown:
            raise ValueError(f"route contains unknown workspace refs: {sorted(unknown)}")
        artifacts = _artifact_map(workspace)
        explicit_artifacts = tuple(
            artifacts[record_id]
            for record_id in normalized_refs
            if record_id in artifacts
        )
        if explicit_artifacts and MessageModality.LATENT not in normalized_modalities:
            raise ValueError("artifact refs require the latent modality")
        for artifact in explicit_artifacts:
            if artifact.kind != ArtifactKind.LATENT:
                raise ValueError("the latent modality may reference only latent artifacts")
            if receiver not in artifact.compatible_consumers:
                raise PermissionError(
                    f"receiver {receiver!r} is not compatible with {artifact.artifact_id!r}"
                )

        identity_payload = {
            "episode_id": workspace.episode_id,
            "workspace_version": workspace.version,
            "sender": sender,
            "receiver": receiver,
            "refs": normalized_refs,
            "modalities": tuple(item.value for item in normalized_modalities),
            "step": step,
            "scope": scope,
            "sequence": self._sequence,
        }
        digest = hashlib.sha256(
            json.dumps(
                identity_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        self._sequence += 1
        envelope = MessageEnvelope(
            message_id=f"message-{digest}",
            sender=sender,
            receiver=receiver,
            modalities=normalized_modalities,
            workspace_refs=normalized_refs,
            scope=scope,
            created_step=step,
        )
        cost = self.measure_cost(workspace, envelope)
        grant = AccessGrant(
            grant_id=f"grant-{digest}",
            message_id=envelope.message_id,
            grantor=sender,
            grantee=receiver,
            workspace_refs=envelope.workspace_refs,
            created_step=step,
            cost=cost,
        )
        return envelope, grant


class ObservationBuilder:
    """Build only the dependency closure disclosed to one recipient."""

    @staticmethod
    def build(
        workspace: WorkspaceState,
        recipient: str,
        inbox: Inbox,
    ) -> ScopedWorkspaceView:
        if not isinstance(workspace, WorkspaceState):
            raise TypeError("workspace must be a WorkspaceState")
        require_id(recipient, "recipient")
        if not isinstance(inbox, Inbox):
            raise TypeError("inbox must be an Inbox")
        messages = inbox.by_recipient(recipient)
        artifact_ids = frozenset(item.artifact_id for item in workspace.artifacts)
        allowed_artifact_ids = {
            record_id
            for message in messages
            if MessageModality.LATENT in message.modalities
            for record_id in message.workspace_refs
            if record_id in artifact_ids
        }
        authorized_refs = {
            record_id
            for message in messages
            for record_id in message.workspace_refs
            if record_id not in artifact_ids or record_id in allowed_artifact_ids
        }
        return build_scoped_view(
            workspace,
            authorized_refs,
            allowed_artifact_ids=allowed_artifact_ids,
        )


__all__ = ["ContextRouter", "ObservationBuilder"]
