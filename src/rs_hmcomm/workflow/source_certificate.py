"""Disclose existing semantic certificates only with all their observed sources."""
from .batched_pointer_workflow import BatchedPointerWorkflowRuntime


class SourceCertificateRuntime(BatchedPointerWorkflowRuntime):
    def _support_closure(self, selected):
        closed = super()._support_closure(selected)
        ledger = getattr(self, 'ledger', None)
        if ledger is None:
            return closed
        ids = {e.evidence_id for e in closed}
        additions = []
        for evidence in self.state.evidence:
            if evidence.evidence_id in ids:
                continue
            ratings = ledger.assessments.get(evidence.evidence_id)
            if not ratings:
                continue
            sources = set(evidence.payload.get('source_evidence_ids', ()))
            observed = {eid for rating in ratings.values()
                        for eid in rating.get('observed_evidence_ids', ())}
            # No reverse traversal may add a new image or credit omitted views.
            # Use actual ledger registrations, not self-written guidance claims.
            if sources and observed and sources <= ids and observed <= ids:
                additions.append(evidence)
        # Retain uncertain/contradicted certificates as well as supported ones.
        # The existing ledger still decides sufficiency; routing bills all metadata.
        return (*closed, *additions)
