"""Attempt a supported answer early, then route rejection to a real correction.

Does not turn semantic support into correctness, relax finishing, add calls,
accept rejected candidates, or revise answers without model inference.
"""
from itertools import combinations
from .transaction_budget import TransactionBudgetRuntime


class EarlyReasoningRuntime(TransactionBudgetRuntime):
    def _first_attempt_certificate(self):
        if (self.state.answer_candidates or getattr(self,'acquisition_request',None)
                or self.ledger is None or not self.ledger.sufficient()):
            return set()
        lookup={record.evidence_id:record for record in self.state.evidence}
        assessments=[eid for eid in reversed(tuple(self.ledger.assessments)) if eid in lookup]
        for size in range(1,min(len(assessments),self.pointer_limit)+1):
            for subset in combinations(assessments,size):
                closed=self._support_closure([lookup[eid] for eid in subset])
                if (self.ledger.sufficient([e.evidence_id for e in closed])
                        and self._possible_modes('reasoning',subset)):
                    return set(subset)
        return set()

    def _correction_feedback(self):
        if not self.state.answer_candidates:
            return None
        candidate=self.state.answer_candidates[-1]
        if self.verdicts.get(candidate.candidate_id)!='reject':
            return None
        for record in reversed(self.state.evidence):
            if (record.kind=='workflow_guidance' and record.created_by=='verification'
                    and record.payload.get('diagnostic_only') is True
                    and record.payload.get('target_candidate_id')==candidate.candidate_id
                    and record.payload.get('decision')=='reject'):
                return record.evidence_id
        return None

    def _required_pointer_evidence(self,role):
        feedback=self._correction_feedback() if role=='reasoning' else None
        if feedback:
            return {feedback}
        if role=='reasoning':
            return self._first_attempt_certificate()
        return super()._required_pointer_evidence(role)

    def _controller_role_choices(self,choices,context):
        if getattr(self,'acquisition_request',None):
            return choices
        reasoning=[choice for choice in choices if choice['role']=='reasoning']
        if not reasoning:
            return choices
        certificate=self._first_attempt_certificate()
        sufficient=bool(certificate)
        feedback=self._correction_feedback()
        if sufficient or feedback:
            self.events.append({'kind':'early_reasoning_priority',
                'reason':'supported first attempt' if sufficient else 'source-linked rejected candidate correction',
                'required_feedback_id':feedback,'finish_constraints_unchanged':True,
                'required_semantic_certificate_ids':sorted(certificate),
                'reference_used':False})
            return reasoning
        return choices
