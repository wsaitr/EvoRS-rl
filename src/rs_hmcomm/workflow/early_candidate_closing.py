"""Complete an existing source-supported candidate transaction before reopening it.

Only already legal verification actions receive priority. No verifier decision,
semantic rating, answer or finish permission is manufactured.
"""
from .early_adaptive_closing import EarlyAdaptiveClosingRuntime


class EarlyCandidateClosingRuntime(EarlyAdaptiveClosingRuntime):
    def _controller_role_choices(self, choices, context):
        legal = super()._controller_role_choices(choices, context)
        if not self.mechanisms.progress_guard or getattr(self, 'acquisition_request', None):
            return legal
        if not self.state.answer_candidates:
            return legal
        candidate = self.state.answer_candidates[-1]
        if candidate.candidate_id in self.verdicts or self.ledger is None:
            return legal
        verification = [choice for choice in legal if choice['role'] == 'verification']
        if not verification or not self._finish_allowed(candidate):
            return legal
        self.events.append({'kind': 'pending_candidate_verification_priority',
            'candidate_id': candidate.candidate_id,
            'candidate_support_ids': list(candidate.evidence_ids),
            'semantic_support_is_model_estimate': True,
            'verifier_decision_not_forced': True,
            'finish_constraints_unchanged': True, 'reference_used': False})
        return verification
