"""Adapt the evidence-pointer cap to a complete remaining transaction budget.

No extra calls, answer shortcuts, provenance truncation or label-based choices.
"""
from .transaction_budget import TransactionBudgetRuntime


class AdaptiveClosingBudgetRuntime(TransactionBudgetRuntime):
    def _reasoning_call_bound(self):
        groups = max(1, len({tuple(e.payload.get('frame_ids', ()))
                            for e in self.state.evidence if e.kind == 'workflow_visual'}))
        variations = self.mechanisms.generations*(self.mechanisms.population_size-1)
        return 1+2*variations+(1+variations)*groups if self.mechanisms.evolution else 1

    def _controller_decide(self, context, legal):
        if not self.mechanisms.progress_guard or not self.mechanisms.pointer:
            return super()._controller_decide(context, legal)
        declared = self.pointer_limit
        remaining = self.max_model_calls-self._model_calls_used()
        pending = bool(self.state.answer_candidates and
                       self.state.answer_candidates[-1].candidate_id not in self.verdicts)
        # Preserve the parent's conservative reserve constants. Change only the
        # maximal selectable subset, so the retained transaction really fits.
        affordable = remaining-7 if pending else (remaining-10-self._reasoning_call_bound())//2
        self.pointer_limit = max(1, min(declared, affordable))
        self.events.append({'kind': 'adaptive_pointer_transaction_cap',
            'declared_pointer_limit': declared, 'effective_pointer_limit': self.pointer_limit,
            'actual_remaining_model_calls': remaining, 'pending_candidate': pending,
            'required_provenance_not_truncated': True, 'reference_used': False,
            'extra_model_calls': 0})
        try:
            return super()._controller_decide(context, legal)
        finally:
            self.pointer_limit = declared

    def _controller_role_choices(self, choices, context):
        if not self.mechanisms.progress_guard:
            return super()._controller_role_choices(choices, context)
        remaining = self.max_model_calls-self._model_calls_used()
        pending = bool(self.state.answer_candidates and
                       self.state.answer_candidates[-1].candidate_id not in self.verdicts)
        cap = min(len(self.state.evidence), self.pointer_limit)
        closing = 2*cap+10+self._reasoning_call_bound()
        # Knowledge dispatch must leave room for producing AND checking an
        # answer. Its own pointer/action/expert calls are paid before closing.
        filtered = [choice for choice in choices if choice['role'] != 'knowledge'
                    or pending or remaining >= cap+4+closing]
        if len(filtered) != len(choices):
            self.events.append({'kind': 'knowledge_closing_reserve',
                'remaining_model_calls': remaining, 'knowledge_call_bound': cap+4,
                'complete_closing_call_bound': closing, 'reference_used': False})
        return super()._controller_role_choices(filtered, context)
