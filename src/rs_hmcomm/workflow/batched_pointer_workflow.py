"""One sampled JSON subset replaces per-evidence administrative turns."""
from .calculating_workflow import CalculatingWorkflowRuntime
from .bounded_pointer import BoundedPointerRuntime
from .runtime import schema, refs


class BatchedPointerWorkflowRuntime(CalculatingWorkflowRuntime):
    def _sequential_pointer(self):
        return self.training and self.role_coverage_curriculum or self.mechanisms.fixed_schedule

    def _pointer_call_bound(self, ids):
        if self._sequential_pointer():
            return super()._pointer_call_bound(ids)
        # Role choice, one subset completion, communication mode choice.
        return 3

    def _closing_call_bound(self, ids, reasoning_calls):
        if self._sequential_pointer():
            return super()._closing_call_bound(ids, reasoning_calls)
        return 12+reasoning_calls

    def _controller_decide(self, context, legal):
        if self._sequential_pointer():
            return super()._controller_decide(context, legal)
        # Batch selection cost is independent of subset length. Preserve the
        # declared evidence cap rather than applying the old per-ID cap formula.
        return BoundedPointerRuntime._controller_decide(self, context, legal)

    def _controller_role_choices(self, choices, context):
        if self._sequential_pointer():
            return super()._controller_role_choices(choices, context)
        if not self.mechanisms.progress_guard:
            return choices
        remaining = self.max_model_calls-self._model_calls_used()
        pending = bool(self.state.answer_candidates and
                       self.state.answer_candidates[-1].candidate_id not in self.verdicts)
        closing = self._closing_call_bound([], self._reasoning_call_bound())
        return [choice for choice in choices if choice['role'] != 'knowledge'
                or pending or remaining >= 4+closing]

    def _select_pointer(self, role, context, ids, required, missing):
        # Preserve historical fixed-schedule/curriculum ablations exactly.
        if self.training and self.role_coverage_curriculum or self.mechanisms.fixed_schedule:
            return super()._select_pointer(role, context, ids, required, missing)
        value = self._call('controller',
            'Select the addressed evidence subset for this recipient in ONE JSON object '
            'with exactly evidence_ids: a nonempty list of distinct actual evidence IDs. '
            'Retain every required candidate-support ID. Respect maximum_evidence_ids. '
            'Select relevant support for missing requirements; never invent IDs.',
            context={'recipient': role, 'evidence': context['evidence'],
                     'allowed_evidence_ids': list(ids), 'required_evidence_ids': sorted(required),
                     'maximum_evidence_ids': self.pointer_limit,
                     'evidence_pointer_constraint': {'allowed': list(ids),
                         'required': sorted(required), 'maximum': self.pointer_limit},
                     'missing_semantic_requirements': missing,
                     **self._pointer_selection_context(role, ids, required)})
        schema(value, {'evidence_ids'})
        selected = list(refs(value['evidence_ids'], ids))
        if not selected or len(selected) > self.pointer_limit or not set(required) <= set(selected):
            raise ValueError('batched evidence subset exceeds cap or drops required provenance')
        if not self._possible_modes(role, selected):
            raise ValueError('batched subset has no affordable undispatched communication mode')
        self.events.append({'kind': 'batched_evidence_pointer', 'recipient': role,
            'selected': selected, 'required': sorted(required), 'maximum_evidence_ids': self.pointer_limit,
            'selection_model_calls': 1, 'reference_used': False,
            'decision_index': len(self.policy.records)-1})
        return selected

    def _pointer_selection_context(self, role, ids, required):
        return {}
