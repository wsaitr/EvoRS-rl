"""Optional, sampled private search; the historical fixed search stays intact."""
from .policy_terminal import PolicyTerminalDomainRuntime
from .rejection_feedback import RejectionFeedbackRuntime
from .runtime import WorkflowRuntime, schema


class AdaptivePrivateSearchRuntime(PolicyTerminalDomainRuntime):
    adaptive_private_search=True
    def _private_search_actions(self, selected, mode, remaining, search_calls):
        search_modes = super()._possible_modes('reasoning', selected, include_dispatched=True)
        legal = [{'private_search': 'skip'}]
        if mode in search_modes and remaining >= search_calls+2:
            legal.append({'private_search': 'run'})
        return legal

    def _private_search_action_fields(self, action):
        return {'private_search'}

    def _private_search_context(self):
        return {}

    def _private_search_instruction(self):
        return ('Choose whether to answer directly from the selected evidence or run '
            'private candidate revision and source-group fitness checks. Use run '
            'when re-examining these same observations is worth its actual cost. '
            'Use skip to let reasoning answer directly; skip does not submit an '
            'answer or certify it. New observations remain separate workflow actions.')

    def _reasoning_call_bound(self):
        # One dispatch plus one direct completion is always the cheap branch.
        return 2 if self.mechanisms.evolution else 1

    def _possible_modes(self, role, selected, *, include_dispatched=False):
        if role == 'reasoning' and self.mechanisms.evolution:
            # Do not require the optional fitness routes to afford direct reasoning.
            return RejectionFeedbackRuntime._possible_modes(
                self, role, selected, include_dispatched=include_dispatched)
        return super()._possible_modes(role, selected, include_dispatched=include_dispatched)

    def _expert(self, role, context, packets, selected):
        if role != 'reasoning' or not self.mechanisms.evolution:
            return super()._expert(role, context, packets, selected)
        remaining = self.max_model_calls-self._model_calls_used()
        if remaining < 3:
            raise ValueError('dispatch, direct reasoning and final submission require three calls')
        variations = self.mechanisms.generations*(self.mechanisms.population_size-1)
        evaluations = 1+variations
        groups = self._search_groups(selected)
        # Include the inherited conservative post-seed budget check as well as
        # actual per-source fitness calls. Reserve dispatch and one submission.
        search_calls = max(1+2*variations+evaluations*len(groups),
                           2+evaluations+2*variations)
        mode = self.last_modes.get('reasoning', 'tokens')
        legal = self._private_search_actions(selected, mode, remaining, search_calls)
        action = self._call('controller',
            self._private_search_instruction(),
            # These packets were routed to reasoning, not to the controller.
            # Select the search plan from its metadata; keep the original grant
            # and pixels for the receiving expert's execution.
            (), {**context, 'selected_evidence_ids': list(selected),
                      'legal_action_examples': legal,
                      'private_search_model_call_bound': search_calls,
                      'direct_reasoning_model_calls': 1,
                      'remaining_model_calls_before_dispatch': remaining,
                      **self._private_search_context()})
        schema(action, self._private_search_action_fields(action))
        if action not in legal:
            raise ValueError('unaffordable or unknown private search action')
        self.events.append({'kind': 'adaptive_private_search_dispatch',
            'action': action['private_search'], 'decision_index': len(self.policy.records)-1,
            **({'fitness_mode': action['fitness_mode']} if 'fitness_mode' in action else {}),
            'source_evidence_ids': list(selected), 'reference_used': False,
            'search_model_call_bound': search_calls, 'selection_model_calls': 1})
        if action['private_search'] == 'skip':
            # Same evidence, output validation and candidate commit as the base
            # direct reasoning stage. No fabricated private trace or fitness.
            return WorkflowRuntime._expert(self, role, context, packets, selected)
        if 'fitness_mode' not in action:
            return super()._expert(role, context, packets, selected)
        # The same sampled controller turn chooses this opt-in representation.
        # It does not alter the existing reasoning grant or add a model call.
        previous = getattr(self, '_active_private_fitness_mode', None)
        self._active_private_fitness_mode = action['fitness_mode']
        try:
            return super()._expert(role, context, packets, selected)
        finally:
            if previous is None:
                del self._active_private_fitness_mode
            else:
                self._active_private_fitness_mode = previous
