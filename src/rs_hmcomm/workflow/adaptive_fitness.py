"""Sample source-faithful private fitness representation within the workflow.

The reasoning route is already paid when this choice is made. Reserve all real
candidate/source checks and one affordable closing route. No pixels are granted
to the dispatching controller; the normal router bills the receiving verifier.
"""
from contextlib import contextmanager

from .grounded_agent import GroundedAgentRuntime
from .rejection_feedback import RejectionFeedbackRuntime


class AdaptiveFitnessRuntime(GroundedAgentRuntime):
    @contextmanager
    def _fitness_source_scope(self):
        previous = getattr(self, '_private_fitness_source_scope', False)
        self._private_fitness_source_scope = True
        try:
            yield
        finally:
            self._private_fitness_source_scope = previous

    def _anchored(self, role, selected):
        if role == 'verification' and getattr(self, '_private_fitness_source_scope', False):
            # A global multi-frame overview must not enter a single-source vote.
            # An overview belonging to this group is already among its IDs.
            return tuple(selected)
        return super()._anchored(role, selected)

    def _fitness_transaction_options(self, selected):
        groups = self._search_groups(selected)
        if not groups:
            return []
        closing_modes = RejectionFeedbackRuntime._possible_modes(
            self, 'verification', selected, include_dispatched=True)
        if not closing_modes:
            return []
        closing = min(self._route_bound('verification', selected, m) for m in closing_modes)
        evaluations = 1+self.mechanisms.generations*(self.mechanisms.population_size-1)
        remaining = self.max_packet_bytes-self.cost.message_latent_bytes-self.cost.message_structure_bytes
        result = []
        with self._fitness_source_scope():
            per_group_modes = [RejectionFeedbackRuntime._possible_modes(
                self, 'verification', group, include_dispatched=True) for group in groups]
            for mode in self.mechanisms.modes:
                if not all(mode in available for available in per_group_modes):
                    continue
                bound = evaluations*sum(self._route_bound('verification', group, mode) for group in groups)
                if bound+closing <= remaining:
                    result.append({'fitness_mode': mode, 'fitness_wire_bound': bound,
                                   'closing_wire_reserve': closing, 'remaining_wire_bytes': remaining})
        return result

    def _private_search_actions(self, selected, mode, remaining, search_calls):
        options = self._fitness_transaction_options(selected)
        self._last_fitness_transaction_options = options
        self.events.append({'kind': 'adaptive_fitness_transaction_mask',
            'reasoning_mode_already_paid': mode, 'source_evidence_groups': self._search_groups(selected),
            'options': options, 'remaining_model_calls_before_dispatch': remaining,
            'search_model_call_bound': search_calls, 'reserved_dispatch_and_submission_calls': 2,
            'reference_used': False, 'extra_model_calls': 0})
        legal = [{'private_search': 'skip'}]
        if remaining >= search_calls+2:
            legal.extend({'private_search': 'run', 'fitness_mode': row['fitness_mode']} for row in options)
        return legal

    def _private_search_action_fields(self, action):
        return ({'private_search', 'fitness_mode'} if isinstance(action, dict)
                and action.get('private_search') == 'run' else {'private_search'})

    def _private_search_context(self):
        return {'private_fitness_transaction_options': self._last_fitness_transaction_options,
                'fitness_costs_include_all_candidates_and_sources': True,
                'reasoning_route_is_already_paid': True}

    def _private_search_instruction(self):
        return (super()._private_search_instruction()+' When choosing run, also choose fitness_mode '
            'for independent checks of the original source groups. This may differ from the reasoning '
            'representation. tokens sends the authorized native visual evidence; compressed sends '
            'its learned representation; text sends only authorized descriptions; temporal requires '
            'a registered pair inside each group. Use only a currently listed affordable action. '
            'Native verification can re-examine labels and fine spatial details lost in descriptions '
            'or compression, but is not guaranteed correct. Agreement is not ground truth.')

    def _private_fitness_route(self, selected, mode):
        with self._fitness_source_scope():
            context, packets, routed = super()._private_fitness_route(selected, mode)
        self.events.append({'kind': 'adaptive_fitness_source_route', 'mode': mode,
            'decision_index_before_fitness': len(self.policy.records),
            'source_evidence_ids': list(selected), 'actual_routed_evidence_ids': list(routed),
            'actual_packet_ids': list(packets), 'cross_source_global_anchor_added': False,
            'reference_used': False, 'extra_model_calls': 0})
        return context, packets, routed
