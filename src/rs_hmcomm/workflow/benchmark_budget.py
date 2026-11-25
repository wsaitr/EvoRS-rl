"""Expose true remaining resources; no extra calls or unverified acceptance."""
from .native_benchmark_tools import TaskBoundNativeRuntime, ObservingBenchmarkSingle


class BudgetObservationMixin:
    def action_context(self, public, history, cost):
        context = super().action_context(public, history, cost)
        # Includes planning, domain guidance, actual tool VLM calls and review.
        used = len(self.policy.records)
        remaining = max(0, self.max_calls-used)
        closing = 3 if self.multi else 1
        return {**context, 'execution_budget': {
            'maximum_model_calls': self.max_calls,
            'actual_model_calls_used': used,
            'remaining_model_calls_including_current_action': remaining,
            'maximum_tool_calls': self.max_tools,
            'actual_tool_calls_used': cost.tool_calls,
            'remaining_tool_calls': max(0, self.max_tools-cost.tool_calls),
            'closing_model_calls_including_answer_action': closing,
            'closing_sequence': ['answer_action', 'final_reasoning', 'verification']
                if self.multi else ['answer_action'],
            'can_close_with_current_budget': remaining >= closing,
            'inspect_or_detect_model_calls_including_tool_action': 2,
            'instruction': 'Once evidence is sufficient, submit the answer while the complete closing sequence still fits. Use a further visual tool only when its two calls and the required closing sequence fit. An unverified multi-agent proposal is never a valid final answer.'},
            'budget_reference_used': False}


class BudgetTaskBoundRuntime(BudgetObservationMixin, TaskBoundNativeRuntime):
    pass


class BudgetObservingSingle(BudgetObservationMixin, ObservingBenchmarkSingle):
    pass
