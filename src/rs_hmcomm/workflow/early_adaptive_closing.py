"""Source-certified first attempts with the existing adaptive budget controls."""
from .early_reasoning import EarlyReasoningRuntime
from .adaptive_closing_budget import AdaptiveClosingBudgetRuntime


class EarlyAdaptiveClosingRuntime(EarlyReasoningRuntime, AdaptiveClosingBudgetRuntime):
    def _controller_role_choices(self, choices, context):
        legal = AdaptiveClosingBudgetRuntime._controller_role_choices(self, choices, context)
        return EarlyReasoningRuntime._controller_role_choices(self, legal, context)
