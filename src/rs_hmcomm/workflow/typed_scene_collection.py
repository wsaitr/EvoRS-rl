"""Public-schema action grammar for closed-choice scene collections."""
import json
from .all_source_collection import AllSourceSceneCollectionRuntime
from .scene_collection_expert import public_scene_classes
from .runtime import question_choices


class TypedSceneCollectionRuntime(AllSourceSceneCollectionRuntime):
    def tool_action_choices(self, public, history, cost):
        classes = public_scene_classes(public['question'])
        answers = tuple(dict.fromkeys(option[0] for option in question_choices(public['question'])))
        if classes is None or not answers:
            return None
        context = self.action_context(public, history, cost)
        state = context['collection_observation_state']
        missing = state['not_yet_observed_assets']
        remaining = self.max_calls-len(self.policy.records)
        closing = 3 if self.multi else 1
        choices = []
        successful = {json.dumps(item['action'], sort_keys=True) for item in history
                      if 'action' in item and 'result' in item}
        if not missing and remaining >= closing:
            choices += [{'tool': 'answer', 'answer': a} for a in answers]
        if cost.tool_calls < self.max_tools:
            # No Qwen output turn is hidden in classifier execution; native
            # encoding and auxiliary latency are still charged by the tool.
            if missing and len(missing) <= 32 and remaining >= 1+closing:
                choices.append({'tool': 'classify_scene_collection', 'assets': missing, 'labels': classes})
            candidates = ([missing] if missing and len(missing) <= 32 else [])
            candidates += [[name] for name in public['assets']]
            for names in candidates:
                action = {'tool': 'inspect_collection', 'assets': names, 'question': public['question']}
                if (remaining >= 1+len(names)+closing
                        and json.dumps(action, sort_keys=True) not in successful
                        and action not in choices):
                    choices.append(action)
        # Empty means no complete tool/answer transaction fits. The caller
        # retains this failure rather than inventing a valid final answer.
        return choices
