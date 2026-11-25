"""Keep the latest valid sampled plan; invalid revisions cannot erase it."""
from .runtime import strict_json


def validated_plan(text, allowed):
    flow = strict_json(text)['tool_flow']
    if not isinstance(flow, list) or any(not isinstance(t, str) or t not in allowed for t in flow):
        raise ValueError('unknown plan tool or invalid plan list')
    return flow


def commit_plan(initial, revision, allowed):
    """No reference-based selection or dependency-correctness claim."""
    accepted = None
    events = []
    for stage, text in [('initial', initial), ('revision', revision)]:
        if text is None:
            continue
        try:
            accepted = validated_plan(text, allowed)
            events.append({'stage': stage, 'valid': True, 'committed': True})
        except (ValueError, KeyError, TypeError) as error:
            events.append({'stage': stage, 'valid': False, 'committed': False,
                           'reason': str(error), 'prior_valid_plan_retained': accepted is not None})
    return accepted, events
