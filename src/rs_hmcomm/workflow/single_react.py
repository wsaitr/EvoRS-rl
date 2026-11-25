"""Opt-in single-Agent Thought/Action/Observation loop; no expert dispatch.

Adapted from the interaction contract in ThinkGeo's pinned Lagent ReAct agent,
not a reproduction of its model, perception tools, or benchmark scores.
"""
import json

VERSION = 'single-react-development-r1'
UPSTREAM = {
    'repository': 'https://github.com/mbzuai-oryx/ThinkGeo',
    'commit': 'c8108de440db9920d02b9de3f91dffb457c8afd4',
    'path': 'opencompass/opencompass/lagent/agents/react.py',
}


def planning_schema(actions):
    if not actions:
        raise ValueError('at least one legal action required')
    return [['analysis', 'string'], ['action', {'enum': [str(i) for i in range(len(actions))]}]]


def planning_prompt(question, history, actions):
    return (
        'You are one remote-sensing tool-using Agent. Solve the task with a '
        'Thought/Action/Observation loop. First identify what is already visible '
        'and what fact is still needed. Compare the public answer alternatives. '
        'Inspect a crop when a small label, legend, symbol or spatial detail is '
        'unclear; use calculate only with explicit public or observed numbers. '
        'After an observation, update your interpretation and choose the next '
        'useful action. Previous analysis is your hypothesis, not independent '
        'evidence. Do not repeat a failed request without repairing its cause. '
        'Choose answer when the observations are sufficient. No required tool '
        'quota. Keep analysis within 80 words. Return analysis followed by the '
        'selected action code, using exactly the declared JSON fields.\nTask: '
        + question + '\nActual execution trace: ' + json.dumps(history)
        + '\nLegal action codes: ' + json.dumps({str(i): a for i, a in enumerate(actions)})
    )


def resolve_action(response, actions):
    if set(response) != {'analysis', 'action'} or not isinstance(response['analysis'], str):
        raise ValueError('ReAct response requires analysis and action')
    codes = {str(i): a for i, a in enumerate(actions)}
    code = response['action']
    if not isinstance(code, str) or code not in codes:
        raise ValueError('action code outside current legal actions')
    return dict(codes[code])
