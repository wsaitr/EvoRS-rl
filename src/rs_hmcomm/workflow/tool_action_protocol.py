"""Lossless tool-envelope normalization; arguments still require tool validation."""
from .runtime import strict_json


def parse_tool_action(text):
    value=strict_json(text)
    if not isinstance(value,dict):raise ValueError('tool action must be an object')
    if set(value)=={'action'}:
        value=value['action']
    if not isinstance(value,dict) or not isinstance(value.get('tool'),str) or not value['tool'].strip():
        raise ValueError('expected top-level tool action or exactly one action envelope')
    if 'action' in value:raise ValueError('ambiguous nested tool action')
    return value
