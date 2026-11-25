"""Reject literal schema instructions, without claiming semantic grounding."""
import re

_PLACEHOLDERS = frozenset({
    'name', 'category', 'category name', 'class', 'class name', 'label',
    'actual class', 'actual class name', 'actual observed class',
    'actual observed class string', 'actual observed class name',
    'target objects', 'object',
})


def usable_object_label(value):
    """Only a syntax gate: accepted labels can still be visually incorrect."""
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        return False
    normalized = re.sub(r'[\s_-]+', ' ', value.strip().casefold())
    return normalized not in _PLACEHOLDERS
