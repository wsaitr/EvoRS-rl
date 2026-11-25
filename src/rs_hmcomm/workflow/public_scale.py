"""Metric scale accepted only from explicitly stated public GSD."""
import math
import re


def public_pixel_scale(question):
    number = r'([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)'
    pattern = r'\bGSD\s*(?:of|:|=)?\s*'+number+r'\s*(?:m\s*/\s*(?:pixel|px)|meters?\s+per\s+pixel)\b'
    scales = [float(s) for s in re.findall(pattern, question, re.I)]
    if not scales:
        return None
    if any(not math.isfinite(s) or s <= 0 for s in scales) or len(set(scales)) != 1:
        raise ValueError('ambiguous or invalid public GSD')
    return scales[0]
