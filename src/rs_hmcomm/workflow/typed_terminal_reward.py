"""Strict external answer contracts, not a general natural-language judge.

Requires explicit typed annotations. No label inference from model output,
implicit pixel-to-meter conversion, word-overlap fallback or substring match.
"""
import math
import re
import unicodedata
from decimal import Decimal


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
COUNT_WORDS = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve '
                       'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty').split(), range(21)))
UNITS = {
    'mm': ('length', .001), 'cm': ('length', .01), 'm': ('length', 1.), 'km': ('length', 1000.),
    'millimeter': ('length', .001), 'millimeters': ('length', .001),
    'centimeter': ('length', .01), 'centimeters': ('length', .01),
    'meter': ('length', 1.), 'meters': ('length', 1.), 'metre': ('length', 1.), 'metres': ('length', 1.),
    'kilometer': ('length', 1000.), 'kilometers': ('length', 1000.),
    'cm2': ('area', .0001), 'm2': ('area', 1.), 'km2': ('area', 1000000.),
    'square meter': ('area', 1.), 'square meters': ('area', 1.),
    'square metre': ('area', 1.), 'square metres': ('area', 1.),
    'px': ('pixel_length', 1.), 'pixel': ('pixel_length', 1.), 'pixels': ('pixel_length', 1.),
    '%': ('percentage', 1.), 'percent': ('percentage', 1.),
}


def normalized(text):
    return unicodedata.normalize('NFKC', text or '').strip().casefold()


def finite_nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('annotation must be a finite nonnegative number')
    return float(value)


def typed_terminal_score(answer, reference):
    metric = reference['metric']
    text = normalized(answer)
    if metric == 'count_exact':
        expected = reference['value']
        if type(expected) != int or expected < 0:
            raise ValueError('count annotation must be a nonnegative integer')
        if text in COUNT_WORDS:
            return float(COUNT_WORDS[text] == expected)
        if not re.fullmatch(r'\d+', text):
            return 0.
        return float(int(text) == expected)
    if metric == 'polar_exact':
        expected = normalized(reference['value'])
        if expected not in {'yes', 'no'}:
            raise ValueError('polar annotation must be Yes or No')
        # A canonical answer avoids accepting contradictory trailing claims.
        return float(text in {'yes', 'no'} and text == expected)
    if metric != 'quantity_range':
        raise ValueError('unknown typed terminal metric')
    unit = normalized(reference['unit'])
    if unit not in UNITS:
        raise ValueError('unsupported annotated quantity unit')
    bounds = reference['range']
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError('explicit two-value annotation range required')
    low, high = map(finite_nonnegative, bounds)
    if low > high:
        raise ValueError('annotation range is reversed')
    match = re.fullmatch(r'('+NUMBER+r')\s*([a-z0-9% ]+)', text)
    if not match:
        return 0.
    value = Decimal(match.group(1)); predicted_unit = match.group(2).strip()
    if not value.is_finite() or value < 0 or predicted_unit not in UNITS:
        return 0.
    dimension, scale = UNITS[unit]; predicted_dimension, predicted_scale = UNITS[predicted_unit]
    if dimension != predicted_dimension:
        return 0.
    canonical = value*Decimal(str(predicted_scale))
    return float(Decimal(str(low))*Decimal(str(scale)) <= canonical <=
                 Decimal(str(high))*Decimal(str(scale)))
