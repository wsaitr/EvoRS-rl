"""Pinned compiled field grammar; sampling and likelihood replay share masks.

This is an explicit new decoder protocol. It does not claim equality to the
historical Python grammar, including that grammar's token-budget closure rule.
"""
import hashlib
from importlib.metadata import version
import json


BACKEND = 'xgrammar-0.2.8-fields-v1'


def field_json_schema(fields):
    def kind_schema(kind):
        if isinstance(kind, str):
            if kind == 'nullable_integer':
                return {'type': ['integer', 'null']}
            if kind in ('string', 'number', 'integer', 'object', 'array'):
                return {'type': kind}
            raise ValueError('specialized grammar requires the historical backend: ' + kind)
        if not isinstance(kind, dict):
            raise ValueError('declared field type required')
        if set(kind) == {'enum'}:
            return {'type': 'string', 'enum': kind['enum']}
        if set(kind) == {'object'}:
            return field_json_schema(kind['object'])
        if set(kind) == {'tuple'}:
            return {'type': 'array', 'prefixItems': [kind_schema(v) for v in kind['tuple']],
                    'minItems': len(kind['tuple']), 'maxItems': len(kind['tuple'])}
        if set(kind) in ({'array', 'maximum'}, {'array', 'minimum', 'maximum'}):
            return {'type': 'array', 'items': kind_schema(kind['array']),
                    'minItems': kind.get('minimum', 0), 'maxItems': kind['maximum']}
        raise ValueError('unsupported compiled field schema')
    names = [name for name, _ in fields]
    if not names or len(set(names)) != len(names):
        raise ValueError('distinct required fields must be declared')
    return {'type': 'object', 'properties': {n: kind_schema(k) for n, k in fields},
            'required': names, 'additionalProperties': False}


class CompiledFieldGrammar:
    backend = BACKEND

    def __init__(self, compiler, fields, vocab_size):
        import torch
        import xgrammar as xgr
        if version('xgrammar') != '0.2.8':
            raise ValueError('the declared decoder requires xgrammar 0.2.8')
        self.schema = field_json_schema(fields)
        self.schema_sha256 = hashlib.sha256(
            json.dumps(self.schema, sort_keys=True).encode()).hexdigest()
        self.compiled = compiler.compile_json_schema(
            self.schema, any_whitespace=True, strict_mode=True, any_order=False)
        self.matcher = xgr.GrammarMatcher(self.compiled)
        self.mask = torch.zeros((1, (vocab_size + 31) // 32), dtype=torch.int32)
        self.vocab_size = vocab_size
        self.prefix = ()
        self.cached_allowed = None

    def _advance(self, prefix):
        prefix = tuple(prefix)
        if prefix[:len(self.prefix)] != self.prefix:
            self.matcher.reset()
            self.prefix = ()
            self.cached_allowed = None
        for token in prefix[len(self.prefix):]:
            if not self.matcher.accept_token(token):
                raise ValueError('recorded token violates its compiled field grammar')
            self.cached_allowed = None
        self.prefix = prefix

    def allowed(self, prefix, token_budget=None):
        import numpy as np
        self._advance(prefix)
        if self.cached_allowed is None:
            self.matcher.fill_next_token_bitmask(self.mask)
            bits = np.unpackbits(self.mask.numpy().view(np.uint8), bitorder='little')
            self.cached_allowed = tuple(np.flatnonzero(bits[:self.vocab_size]).tolist())
        return self.cached_allowed

    def complete(self, prefix):
        self._advance(prefix)
        return self.matcher.is_completed()
