"""Lazy canonical JSON grammar for a bounded set of real evidence addresses.

Does not enumerate subsets/permutations. Token pieces are traversed against
character states; sampling and likelihood replay use exactly the same language.
"""
import json
from functools import lru_cache


class PointerTokenLexicon:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        if not isinstance(self.eos, int):
            raise ValueError('one EOS token required')
        ids = sorted(set(tokenizer.get_vocab().values())-set(tokenizer.all_special_ids))
        pieces = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
        self.root = {}
        for token, piece in zip(ids, pieces):
            # json.dumps escapes non-ASCII addresses, so the language is ASCII.
            if not piece or not piece.isascii():
                continue
            node = self.root
            for char in piece:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(token)


class EvidencePointerGrammar:
    def __init__(self, lexicon, specification, feasibility=None):
        if set(specification) != {'allowed', 'required', 'maximum'}:
            raise ValueError('invalid pointer grammar specification')
        allowed, required = specification['allowed'], specification['required']
        if (not isinstance(allowed, (list, tuple)) or not allowed or
                any(not isinstance(x, str) or not x for x in allowed) or len(set(allowed)) != len(allowed)):
            raise ValueError('pointer grammar requires distinct actual IDs')
        if (not isinstance(required, (list, tuple)) or any(not isinstance(x, str) for x in required)
                or len(set(required)) != len(required) or not set(required) <= set(allowed)):
            raise ValueError('required pointer IDs must be actual distinct IDs')
        maximum = specification['maximum']
        if type(maximum) != int or maximum < 1 or len(required) > maximum:
            raise ValueError('unsatisfiable pointer cap')
        self.feasibility = feasibility
        self.lexicon = lexicon
        self.allowed_ids = tuple(allowed); self.required_ids = frozenset(required)
        self.maximum = maximum
        # States: (phase, remaining_literal, selected_ID_tuple, pending_ID).
        self.initial = frozenset({('header', '{"evidence_ids":[', (), None)})

    @lru_cache(maxsize=8192)
    def transitions(self, states, char):
        result = set()
        for phase, tail, selected, pending in states:
            if phase == 'select':
                missing = self.required_ids-set(selected)
                choices = missing if len(selected)+len(missing) == self.maximum else self.allowed_ids
                for eid in choices:
                    if eid in selected or (self.feasibility is not None and not self.feasibility((*selected, eid), False)):
                        continue
                    literal = json.dumps(eid, ensure_ascii=True)
                    if literal[0] == char:
                        result.add(('id', literal[1:], selected, eid))
            elif phase == 'after':
                # A valid closing subset may have no affordable extension. Do
                # not emit a comma unless some actual next ID can still reach
                # a valid closure; otherwise the next decoding step dead-ends.
                if (char == ',' and len(selected) < min(self.maximum, len(self.allowed_ids))
                        and any(eid not in selected and (self.feasibility is None or
                            self.feasibility((*selected, eid), False)) for eid in self.allowed_ids)):
                    result.add(('select', '', selected, None))
                elif char == ']' and self.required_ids <= set(selected) and (self.feasibility is None or self.feasibility(selected, True)):
                    result.add(('close', '}', selected, None))
            elif phase in {'header', 'id', 'close'} and tail and tail[0] == char:
                rest = tail[1:]
                if rest:
                    result.add((phase, rest, selected, pending))
                elif phase == 'header':
                    result.add(('select', '', selected, None))
                elif phase == 'id':
                    result.add(('after', '', (*selected, pending), None))
                else:
                    result.add(('end', '', selected, None))
        return frozenset(result)

    @lru_cache(maxsize=4096)
    def states_for_text(self, text):
        states = self.initial
        for char in text:
            states = self.transitions(states, char)
            if not states:
                raise ValueError('completion violates evidence pointer grammar')
        return states

    def decoded_states(self, prefix):
        ended = False
        if any(token in self.lexicon.tokenizer.all_special_ids for token in prefix):
            if not prefix or prefix[-1] != self.lexicon.eos:
                raise ValueError('unexpected special token in pointer completion')
            prefix = prefix[:-1]
            ended = True
        text = self.lexicon.tokenizer.decode(prefix, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
        states = self.states_for_text(text)
        if ended and not any(state[0] == 'end' for state in states):
            raise ValueError('EOS before a complete pointer')
        return states

    def complete(self, prefix):
        return any(state[0] == 'end' for state in self.decoded_states(prefix))

    def allowed(self, prefix):
        states = self.decoded_states(prefix)
        if any(state[0] == 'end' for state in states):
            return (self.lexicon.eos,)
        return self.allowed_states(states)

    @lru_cache(maxsize=4096)
    def allowed_states(self, states):
        tokens = set()
        def visit(node, current):
            tokens.update(node.get(None, ()))
            for char, child in node.items():
                if char is None:
                    continue
                following = self.transitions(current, char)
                if following:
                    visit(child, following)
        # Empty pieces are excluded, so only complete nonempty token pieces count.
        for char, child in self.lexicon.root.items():
            if char is None:
                continue
            following = self.transitions(states, char)
            if following:
                visit(child, following)
        if not tokens:
            raise ValueError('tokenizer cannot extend legal evidence pointer prefix')
        return tuple(sorted(tokens))
