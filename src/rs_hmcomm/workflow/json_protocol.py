"""Output framing and an object-root mask, independent of task labels."""
import json


def complete_object(text):
    """Return the first complete object frame; preserve raw bytes/duplicate keys.

    A subword token can contain both the closing brace and trailing characters.
    The model's original response remains in the rollout; framing only controls
    the action passed to the strict downstream parser.
    """
    framed=text.strip()
    if framed.startswith('```json'):framed=framed[7:].lstrip()
    elif framed.startswith('```'):framed=framed[3:].lstrip()
    try:
        value,end=json.JSONDecoder().raw_decode(framed)
    except (ValueError,json.JSONDecodeError):return None
    return framed[:end] if isinstance(value,dict) else None


class ObjectRootGrammar:
    """Require an object root, without constraining free-form field contents.

    This is deliberately NOT a full JSON Schema grammar. Nested types, required
    keys and semantic constraints are still validated by the action parser.
    Sampling and PPO likelihood replay must use the identical root mask.
    """
    def __init__(self,tokenizer):
        ids=sorted(set(tokenizer.get_vocab().values())-set(tokenizer.all_special_ids))
        pieces=tokenizer.batch_decode([[i] for i in ids],skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
        self.starts=set();self.whitespace=set()
        for token,piece in zip(ids,pieces):
            stripped=piece.lstrip(' \r\n\t')
            if stripped.startswith('{'):self.starts.add(token)
            elif piece and not stripped:self.whitespace.add(token)
        if not self.starts:raise ValueError('tokenizer cannot start a JSON object')
        self.root_tokens=tuple(sorted(self.starts|self.whitespace))

    def allowed(self,prefix):
        for token in prefix:
            if int(token) in self.starts:return None
            if int(token) not in self.whitespace:raise ValueError('completion violates JSON object root')
        return self.root_tokens
