"""External task scoring. Reference labels never belong to runtime context.

Caption token F1 is a transparent development reward, not CIDEr/METEOR.
Official corpus caption scores must be computed separately on saved outputs.
"""
import json
import re
import unicodedata
from collections import Counter


def tokens(text):
    return re.findall(r'\w+',unicodedata.normalize('NFKC',text or '').casefold())


def token_f1(prediction,reference):
    p,r=Counter(tokens(prediction)),Counter(tokens(reference))
    overlap=sum((p&r).values())
    return 2*overlap/max(1,sum(p.values())+sum(r.values()))


def task_score(answer,reference):
    if isinstance(reference,str):
        if not reference.strip():raise ValueError('empty external reference')
        return float(tokens(answer)==tokens(reference))
    if not isinstance(reference,dict):raise ValueError('typed external reference required')
    metric=reference.get('metric');values=reference.get('answers')
    if metric in {'count_exact', 'polar_exact', 'quantity_range'}:
        from .typed_terminal_reward import typed_terminal_score
        return typed_terminal_score(answer, reference)
    if metric in {'exact','caption_token_f1'}:
        if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() for v in values):
            raise ValueError('references must be nonempty strings')
        fn=token_f1 if metric=='caption_token_f1' else lambda a,b:float(tokens(a)==tokens(b))
        return max(fn(answer,v) for v in values)
    if metric=='bbox_iou':
        from .semantic import bbox_iou
        # Invalid or missing predicted boxes score zero; invalid references fail.
        target=reference['bbox'];bbox_iou(target,target)
        try:
            value=json.loads(answer);return bbox_iou(value['bbox'],target)
        except (ValueError,TypeError,KeyError,json.JSONDecodeError):return 0.
    raise ValueError('unknown task metric')
