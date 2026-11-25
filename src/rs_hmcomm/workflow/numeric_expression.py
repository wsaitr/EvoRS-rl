"""Incremental ASCII numeric-expression syntax; never evaluates model values.

Recognizes decimal/scientific numbers, arithmetic, grouping and supported
scalar calls. Numeric ranges, division, units and evidence stay caller checks.
"""
import ast
from functools import lru_cache

FUNCTION_ARITY={'sqrt':(1,1),'abs':(1,1),'min':(2,128),'max':(2,128),'round':(1,2)}
START=('start',(), 'operand','','')


def valid_number(text):
    try:
        node=ast.parse(text,mode='eval').body
        return isinstance(node,ast.Constant) and type(node.value) in (int,float)
    except (SyntaxError,ValueError):return False


def raw_step(state,char):
    phase,stack,mode,token,text=state
    if phase=='done':return None
    if phase=='start':
        if char in ' \r\n\t':return state
        return ('in',(), 'operand','','') if char=='"' else None
    if char=='"':
        if mode.startswith('n_') and mode not in ('n_dot','n_exp','n_exp_sign') and valid_number(token):mode='operator'
        return ('done',stack,mode,token,text) if mode=='operator' and not stack and text else None
    if char not in '0123456789eE.+-*/%(), '+''.join(FUNCTION_ARITY):return None
    new_text=text+char
    def value(next_mode,next_token='',next_stack=stack):return ('in',next_stack,next_mode,next_token,new_text)
    if mode.startswith('n_'):
        if char.isdigit():
            next_mode='n_exp_digits' if mode in ('n_exp','n_exp_sign','n_exp_digits') else 'n_frac' if mode in ('n_dot','n_frac') else 'n_int'
            return value(next_mode,token+char)
        if mode=='n_int' and char=='.':return value('n_frac',token+char)
        if mode in ('n_int','n_frac') and char in 'eE':return value('n_exp',token+char)
        if mode=='n_exp' and char in '+-':return value('n_exp_sign',token+char)
        if mode in ('n_dot','n_exp','n_exp_sign') or not valid_number(token):return None
        mode='operator'
    if mode in ('name','name_wait'):
        if char=='(' and token in FUNCTION_ARITY:
            return value('operand',next_stack=stack+(('call',token,0),)) if len(stack)<32 else None
        if char==' ' and token in FUNCTION_ARITY:return value('name_wait',token)
        if mode=='name' and any(fn.startswith(token+char) for fn in FUNCTION_ARITY):return value('name',token+char)
        return None
    if mode in ('star','slash'):
        if mode=='star' and char=='*':return value('operand')
        if mode=='slash' and char=='/':return None
        mode='operand'
    if mode=='operand':
        if char==' ':return value('operand') if text else None
        if char.isdigit():return value('n_int',char)
        if char=='.':return value('n_dot',char)
        if char in '+-':return value('operand')
        if char=='(':
            return value('operand',next_stack=stack+(('group','',0),)) if len(stack)<32 else None
        if any(fn.startswith(char) for fn in FUNCTION_ARITY):return value('name',char)
        return None
    if mode=='operator':
        if char==' ':return value('operator')
        if char in '+-%':return value('operand')
        if char=='*':return value('star')
        if char=='/':return value('slash')
        if char==')' and stack:
            kind,fn,args=stack[-1]
            if kind=='call' and not FUNCTION_ARITY[fn][0]<=args+1<=FUNCTION_ARITY[fn][1]:return None
            return value('operator',next_stack=stack[:-1])
        if char==',' and stack and stack[-1][0]=='call':
            kind,fn,args=stack[-1]
            if args+1>=FUNCTION_ARITY[fn][1]:return None
            return value('operand',next_stack=stack[:-1]+((kind,fn,args+1),))
    return None


@lru_cache(maxsize=16384)
def closure(state):
    suffix=[];current=state
    while current[0]!='done':
        phase,stack,mode,token,text=current
        if phase=='start':piece='"0"'
        elif mode=='name':piece=min((fn[len(token):]+'(' for fn in FUNCTION_ARITY if fn.startswith(token)),key=len)
        elif mode=='name_wait':piece='('
        elif mode in ('operand','star','slash','n_dot','n_exp','n_exp_sign'):piece='0'
        elif mode=='n_int' and not valid_number(token):piece='.0'
        elif mode=='operator' or mode in ('n_int','n_frac','n_exp_digits'):
            if not stack:piece='"'
            elif stack[-1][0]=='call' and stack[-1][2]+1<FUNCTION_ARITY[stack[-1][1]][0]:piece=',0'
            else:piece=')'
        else:raise ValueError('no numeric expression closure')
        suffix.append(piece)
        for char in piece:
            current=raw_step(current,char)
            if current is None:raise ValueError('invalid numeric closure')
    return ''.join(suffix)


@lru_cache(maxsize=65536)
def step(state,char):
    result=raw_step(state,char)
    if result is None:return None
    # Reject a prefix before it consumes room needed for its verified closure.
    suffix=closure(result) if result[0]!='done' else ''
    length=len(result[4])+len(suffix)-(1 if suffix.endswith('"') else 0)
    return result if length<=256 else None
