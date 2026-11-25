"""Incremental JSON syntax mask; field schemas and facts remain downstream checks."""
from functools import lru_cache
from collections import OrderedDict


# State: container stack, lexical mode, escape state, remaining literal.
START=((), 'root', '', '')


def finish_value(stack):
    if not stack:return ((), 'done', '', '')
    return (stack[:-1]+((stack[-1][0], 'comma'),), 'normal', '', '')


@lru_cache(maxsize=65536)
def advance(state, char):
    stack,mode,escape,literal=state
    if mode=='done':return state if char in ' \r\n\t' else None
    if mode=='root':
        if char in ' \r\n\t':return state
        return ((('object','first'),),'normal','','') if char=='{' else None
    if mode in ('key','string'):
        if escape.startswith('u'):
            if char not in '0123456789abcdefABCDEF':return None
            left=int(escape[1:])-1
            return (stack,mode,'u'+str(left) if left else '', '')
        if escape=='slash':
            if char=='u':return (stack,mode,'u4','')
            return (stack,mode,'','') if char in '"\\/bfnrt' else None
        if ord(char)<32:return None
        if char=='\\':return (stack,mode,'slash','')
        if char=='"':
            if mode=='key':return (stack[:-1]+(('object','colon'),),'normal','','')
            return finish_value(stack)
        return state
    if mode=='literal':
        if not literal or char!=literal[0]:return None
        return (stack,'literal','',literal[1:]) if len(literal)>1 else finish_value(stack)
    if mode=='number':
        transitions={'sign':{'0':'zero','digit':'int'},'zero':{'.':'dot','e':'exp','E':'exp'},
            'int':{'0':'int','digit':'int','.':'dot','e':'exp','E':'exp'},
            'dot':{'0':'frac','digit':'frac'},'frac':{'0':'frac','digit':'frac','e':'exp','E':'exp'},
            'exp':{'+':'exp_sign','-':'exp_sign','0':'exp_digits','digit':'exp_digits'},
            'exp_sign':{'0':'exp_digits','digit':'exp_digits'},
            'exp_digits':{'0':'exp_digits','digit':'exp_digits'}}
        category='digit' if char in '123456789' else char
        next_mode=transitions[escape].get(category)
        if next_mode:return (stack,'number',next_mode,'')
        if escape in ('zero','int','frac','exp_digits') and char in ' \r\n\t,]}':
            return advance(finish_value(stack),char)
        return None
    if char in ' \r\n\t':return state
    kind,expect=stack[-1]
    if kind=='object' and expect in ('first','key'):
        if char=='}' and expect=='first':return finish_value(stack[:-1])
        return (stack,'key','','') if char=='"' else None
    if expect=='colon':return (stack[:-1]+((kind,'value'),),'normal','','') if char==':' else None
    if expect=='comma':
        if char==('}' if kind=='object' else ']'):return finish_value(stack[:-1])
        if char==',':return (stack[:-1]+((kind,'key' if kind=='object' else 'value'),),'normal','','')
        return None
    if kind=='array' and expect=='first' and char==']':return finish_value(stack[:-1])
    if char in '{[':
        if len(stack)>=32:return None
        return (stack+(('object' if char=='{' else 'array','first'),),'normal','','')
    if char=='"':return (stack,'string','','')
    if char in '-0123456789':return (stack,'number','sign' if char=='-' else 'zero' if char=='0' else 'int','')
    if char in 'tfn':return (stack,'literal','',{'t':'rue','f':'alse','n':'ull'}[char])
    return None


def consume(state, text):
    for char in text:
        state=advance(state,char)
        if state is None:return None
    return state


def canonical_piece(current):
    stack,mode,escape,literal=current
    if mode=='root':return '{}'
    if mode in ('key','string'):return '0'*int(escape[1:]) if escape.startswith('u') else 'n' if escape=='slash' else '"'
    if mode=='literal':return literal
    if mode=='number':return '0' if escape in ('sign','dot','exp','exp_sign') else ('}' if stack[-1][0]=='object' else ']')
    kind,expect=stack[-1]
    return '}' if kind=='object' and expect in ('first','comma') else ']' if kind=='array' and expect in ('first','comma') else '"' if expect=='key' else ':' if expect=='colon' else 'null'


class JSONSyntaxGrammar:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        ids=sorted(set(tokenizer.get_vocab().values())-set(tokenizer.all_special_ids))
        pieces=tokenizer.batch_decode([[i] for i in ids],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        self.pieces={i:p for i,p in zip(ids,pieces) if p}
        self.safe=tuple(i for i,p in self.pieces.items() if all(ord(c)>=32 and c not in '"\\' for c in p))
        safe=set(self.safe);self.other={i:p for i,p in self.pieces.items() if i not in safe}
        self.cache=OrderedDict();self.closures={}
        # A deterministic minimum tokenization of a canonical closure. A normal
        # tokenizer.encode call need not minimize token count over its vocabulary.
        self.closure_trie={}
        alphabet=set('{}[]:"nulltruefalse0')
        for token,piece in self.pieces.items():
            if set(piece)<=alphabet:
                node=self.closure_trie
                for char in piece:node=node.setdefault(char,{})
                node.setdefault(None,[]).append(token)

    def closure_tokens(self,text):
        best={len(text):()}
        for index in range(len(text)-1,-1,-1):
            node=self.closure_trie
            for end in range(index,len(text)):
                if text[end] not in node:break
                node=node[text[end]]
                if None in node and end+1 in best:
                    candidate=(min(node[None]),)+best[end+1]
                    if index not in best or (len(candidate),candidate)<(len(best[index]),best[index]):best[index]=candidate
        if 0 not in best:raise ValueError('tokenizer cannot spell canonical JSON continuation')
        return best[0]

    def closure(self,state):
        """A verified token continuation, an upper bound rather than shortest path.

        Canonical suffixes agree after consuming their prefix, so the shortest
        tokenization retains an affordable suffix as the budget decreases.
        No continuation is appended to a sampled response by the host program.
        """
        if state in self.closures:return self.closures[state]
        current=state;chars=[];seen=set()
        while current[1]!='done':
            if current in seen:raise ValueError('JSON closure failed to make progress')
            seen.add(current)
            piece=canonical_piece(current)
            chars.append(piece);current=consume(current,piece)
            if current is None:raise ValueError('invalid canonical JSON continuation')
        tokens=self.closure_tokens(''.join(chars))
        current=state
        for token in tokens:
            if token not in self.pieces:raise ValueError('JSON continuation uses excluded token')
            current=consume(current,self.pieces[token])
            if current is None:raise ValueError('tokenizer does not preserve JSON continuation')
        if current[1]!='done':raise ValueError('tokenized JSON continuation is incomplete')
        self.closures[state]=tokens
        return self.closures[state]

    def state(self,prefix):
        # Parse the actual decoded prefix so split Unicode byte tokens are resolved.
        text=self.tokenizer.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        state=consume(START,text)
        if state is None:raise ValueError('completion violates JSON syntax')
        return state

    def ordinary(self,state):return state[1] in ('key','string') and not state[2]

    def consume(self,state,text):return consume(state,text)

    def allowed(self,prefix,token_budget=None):
        state=self.state(prefix)
        remaining=None if token_budget is None else token_budget-len(prefix)
        if remaining is not None and remaining<=0:raise ValueError('JSON token budget exhausted')
        key=(state,remaining)
        if key not in self.cache:
            if remaining is not None and len(self.closure(state))>remaining:
                raise ValueError('JSON prefix has no verified affordable closure')
            ordinary=self.ordinary(state)
            allowed=list(self.safe) if ordinary and (remaining is None or len(self.closure(state))<remaining) else []
            for token,piece in (self.other if ordinary else self.pieces).items():
                next_state=self.consume(state,piece)
                if next_state is not None and (remaining is None or len(self.closure(next_state))<remaining):allowed.append(token)
            if not allowed:raise ValueError('tokenizer cannot extend JSON syntax')
            self.cache[key]=tuple(sorted(allowed))
            if len(self.cache)>256:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]
