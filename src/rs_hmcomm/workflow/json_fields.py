"""Ordered required root fields and container types, with free model values.

Declared objects and bounded typed arrays can constrain their contents. Plain
array/object types remain unrestricted; IDs and facts need caller validation.
"""
from collections import OrderedDict
from functools import lru_cache
import json,re
from .json_syntax import JSONSyntaxGrammar,advance,canonical_piece
from .numeric_expression import START as NUMERIC_START,step as numeric_step,closure as numeric_closure,FUNCTION_ARITY

VALUE_START=((('object','value'),),'normal','','')
VALUE_DONE=((('object','comma'),),'normal','','')


def field_schema_key(fields):
    return json.dumps(fields,sort_keys=True,separators=(',',':'))


class JSONFieldGrammar(JSONSyntaxGrammar):
    def __init__(self,tokenizer,fields,base=None):
        fields=json.loads(field_schema_key(fields))
        names=[];field_count=0;enum_literals=[]
        def build_kind(kind,depth):
            if depth>8:raise ValueError('bounded field schema required')
            if isinstance(kind,dict) and set(kind)=={'enum'}:
                values=kind['enum']
                if (not isinstance(values,list) or not 1<=len(values)<=128
                        or any(not isinstance(v,str) or not v or len(v)>256 for v in values)
                        or len(set(values))!=len(values)):
                    raise ValueError('bounded distinct string enum required')
                encoded=tuple(json.dumps(v) for v in values)
                enum_literals.extend(encoded)
                return (('string_enum',encoded),)
            if isinstance(kind,dict) and set(kind)=={'object'}:
                return build(kind['object'],depth+1)
            if isinstance(kind,dict) and set(kind) in ({'array','maximum'},{'array','maximum','minimum'}):
                maximum=kind['maximum']
                if type(maximum)!=int or not 0<=maximum<=12:
                    raise ValueError('bounded array maximum required')
                minimum=kind.get('minimum',0)
                if type(minimum)!=int or not 0<=minimum<=maximum:
                    raise ValueError('bounded array minimum required')
                return (('typed_array',(build_kind(kind['array'],depth+1),maximum,minimum)),)
            if isinstance(kind,str) and kind in ('string','array','object'):
                return (('value',kind),)
            if kind in ('number','integer','nullable_integer'):return (('scalar_number',kind),)
            if isinstance(kind,dict) and set(kind)=={'tuple'}:
                rows=kind['tuple']
                if not isinstance(rows,list) or not 1<=len(rows)<=12:raise ValueError('bounded tuple schema required')
                segments=[('literal','[')]
                for index,item in enumerate(rows):
                    if index:segments.append(('literal',','))
                    segments.extend(build_kind(item,depth+1))
                segments.append(('literal',']'))
                return tuple(segments)
            if kind=='numeric_expression':return (('numeric_expression',256),)
            raise ValueError('unsupported field schema')
        def build(rows,depth=0):
            nonlocal field_count
            if depth>8 or not isinstance(rows,list) or not rows or len(rows)>12 or any(not isinstance(row,list) or len(row)!=2 for row in rows):
                raise ValueError('bounded field schema required')
            keys=[row[0] for row in rows]
            if any(not isinstance(name,str) or re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,79}',name) is None for name in keys):
                raise ValueError('unsupported field schema')
            if len(set(keys))!=len(keys):raise ValueError('duplicate field schema')
            field_count+=len(rows)
            if field_count>64:raise ValueError('bounded field schema required')
            names.extend(keys);segments=[('literal','{')]
            for index,(name,kind) in enumerate(rows):
                if index:segments.append(('literal',','))
                segments.append(('literal',json.dumps(name)+':'))
                segments.extend(build_kind(kind,depth))
            segments.append(('literal','}'))
            return tuple(segments)
        segments=build(fields)
        self.schema_key=field_schema_key(fields)
        self.fields=fields;self.base=base or JSONSyntaxGrammar(tokenizer);self.tokenizer=tokenizer
        self.pieces=self.base.pieces;self.safe=self.base.safe;self.other=self.base.other
        self.segments=segments
        self.start=(0,0,None);self.cache=OrderedDict();self.closures={};self.closure_trie={}
        alphabet=set('{}[]:,"nulltruefalse0123456789eE.+-*/%() '+''.join(names)+''.join(FUNCTION_ARITY)+''.join(enum_literals))
        for token,piece in self.pieces.items():
            if set(piece)<=alphabet:
                node=self.closure_trie
                for char in piece:node=node.setdefault(char,{})
                node.setdefault(None,[]).append(token)

    @lru_cache(maxsize=65536)
    def step(self,state,char):
        return self._step_segments(self.segments,state,char)

    @lru_cache(maxsize=65536)
    def _step_segments(self,segments,state,char):
        stage,offset,value=state
        if stage==len(segments):return state if char in ' \r\n\t' else None
        kind,piece=segments[stage]
        if kind=='string_enum':
            prefix=value or ''
            if not prefix and char in ' \r\n\t':return state
            candidate=prefix+char
            if not any(item.startswith(candidate) for item in piece):return None
            return (stage+1,0,None) if candidate in piece else (stage,0,candidate)
        if kind=='scalar_number':
            text=value or ''
            if char in ' \r\n\t,]}':
                if not text:return state if char in ' \r\n\t' else None
                try:parsed=json.loads(text)
                except ValueError:return None
                valid=(parsed is None or type(parsed) is int) if piece=='nullable_integer' else type(parsed) in ({int} if piece=='integer' else {int,float})
                if not valid:return None
                return self._step_segments(segments,(stage+1,0,None),char)
            if piece=='nullable_integer' and (text.startswith('n') or char=='n'):
                candidate=text+char
                return (stage,0,candidate) if 'null'.startswith(candidate) else None
            if char not in ('-0123456789' if piece in ('integer','nullable_integer') else '-0123456789.eE+'):return None
            candidate=text+char
            probe=VALUE_START
            for c in candidate:
                probe=advance(probe,c)
                if probe is None:return None
            if probe[1]!='number':return None
            return (stage,0,candidate)
        if kind=='numeric_expression':
            next_value=numeric_step(value or NUMERIC_START,char)
            if next_value is None:return None
            return (stage+1,0,None) if next_value[0]=='done' else (stage,0,next_value)
        if kind=='typed_array':
            items,maximum,minimum=piece
            phase,count,inner=value or ('start',0,None)
            if phase!='item' and char in ' \r\n\t':return state
            if phase=='start':
                return (stage,0,('first',0,None)) if char=='[' else None
            if phase in ('first','after') and char==']':
                return (stage+1,0,None) if count>=minimum else None
            if phase=='after':
                return (stage,0,('next',count,None)) if char==',' and count<maximum else None
            if phase in ('first','next'):
                if count>=maximum:return None
                inner=(0,0,None)
            # A scalar number is completed by its delimiter. The delimiter
            # belongs to this array and must be consumed after counting the
            # completed item, rather than discarded by the inner grammar.
            if (phase=='item' and inner[0]==len(items)-1
                    and items[inner[0]][0]=='scalar_number' and char in ',]'):
                completed=self._step_segments(items,inner,' ')
                if completed is None or completed[0]!=len(items):return None
                return self._step_segments(segments,(stage,0,('after',count+1,None)),char)
            inner=self._step_segments(items,inner,char)
            if inner is None:return None
            return (stage,0,('after',count+1,None)) if inner[0]==len(items) else (stage,0,('item',count,inner))
        if kind=='literal':
            # Whitespace outside literal keys; whitespace within a key changes it.
            outside=offset==0 or piece[offset] in '{}:,'
            if outside and char in ' \r\n\t':return state
            if char!=piece[offset]:return None
            return (stage+1,0,None) if offset+1==len(piece) else (stage,offset+1,None)
        value=value or VALUE_START
        if value==VALUE_START and char not in ' \r\n\t' and char!={'string':'"','array':'[','object':'{'}[piece]:return None
        value=advance(value,char)
        if value is None:return None
        return (stage+1,0,None) if value==VALUE_DONE else (stage,0,value)

    def consume(self,state,text):
        for char in text:
            state=self.step(state,char)
            if state is None:return None
        return state

    def state(self,prefix):
        state=self.consume(self.start,self.tokenizer.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False))
        if state is None:raise ValueError('completion violates required field grammar')
        return state

    def ordinary(self,state):
        return self._ordinary_segments(self.segments,state)

    def _ordinary_segments(self,segments,state):
        stage,_,value=state
        if stage==len(segments):return False
        kind,piece=segments[stage]
        if kind in ('numeric_expression','scalar_number','string_enum'):return False
        if kind=='typed_array':
            return value is not None and value[0]=='item' and self._ordinary_segments(piece[0],value[2])
        return value is not None and value[1] in ('key','string') and not value[2]

    def complete(self,state):return state[0]==len(self.segments)

    def closure(self,state):
        if state in self.closures:return self.closures[state]
        tokens=self.closure_tokens(self._closure_text(self.segments,state));current=state
        for token in tokens:
            current=self.consume(current,self.pieces[token])
            if current is None:raise ValueError('tokenizer does not preserve field continuation')
        if not self.complete(current):raise ValueError('tokenized field continuation incomplete')
        self.closures[state]=tokens
        return tokens

    def _closure_text(self,segments,state):
        current=state;chars=[];seen=set()
        while current[0]!=len(segments):
            if current in seen:raise ValueError('field closure failed to make progress')
            seen.add(current);stage,offset,value=current;kind,piece=segments[stage]
            if kind=='typed_array':
                phase,count,inner=value or ('start',0,None)
                if phase=='start':tail='['
                elif phase in ('first','after') and count>=piece[2]:tail=']'
                elif phase=='after':tail=','
                else:tail=self._closure_text(piece[0],inner or (0,0,None))
            elif kind=='numeric_expression':tail=numeric_closure(value or NUMERIC_START)
            elif kind=='string_enum':
                prefix=value or ''
                choices=[item for item in piece if item.startswith(prefix)]
                if not choices:raise ValueError('invalid string enum prefix')
                tail=min(choices,key=lambda item:(len(item),item))[len(prefix):]
            elif kind=='scalar_number':
                text=value or ''
                try:
                    parsed=json.loads(text)
                    finished=(parsed is None or type(parsed) is int) if piece=='nullable_integer' else type(parsed) in ({int} if piece=='integer' else {int,float})
                except ValueError:finished=False
                tail=' ' if finished else ('null'[len(text):]+' ' if piece=='nullable_integer' and (not text or text.startswith('n')) else '0 ')
            else:
                tail=piece[offset:] if kind=='literal' else {'string':'""','array':'[]','object':'{}'}[piece] if value is None or value==VALUE_START else canonical_piece(value)
            chars.append(tail)
            for char in tail:
                current=self._step_segments(segments,current,char)
                if current is None:break
            if current is None:raise ValueError('invalid required field continuation')
        return ''.join(chars)
