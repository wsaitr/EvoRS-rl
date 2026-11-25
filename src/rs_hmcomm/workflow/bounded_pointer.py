"""Bounded evidence pointer with matching completion reserves; no label access."""
from .pixel_grounded import PixelGroundedRuntime


class BoundedPointerRuntime(PixelGroundedRuntime):
    def __init__(self,*args,pointer_limit=6,**kwargs):
        if type(pointer_limit)!=int or not 1<=pointer_limit<=16:raise ValueError('invalid pointer limit')
        super().__init__(*args,**kwargs);self.pointer_limit=pointer_limit

    def _required_pointer_evidence(self,role):
        if role=='verification' and self.state.answer_candidates:
            return set(self.state.answer_candidates[-1].evidence_ids)
        return set()

    def _controller_role_choices(self,choices,context):
        return choices

    def _feasible_subset(self,role,prefix,optional,required=()):
        """Can this pointer prefix complete to a new, affordable dispatch?

        Exact search stops at the first feasible completion. Previously executed
        signatures are bounded by the dispatch cap. Native-byte affordability is
        monotone with added evidence, allowing hopeless branches to be pruned.
        Knowledge-equivalent sources are collapsed; permutations are not new
        evidence. This checks protocol feasibility, never answer labels/quality.
        """
        if set(required)-set(prefix)-set(optional):return False
        selected=tuple(dict.fromkeys((*prefix,*required)))
        remaining=[e for e in optional if e not in selected]
        memo=set()
        def visit(chosen,rest):
            if len(chosen)>self.pointer_limit:return False
            if chosen and self._possible_modes(role,chosen):return True
            if not rest:return False
            key=(self._dispatch_key(role,chosen,'_feasibility')[1],tuple(rest))
            if key in memo:return False
            memo.add(key)
            fits=self._possible_modes(role,chosen,include_dispatched=True) if chosen else []
            if not fits:
                # A future pair or a first nonempty packet can unlock fusion;
                # adding evidence cannot make an oversized raw packet smaller.
                total=tuple(dict.fromkeys((*chosen,*rest)))
                lookup={e.evidence_id:e for e in self.state.evidence}
                has_packet=any(e.artifact_refs for e in self._support_closure([lookup[e] for e in chosen]))
                if has_packet and not self._possible_modes(role,total,include_dispatched=True):return False
                if not has_packet and not any(e.artifact_refs for e in self._support_closure([lookup[e] for e in total])) and 'text' not in self.mechanisms.modes:return False
            for index,eid in enumerate(rest):
                augmented=(*chosen,eid)
                if role=='knowledge' and self._dispatch_key(role,augmented,'_feasibility')[1]==self._dispatch_key(role,chosen,'_feasibility')[1]:continue
                if visit(augmented,rest[index+1:]):return True
            return False
        return visit(selected,remaining)

    def _select_pointer(self,role,context,ids,required,missing):
        selected=[]
        for _ in range(min(len(ids),self.pointer_limit)+1):
            options=[{'add':eid} for eid in ids if len(selected)<self.pointer_limit and eid not in selected and self._feasible_subset(
                role,[*selected,eid],[other for other in ids if other not in selected and other!=eid],required)]
            if selected and required<=set(selected) and self._possible_modes(role,selected):options.append({'stop':True})
            if self.training and self.role_coverage_curriculum or self.mechanisms.fixed_schedule:
                # Same complete evidence set for communication/search paired ablations.
                additions=[o for o in options if 'add' in o]
                options=additions[:1] or [o for o in options if o.get('stop')]
            if not options:raise ValueError('evidence subset cannot satisfy candidate provenance')
            value=self._call('controller','Select the next addressed evidence node, or finish the subset. You may select any combination. Prefer missing-requirement support and avoid duplicate source views.',context={
                'recipient':role,'evidence':context['evidence'],'selected':selected,'required':sorted(required),
                'marginal_statistics':self._marginals(selected,ids),
                'missing_semantic_requirements':missing,'legal_action_examples':options})
            if value.get('stop'):break
            selected.append(value['add'])
        return selected

    def _pointer_call_bound(self,ids):
        return min(len(ids),self.pointer_limit)+3

    def _closing_call_bound(self,ids,reasoning_calls):
        return 2*min(len(ids),self.pointer_limit)+10+reasoning_calls

    def _controller_decide(self,context,legal):
        if not self.mechanisms.pointer:return super()._controller_decide(context,legal)
        ids=[e.evidence_id for e in self.state.evidence]
        remaining=self.max_model_calls-self._model_calls_used()
        roles=['abort']
        if remaining>=5 and self.cost.tool_calls<self.max_tool_calls:roles.append('visual')
        if ids and remaining>=5:roles+=[r for r in ('knowledge','reasoning') if r in self.mechanisms.expert_roles and self._feasible_subset(r,[],ids,self._required_pointer_evidence(r))]
        if 'verification' in self.mechanisms.expert_roles and ids and remaining>=5 and self.state.answer_candidates and self.state.answer_candidates[-1].candidate_id not in self.verdicts and self._feasible_subset('verification',[],ids,self.state.answer_candidates[-1].evidence_ids):roles.append('verification')
        finish=[{'role':'finish','candidate_id':c.candidate_id} for c in self.state.answer_candidates if self._candidate_terminal_allowed(c) and self._finish_allowed(c)]
        choices=[{'role':r} for r in roles]+finish
        choices=self._pointer_progress_choices(choices, ids, remaining, finish)
        if self.training and self.role_coverage_curriculum or self.mechanisms.fixed_schedule:
            desired='visual' if not ids else 'knowledge' if 'knowledge' in self.mechanisms.expert_roles and not any(e.kind=='workflow_guidance' and e.created_by=='knowledge' for e in self.state.evidence) else 'reasoning' if not self.state.answer_candidates else 'verification' if self.state.answer_candidates[-1].candidate_id not in self.verdicts else 'finish' if finish else 'abort'
            filtered=[c for c in choices if c['role']==desired]
            if filtered:choices=filtered
        missing=self.ledger.missing() if self.ledger else []
        filtered=self._controller_role_choices(choices,context)
        if not filtered or any(choice not in choices for choice in filtered):
            raise ValueError('scheduler priority cannot introduce an illegal action')
        choices=filtered
        action=self._call('controller','Choose the next expert or a valid final candidate. Missing semantic requirements should guide acquisition; do not repeat interpretations without gaining evidence.',context={**context,
            'graph':[n.to_dict() for n in self.state.spatial_nodes], 'objects':self.objects.to_dict(),
            'missing_semantic_requirements':missing,'legal_action_examples':choices})
        if action['role'] in {'abort','finish'}:return action
        if action['role']=='visual':return {'role':'visual','evidence_ids':[],'mode':'tokens'}
        role=action['role'];selected=[]
        required=self._required_pointer_evidence(role)
        selected=self._select_pointer(role,context,ids,required,missing)
        packets=[p for e in self._support_closure([e for e in self.state.evidence if e.evidence_id in selected]) for p in e.artifact_refs]
        modes=self._possible_modes(role,selected)
        options=[{'mode':m} for m in modes]
        if self.training and self.role_coverage_curriculum:
            preferred='temporal' if self._find_pair(selected) and 'temporal' in modes else 'compressed'
            course=[o for o in options if o['mode']==preferred]
            if course:options=course
        remaining_bytes=self.max_packet_bytes-self.cost.message_latent_bytes
        raw_bytes=sum(self.policy.packets[p].artifact.byte_size for p in set(packets))
        compressed_bytes=getattr(self.policy,'compressed_tokens',16)*getattr(getattr(self.policy,'base',None),'config',type('Config',(),{'hidden_size':3584})()).hidden_size*4
        options=[o for o in options if o['mode']=='text' or (raw_bytes if o['mode']=='tokens' else compressed_bytes)<=remaining_bytes]
        if not options:raise ValueError('no communication mode fits remaining budget')
        value=self._call('controller','Choose communication mode for this recipient under the remaining budget. compressed is question/recipient-conditioned visual tokens; temporal includes registered signed change features.',context={
            'recipient':role,'selected':selected,'raw_bytes':sum(self.policy.packets[p].artifact.byte_size for p in set(packets)),
            'remaining_bytes':self.max_packet_bytes-self.cost.message_latent_bytes,'legal_action_examples':options})
        return {'role':role,'evidence_ids':selected,'mode':value['mode']}

    def _pointer_progress_choices(self, choices, ids, remaining, finish):
        if self.mechanisms.progress_guard:
            # A public candidate must be checked before it can be replaced.
            # Once independently accepted with complete support, additional
            # dispatches cannot improve this same completed task certificate.
            pending=bool(self.state.answer_candidates and self.state.answer_candidates[-1].candidate_id not in self.verdicts)
            if finish:choices=finish
            elif pending:choices=[c for c in choices if c['role'] in {'abort','visual','verification'}]
            pointer_calls=self._pointer_call_bound(ids)
            choices=[c for c in choices if c["role"] not in {"knowledge","verification"} or remaining>=pointer_calls+(4 if c["role"]=="verification" else 1)]
            remaining_steps=self.max_steps-self.step
            # Reserve dispatch slots for producing, checking and submitting an
            # answer. Inspection cannot consume the last required closing slots.
            if not finish and remaining_steps<=3 and not pending:
                choices=[c for c in choices if c['role'] in {'abort','reasoning'}]
            if not finish and remaining_steps<=2 and pending:
                choices=[c for c in choices if c['role'] in {'abort','verification'}]
            groups=max(1,len({tuple(e.payload.get('frame_ids',())) for e in self.state.evidence if e.kind=='workflow_visual'}))
            variations=self.mechanisms.generations*(self.mechanisms.population_size-1)
            reasoning_calls=1+2*variations+(1+variations)*groups if self.mechanisms.evolution else 1
            # Bound both pointers to the declared limit; reserve an additional
            # call for blind verification and the final controller turn.
            closing_calls=self._closing_call_bound(ids,reasoning_calls)
            choices=[c for c in choices if c['role']!='reasoning' or remaining>=closing_calls]
            visual_calls=3+self._mandatory_visual_assessment_calls()+int(self.mechanisms.objects)
            reserve=(len(ids)+8 if pending else closing_calls+4)
            choices=[c for c in choices if c['role']!='visual' or remaining>=visual_calls+reserve]
        return choices
