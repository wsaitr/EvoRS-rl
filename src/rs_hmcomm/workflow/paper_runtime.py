"""Complete addressed-evidence workflow; every learned choice is in the rollout."""
from dataclasses import dataclass,asdict
import json
import math
from datetime import datetime

from rs_hmcomm.contracts import Evidence,AnswerCandidate,CostVector
from rs_hmcomm.spatial import SpatialNode,SpatialLevel
from rs_hmcomm.evolution import IntraAgentEvolution
from rs_hmcomm.evolution.protocol import Candidate,EvolutionConfig,EvolutionBudget,VariationOutcome,FitnessEvaluation
from rs_hmcomm.communication import Inbox,ObservationBuilder
from .runtime import WorkflowRuntime,answer_valid,answer_instruction,schema,refs,rectangle_union_area
from .semantic import SemanticLedger,parse_requirements,ObjectRegistry,bbox_iou
from .learned import state_vector


@dataclass(frozen=True)
class PaperMechanisms:
    semantic: bool=True
    compression: bool=True
    temporal: bool=True
    evolution: bool=True
    pointer: bool=True
    objects: bool=True
    modes: tuple[str,...]=('text','tokens','compressed','temporal')
    fixed_schedule: bool=False
    expert_roles: tuple[str,...]=('knowledge','reasoning','verification')
    search_type: str='revision'
    payload_intervention: str='none'
    population_size: int=2
    generations: int=1
    initial_global_observation: bool=False
    structured_requirement_actions: bool=False
    progress_guard: bool=False
    observed_requirement_planning: bool=False
    public_choice_planning: bool=False
    planner_visual_context: str='none'

    def __post_init__(self):
        if self.planner_visual_context not in {'none','tokens','compressed','adaptive'}:
            raise ValueError('unknown planner visual context')
        if self.planner_visual_context!='none' and not self.observed_requirement_planning:
            raise ValueError('planner visual context requires observed requirement planning')
        if self.planner_visual_context in {'tokens','compressed'} and self.planner_visual_context not in self.modes:
            raise ValueError('planner visual context exceeds declared communication modes')
        if self.public_choice_planning and not self.observed_requirement_planning:
            raise ValueError("public choice planning requires observed requirement planning")
        if self.observed_requirement_planning and not (self.semantic and self.initial_global_observation):
            raise ValueError('observed requirement planning needs semantic requirements and an initial overview')
        if not 2<=self.population_size<=4 or not 1<=self.generations<=4:raise ValueError('private search bounds exceeded')
        if not self.modes or set(self.modes)-{'text','tokens','compressed','temporal'}:raise ValueError('unknown communication mode')
        if set(self.expert_roles)-{'knowledge','reasoning','verification'}:raise ValueError('unknown expert role')
        if self.search_type not in {'revision','best_of_n'} or self.payload_intervention not in {'none','zero','permute'}:raise ValueError('unknown experiment intervention')


class PaperWorkflowRuntime(WorkflowRuntime):
    def _mandatory_visual_assessment_calls(self):
        return len(self.ledger.requirements) if self.ledger else 0

    def __init__(self,policy,*,mechanisms=None,**kwargs):
        super().__init__(policy,**kwargs)
        self.mechanisms=mechanisms or PaperMechanisms()

    def _initialize_task(self):
        self.ledger=None;self.objects=ObjectRegistry();self.private_search=[];self.acquisition_request=None;self.last_modes={}
        self.planning_packets=();self.planning_disclosure_mode='none'
        self.events.append({'kind':'paper_mechanisms','configuration':asdict(self.mechanisms)})
        if not self.mechanisms.semantic:
            if self.mechanisms.initial_global_observation:self._observe_initial_globals()
            return
        overview=self._observe_initial_globals(assess=False) if self.mechanisms.observed_requirement_planning else None
        if self.mechanisms.planner_visual_context!='none':self._disclose_planning_view()
        if self.mechanisms.structured_requirement_actions:value=self._structured_requirements()
        else:value=self._call('controller','Decompose ONLY the question into 1 to 4 evidence requirements. Return JSON {"requirements":[{"query":"what must be established","kind":"object|count|relation|change|scene","scope":"local|global","frame_ids":["actual frame ID"]}]}. Do not answer. Global scope is necessary for exhaustive counts or absence claims; local scope is sufficient for a grounded local relation or presence. Use both frames for change. No additional keys.'+self._planning_view_instruction(),
                             self.planning_packets,context=self._requirement_planning_context())
        self.ledger=SemanticLedger(parse_requirements(value,self.task,limit=4))
        self.events.append({'kind':'requirements','requirements':[r.to_dict() for r in self.ledger.requirements]})
        if overview is not None:
            self._assess_visual(overview)
        elif self.mechanisms.initial_global_observation:self._observe_initial_globals()

    def _disclose_planning_view(self):
        """One authorized disclosure, reused by this recipient's planning calls.

        The adaptive choice is itself a paid controller completion. Choosing
        none preserves the fallible-text context; disclosure never earns support.
        The existing router charges latent and metadata bytes and enforces the
        actual wire budget. It may reject a mode whose metadata does not fit.
        """
        available=['none',*(m for m in self._possible_modes('controller',['overview'],include_dispatched=True)
                            if m in {'tokens','compressed'})]
        mode=self.mechanisms.planner_visual_context
        if mode=='adaptive':
            action=self._call('controller','Choose whether requirement planning needs the cached visual overview. '
                'none keeps the fallible textual interpretation; tokens transfers native visual features; '
                'compressed transfers a compact shared visual representation. This does not establish any fact. '
                'Choose one legal action considering uncertainty and the remaining communication budget.',
                context={'source_evidence_ids':['overview'],
                         'remaining_wire_bytes':self.max_packet_bytes-self.cost.message_latent_bytes-self.cost.message_structure_bytes,
                         'legal_action_examples':[{'mode':m} for m in available]})
            schema(action,{'mode'});mode=action['mode']
        if mode not in available:raise ValueError('planner visual disclosure is unavailable')
        if mode!='none':
            _,self.planning_packets,_=self._route('controller',['overview'],mode)
        self.planning_disclosure_mode=mode
        self.events.append({'kind':'requirement_planning_disclosure','mode':mode,
                            'packet_ids':list(self.planning_packets),'source_evidence_ids':['overview'],
                            'support_granted':False})

    def _planning_view_instruction(self):
        if not self.planning_packets:return ''
        return (' Inspect the supplied cached visual representation when choosing what to establish. '
                'Identify observable distinctions relevant to the question; do not assume the identities '
                'or properties proposed by the textual summary are true. A disclosed view is not a verified fact.')

    def _requirement_planning_context(self):
        context={'frames':[f.descriptor() for f in self.task.frames]}
        if self.mechanisms.observed_requirement_planning:
            views=[e for e in self.state.evidence if e.evidence_id=='overview'
                   and e.kind=='workflow_visual' and e.created_by=='visual' and e.artifact_refs]
            if len(views)!=1:raise ValueError('observed planning needs one actual committed overview')
            view=views[0]
            payload=view.to_dict()['payload']
            context['available_observation_summaries']=[{
                'evidence_id':view.evidence_id,'summary':view.summary,
                'frame_ids':payload['frame_ids'],
                'observed_bboxes':payload['observed_bboxes'],
                'is_model_observation':True,'factual_truth_confirmed':False}]
        if self.mechanisms.public_choice_planning and self.task.answer_format=='choice':
            from .runtime import question_choices
            alternatives=list(question_choices(self.task.question))
            labels=[option[0] for option in alternatives]
            if len(alternatives)<2 or len(set(labels))!=len(labels):
                raise ValueError('public choice planning needs distinct explicit alternatives')
            context['public_comparison_target']={
                'alternatives':alternatives,'source':'public_question',
                'all_alternatives_in_scope':True,'selected_answer':None,
                'observation_identity_claims_verified':False,
                'instruction':'Seek observations that distinguish these alternatives. '
                    'An identity proposed in a summary is a competing hypothesis, '
                    'not a selected answer. Do not restrict inspection to confirming it.'}
        if self.planning_disclosure_mode!='none':
            context['planner_visual_disclosure']={'mode':self.planning_disclosure_mode,
                'packet_ids':list(self.planning_packets),'source_evidence_ids':['overview'],
                'interpretation_verified':False}
        return context

    def _structured_requirements(self):
        """Keep requirement language free, but choose typed metadata by grammar.

        Every choice is a recorded trainable controller completion. Attributes
        can never name a nonexistent frame or ask for a one-date change pair.
        The small-frame pilot permits every nonempty frame subset; collections
        larger than four frames use singletons, pairs and the full collection.
        """
        from itertools import combinations
        frames=tuple(f.frame_id for f in self.task.frames)
        sizes=range(1,len(frames)+1) if len(frames)<=4 else (1,2,len(frames))
        subsets=[list(s) for n in sizes for s in combinations(frames,n)]
        options=[{'kind':kind,'scope':scope,'frame_ids':subset}
                 for kind in ('object','count','relation','change','scene') for scope in ('local','global')
                 for subset in subsets if kind!='change' or len(subset)>=2]
        result=[]
        for index in range(4):
            objective='Write ONE distinct visual fact that must be established to answer the question. Use one concise sentence. Do not answer the question or output JSON.'
            if self.mechanisms.progress_guard:
                if self.planning_packets:
                    objective += ' Write a neutral inspection request beginning with Determine whether, Determine how, or Determine the number. Textual summaries remain unverified hypotheses.'
                elif self.mechanisms.observed_requirement_planning:
                    objective += ' You have not received images directly. Write a neutral inspection request beginning with Determine whether, Determine how, or Determine the number. Observation summaries are model interpretations, not confirmed facts. Do not turn them into assertions or answers.'
                else:
                    objective += ' You have not received images. Write a neutral inspection request beginning with Determine whether, Determine how, or Determine the number. Never assert the presence, absence, count or appearance of objects; those are still unknown.'
            if self.mechanisms.observed_requirement_planning:
                objective += ' Treat available observation summaries as fallible context for choosing task-relevant inspection requirements. They do not establish correctness. Ask what distinguishes the original options; do not require hidden properties merely because the object category is associated with them.'
            if self.mechanisms.public_choice_planning and self.task.answer_format=='choice':
                objective += ' Use the complete public_comparison_target. Describe a discriminating inspection across its alternatives, not a test that assumes the identity suggested by an observation. No alternative is selected or verified.'
            objective+=self._planning_view_instruction()
            query=self._call('controller',objective,self.planning_packets,
                context={'existing_requirements':result,**self._requirement_planning_context()},output_format='text')
            attributes=self._call('controller','Choose the kind, scope and actual frame set for this requirement. Exhaustive whole-scene counts or absence claims need global scope. A temporal change needs at least two frames.',
                context={'query':query,'legal_action_examples':options})
            result.append({'query':query,**attributes})
            if index==3:break
            more=self._call('controller','Does a distinct requirement remain? Stop when the question is covered. Do not add paraphrases of existing requirements.',
                context={'requirements':result,'legal_action_examples':[{'more':False},{'more':True}]})
            if not more['more']:break
        return {'requirements':result}

    def _observe_initial_globals(self, *, assess=True):
        """Use the already encoded overview before spending the crop budget.

        This optional development control makes the multi-role initialization
        comparable to the global view already available to direct/single arms.
        Only explicitly registered identity maps can make a temporal pair here.
        Non-identity transforms still require the registered crop tool.
        """
        ids=tuple(self.global_packets[f.frame_id] for f in self.task.frames)
        summary=self._call('visual','Observe the attached whole-image overviews. Describe only question-relevant visible facts in at most two sentences. Compare dates where applicable; state missing or ambiguous evidence.',ids,output_format='text')
        paired=len(self.task.frames)==2 and len({f.timestamp for f in self.task.frames})==2 and all(
            f.reference_to_frame==(1.,0.,0.,0.,1.,0.) and f.alignment_confidence>=self.min_pair_confidence for f in self.task.frames)
        nodes=tuple(SpatialNode('overview-region-'+f.frame_id,SpatialLevel.REGION,'global_overview',
            (0.,0.,1.,1.),'scene',evidence_ids=('overview',),visual_artifact_refs=(p,),
            provenance={'frame_id':f.frame_id,'timestamp':f.timestamp,'source':f.source,'reference_bbox':[0.,0.,1.,1.],'paired':paired})
            for f,p in zip(self.task.frames,ids))
        supports=['frame:'+f.frame_id for f in self.task.frames]+(['pair:registered'] if paired else [])
        evidence=Evidence('overview','workflow_visual',summary,'visual',0,
            spatial_node_ids=tuple(n.node_id for n in nodes),artifact_refs=ids,
            payload={'supports':supports,'source_fingerprint':'overview:'+','.join(f.descriptor()['sha256'] for f in self.task.frames),
                     'frame_ids':[f.frame_id for f in self.task.frames],'reference_bbox':[0.,0.,1.,1.],
                     'observed_bboxes':{f.frame_id:[[0.,0.,1.,1.]] for f in self.task.frames},
                     'pair_id':'overview-registered-pair' if paired else None})
        self._commit('visual',spatial_nodes=nodes,evidence=(evidence,))
        if assess:self._assess_visual(evidence)
        return evidence

    def _call(self,role,objective,packet_ids=(),context=None,**kwargs):
        self.policy.payload_intervention=self.mechanisms.payload_intervention if role in {'knowledge','reasoning'} else 'none'
        if role=='visual' and self.mechanisms.fixed_schedule and not self.state.evidence and (context or {}).get('legal_action_examples'):
            context=dict(context);options=context['legal_action_examples']
            full=[a for a in options if a.get('bbox')==[0.,0.,1.,1.] and a.get('op')==('paired_crop' if len(self.task.frames)>1 and all(f.reference_to_frame for f in self.task.frames) else 'crop')]
            if full:context['legal_action_examples']=full[:1]
        potential=self._coverage() if hasattr(self,'state') else 0.0
        missing=len(self.ledger.missing()) if getattr(self,'ledger',None) is not None else len(getattr(self,'requirements',()))
        features=state_vector(coverage=potential,model_calls=self._model_calls_used(),tool_calls=self.cost.tool_calls,
            evidence_count=len(self.state.evidence),candidate_count=len(self.state.answer_candidates),missing_count=missing,
            role=role,invalid_actions=self.cost.invalid_actions,communication_bytes=self.cost.message_latent_bytes+self.cost.message_structure_bytes)
        before=self.cost.to_dict();length=len(self.policy.records)
        try:return super()._call(role,objective,packet_ids,context,**kwargs)
        finally:
            if len(self.policy.records)>length:
                record=self.policy.records[-1];record.state_features=features;record.cost_before=before;record.potential_before=potential
                if getattr(self.policy,'auxiliary',None) is not None:
                    import torch
                    with torch.no_grad():
                        evaluator = getattr(self.policy, 'record_value', None)
                        record.old_value=float(evaluator(record) if evaluator else self.policy.value(features))

    def _coverage(self,evidence=None):
        if getattr(self,'ledger',None) is None:
            # Observing before planning must not temporarily manufacture
            # semantic reward while the requirements are still undefined.
            if self.mechanisms.observed_requirement_planning:return 0.0
            return super()._coverage(evidence)
        selected=None if evidence is None else [e.evidence_id for e in self._support_closure(evidence)]
        return self.ledger.coverage(selected)

    def _finish_allowed(self,candidate):
        if self.ledger is None:return super()._finish_allowed(candidate)
        selected=[e.evidence_id for e in self._support_closure([e for e in self.state.evidence if e.evidence_id in candidate.evidence_ids])]
        return self.ledger.sufficient(selected)

    def _possible_modes(self,role,selected,*,include_dispatched=False):
        lookup={e.evidence_id:e for e in self.state.evidence}
        closure=self._support_closure([lookup[e] for e in selected])
        packets={p for e in closure for p in e.artifact_refs}
        remaining=self.max_packet_bytes-self.cost.message_latent_bytes-self.cost.message_structure_bytes
        raw=sum(self.policy.packets[p].artifact.byte_size for p in packets)
        config=getattr(getattr(self.policy,'base',None),'config',None)
        encoded=getattr(self.policy,'compressed_tokens',16)*getattr(config,'hidden_size',3584)*4
        result=[]
        for mode in self.mechanisms.modes:
            if mode=='compressed' and not self.mechanisms.compression:continue
            if mode=='compressed' and getattr(self.policy,'spatial_codec',False) and len(packets)>getattr(self.policy,'compressed_tokens',16):continue
            if mode=='temporal' and (not self.mechanisms.temporal or self._find_pair(selected) is None):continue
            if mode!='text' and not packets:continue
            if (0 if mode=='text' else raw if mode=='tokens' else encoded)>remaining:continue
            if not include_dispatched and self._dispatch_key(role,selected,mode) in self.dispatched:continue
            result.append(mode)
        return result

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

    def _controller_decide(self,context,legal):
        if not self.mechanisms.pointer:return super()._controller_decide(context,legal)
        ids=[e.evidence_id for e in self.state.evidence]
        remaining=self.max_model_calls-self._model_calls_used()
        roles=['abort']
        if remaining>=5 and self.cost.tool_calls<self.max_tool_calls:roles.append('visual')
        if ids and remaining>=5:roles+=[r for r in ('knowledge','reasoning') if r in self.mechanisms.expert_roles and self._feasible_subset(r,[],ids)]
        if 'verification' in self.mechanisms.expert_roles and ids and remaining>=5 and self.state.answer_candidates and self.state.answer_candidates[-1].candidate_id not in self.verdicts and self._feasible_subset('verification',[],ids,self.state.answer_candidates[-1].evidence_ids):roles.append('verification')
        finish=[{'role':'finish','candidate_id':c.candidate_id} for c in self.state.answer_candidates if self.verdicts.get(c.candidate_id)=='accept' and self._finish_allowed(c)]
        choices=[{'role':r} for r in roles]+finish
        if self.mechanisms.progress_guard:
            # A public candidate must be checked before it can be replaced.
            # Once independently accepted with complete support, additional
            # dispatches cannot improve this same completed task certificate.
            pending=bool(self.state.answer_candidates and self.state.answer_candidates[-1].candidate_id not in self.verdicts)
            if finish:choices=finish
            elif pending:choices=[c for c in choices if c['role'] in {'abort','visual','verification'}]
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
            # Worst-case pointer selects every currently available record;
            # include independent verification and the final controller turn.
            closing_calls=2*len(ids)+9+reasoning_calls
            choices=[c for c in choices if c['role']!='reasoning' or remaining>=closing_calls]
            visual_calls=3+self._mandatory_visual_assessment_calls()+int(self.mechanisms.objects)
            reserve=(len(ids)+8 if pending else closing_calls+4)
            choices=[c for c in choices if c['role']!='visual' or remaining>=visual_calls+reserve]
        if self.training and self.role_coverage_curriculum or self.mechanisms.fixed_schedule:
            desired='visual' if not ids else 'knowledge' if 'knowledge' in self.mechanisms.expert_roles and not any(e.kind=='workflow_guidance' and e.created_by=='knowledge' for e in self.state.evidence) else 'reasoning' if not self.state.answer_candidates else 'verification' if self.state.answer_candidates[-1].candidate_id not in self.verdicts else 'finish' if finish else 'abort'
            filtered=[c for c in choices if c['role']==desired]
            if filtered:choices=filtered
        missing=self.ledger.missing() if self.ledger else []
        action=self._call('controller','Choose the next expert or a valid final candidate. Missing semantic requirements should guide acquisition; do not repeat interpretations without gaining evidence.',context={**context,
            'graph':[n.to_dict() for n in self.state.spatial_nodes], 'objects':self.objects.to_dict(),
            'missing_semantic_requirements':missing,'legal_action_examples':choices})
        if action['role'] in {'abort','finish'}:return action
        if action['role']=='visual':return {'role':'visual','evidence_ids':[],'mode':'tokens'}
        role=action['role'];selected=[]
        required=set(self.state.answer_candidates[-1].evidence_ids) if role=='verification' else set()
        for _ in range(len(ids)+1):
            options=[{'add':eid} for eid in ids if eid not in selected and self._feasible_subset(
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

    def _marginals(self,selected,ids):
        by_id={e.evidence_id:e for e in self.state.evidence}
        current=self._support_closure([by_id[e] for e in selected]);base=self._coverage(current)
        current_packets={p for e in current for p in e.artifact_refs};result={}
        for eid in ids:
            if eid in selected:continue
            augmented=self._support_closure([by_id[e] for e in (*selected,eid)])
            gain=self._coverage(augmented)-base;overlap=0.
            for new in augmented:
                if new in current or new.kind!='workflow_visual':continue
                for old in current:
                    if old.kind!='workflow_visual':continue
                    for fid,boxes in new.payload.get('observed_bboxes',{}).items():
                        overlap=max(overlap,max((bbox_iou(a,b) for a in boxes for b in old.payload.get('observed_bboxes',{}).get(fid,())),default=0.))
            new_packets={p for e in augmented for p in e.artifact_refs}-current_packets
            marginal_bytes=sum(self.policy.packets[p].artifact.byte_size for p in new_packets)
            price=getattr(self,'communication_dual',.005)
            result[eid]={'semantic_gain_estimate':gain,'same_source_roi_overlap':overlap,'additional_native_bytes':marginal_bytes,
                'utility':gain-.1*overlap-price*marginal_bytes/1048576}
        return result

    def _find_pair(self,selected):
        evidence=self._support_closure([e for e in self.state.evidence if e.evidence_id in selected])
        return next((e for e in evidence if e.payload.get('pair_id') and len(e.artifact_refs)==2),None)

    def _route(self,role,selected,mode):
        self.last_modes[role]=mode
        if role=='visual':return super()._route(role,selected,'tokens')
        evidence={e.evidence_id:e for e in self.state.evidence};selected=refs(selected,evidence)
        disclosed=self._support_closure([evidence[eid] for eid in selected])
        originals=tuple(dict.fromkeys(p for e in disclosed for p in e.artifact_refs))
        elapsed=None
        if mode=='temporal':
            pair=self._find_pair(selected)
            if pair is None:raise ValueError('temporal disclosure requires paired evidence')
            originals=pair.artifact_refs
            frames=[next(f for f in self.task.frames if f.frame_id==fid) for fid in pair.payload['frame_ids']]
            try:elapsed=(datetime.fromisoformat(frames[1].timestamp)-datetime.fromisoformat(frames[0].timestamp)).total_seconds()/86400
            except ValueError:pass
        ids=() if mode=='text' else originals if mode=='tokens' else self.policy.disclose(originals,recipient=role,question=self.task.question,mode=mode,elapsed_days=elapsed)
        count=sum(self.policy.packets[p].artifact.byte_size for p in ids)
        if self.cost.message_latent_bytes+count>self.max_packet_bytes:raise ValueError('communication budget exceeded')
        if mode in {'compressed','temporal'}:self._commit('controller',artifacts=tuple(self.policy.packets[p].artifact for p in ids))
        eids={e.evidence_id for e in disclosed};linked={n for e in disclosed for n in e.spatial_node_ids}
        nodes=[n.node_id for n in self.state.spatial_nodes if n.node_id in linked or eids.intersection(n.evidence_ids)] if self.mechanisms.objects else []
        envelope,grant=self.router.route(self.state,'controller',role,(self.state.task.task_id,*(e.evidence_id for e in disclosed),*nodes,*ids),
            ('text','structure','latent') if ids else ('text','structure'),self.step)
        inbox=Inbox();inbox.append(envelope);view=ObservationBuilder.build(self.state,role,inbox)
        actual=tuple(a.artifact_id for a in view.artifacts)
        if set(actual)!=set(ids):raise ValueError('packet authorization scope disagrees with disclosure')
        context={'evidence':[e.to_dict() for e in view.evidence],'requirements':[r.to_dict() for r in self.ledger.requirements] if self.ledger else [],
            'graph':[n.to_dict() for n in view.spatial_nodes]}
        registry=self.objects.to_dict();allowed={n.node_id for n in view.spatial_nodes}
        context['objects']={
            'objects':[o for o in registry['objects'] if o['object_id'] in allowed],
            'temporal_links':[e for e in registry['temporal_links'] if e['before'] in allowed and e['after'] in allowed],
            'spatial_relations':[e for e in registry['spatial_relations'] if e['subject'] in allowed and e['object'] in allowed]}
        metadata_bytes=len(json.dumps(context,ensure_ascii=False,separators=(',',':')).encode())
        total_bytes=count+metadata_bytes+grant.cost.message_structure_bytes
        used=self.cost.message_latent_bytes+self.cost.message_structure_bytes
        if used+total_bytes>self.max_packet_bytes:raise ValueError('communication budget exceeded')
        self.cost+=grant.cost+CostVector(message_structure_bytes=metadata_bytes)
        self.events.append({'kind':'route','role':role,'decision_index':len(self.policy.records)-1,'mode':mode,'bytes':count,
                            'metadata_bytes':metadata_bytes,'total_wire_bytes':total_bytes,
                            'packet_ids':list(actual),'source_packet_ids':list(originals),'message':envelope.to_dict()})
        return context,actual,selected

    def _visual(self,context,packets):
        super()._visual(context,packets)
        visual=[e for e in self.state.evidence if e.kind=='workflow_visual'];latest=visual[-1]
        self._assess_visual(latest)

    def _assess_visual(self,latest):
        if self.mechanisms.objects:
            try:self._extract_objects(latest)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                # Optional object metadata must not erase a valid raw observation.
                self.cost+=CostVector(invalid_actions=1,errors=1)
                self.events.append({'kind':'optional_component_failure','component':'objects','reason':str(exc),
                                    'decision_index':len(self.policy.records)-1,'evidence_id':latest.evidence_id})
        if self.ledger is None:return
        visual=[e for e in self.state.evidence if e.kind=='workflow_visual']
        assessment_context,ids,_=self._route('verification',[e.evidence_id for e in visual],'tokens')
        observed={f.frame_id:[] for f in self.task.frames}
        for e in visual:
            for fid,boxes in e.payload.get('observed_bboxes',{}).items():observed[fid].extend(boxes)
        assessment_id=f'assessment-{len(self.ledger.assessments)}'
        for requirement in self.ledger.requirements:
            choices=[{'verdict':v} for v in ('supported','uncertain','contradicted')]
            objective='Assess whether the attached actual views and observations establish this requirement. A description alone is not independent visual confirmation. supported requires direct evidence; uncertain covers missing views; contradicted covers conflicting observations.'
            if self.mechanisms.progress_guard:
                objective += ' Assess whether the question can be resolved from this evidence, not whether its answer is affirmative. A clear No, absence, or no-change result is supported when the required scene/dates were inspected. contradicted requires mutually inconsistent observations, not a negative answer.'
            action=self._call('verification',objective,ids,
                {'requirement':requirement.to_dict(),'observations':assessment_context['evidence'],'legal_action_examples':choices})
            global_valid=all(rectangle_union_area(observed[f])>=1-1e-8 for f in requirement.frame_ids)
            pair_valid=any(e.payload.get('pair_id') and set(requirement.frame_ids)<=set(e.payload['frame_ids']) for e in visual)
            sources=','.join(sorted(f.descriptor()['sha256'] for f in self.task.frames if f.frame_id in requirement.frame_ids))
            observed_ids=[e.evidence_id for e in visual] if self.mechanisms.progress_guard else ()
            self.ledger.record(requirement.requirement_id,assessment_id,sources,action['verdict'],global_valid=global_valid,pair_valid=pair_valid,observed_evidence_ids=observed_ids)
        ratings=self.ledger.assessments[assessment_id]
        self._commit('verification',evidence=(Evidence(assessment_id,'workflow_guidance',json.dumps(ratings),'verification',self.step,
            payload={'source_evidence_ids':[e.evidence_id for e in visual],'semantic_assessment':ratings}),))
        self.events.append({'kind':'semantic_assessment','evidence_id':assessment_id,'ratings':ratings,'coverage':self.ledger.coverage()})

    def _extract_objects(self,evidence):
        value=self._call('visual','Return JSON {"objects":[{"category":"name","crop_index":0,"bbox":[0.1,0.1,0.3,0.4]}]}. Extract at most FOUR question-relevant visible objects. bbox is NORMALIZED within its attached crop, never pixel coordinates. Use an empty list if no reliable object can be localized.',evidence.artifact_refs,
                         {'crops':[self.policy.packets[p].label for p in evidence.artifact_refs]})
        schema(value,{'objects'})
        if not isinstance(value['objects'],list) or len(value['objects'])>4:raise ValueError('invalid object list')
        import copy
        previous=copy.deepcopy(self.objects)
        nodes=[];groups={}
        # Validate all rows before mutating the deduplication ledger.
        for item in value['objects']:
            schema(item,{'category','crop_index','bbox'})
            from .evidence import _normalized_bbox
            if not isinstance(item['category'],str) or not item['category'].strip():raise ValueError('missing category')
            _normalized_bbox(item['bbox'])
            if isinstance(item['crop_index'],bool) or not isinstance(item['crop_index'],int) or not 0<=item['crop_index']<len(evidence.artifact_refs):raise ValueError('unknown crop index')
        for item in value['objects']:
            schema(item,{'category','crop_index','bbox'});index=item['crop_index']
            if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(evidence.artifact_refs):raise ValueError('unknown crop index')
            from .evidence import _normalized_bbox
            local=_normalized_bbox(item['bbox']);region=next(n for n in self.state.spatial_nodes if n.node_id==evidence.spatial_node_ids[index])
            x1,y1,x2,y2=region.bbox
            native=(x1+local[0]*(x2-x1),y1+local[1]*(y2-y1),x1+local[2]*(x2-x1),y1+local[3]*(y2-y1))
            frame=next(f for f in self.task.frames if f.frame_id==region.provenance['frame_id']);reference=None
            if frame.reference_to_frame is not None:
                a,b,c,d,e,f=frame.reference_to_frame;det=a*e-b*d
                corners=[((e*(x-c)-b*(y-f))/det,(-d*(x-c)+a*(y-f))/det) for x,y in ((native[0],native[1]),(native[0],native[3]),(native[2],native[1]),(native[2],native[3]))]
                mapped=(min(x for x,y in corners),min(y for x,y in corners),max(x for x,y in corners),max(y for x,y in corners))
                if all(0<=v<=1 for v in mapped):reference=mapped
            oid=self.objects.add(item['category'],native,region.provenance['frame_id'],evidence.evidence_id,reference_bbox=reference)
            if oid not in {n.node_id for n in self.state.spatial_nodes} and oid not in {n.node_id for n in nodes}:
                key=(region.node_id,item['category'].casefold())
                if key not in groups:
                    gid=f'group-{evidence.evidence_id}-{len(groups)}';groups[key]=gid
                    nodes.append(SpatialNode(gid,SpatialLevel.GROUP,item['category'],region.bbox,region.node_id,evidence_ids=(evidence.evidence_id,),provenance=dict(region.provenance)))
                nodes.append(SpatialNode(oid,SpatialLevel.OBJECT,item['category'],native,groups[key],evidence_ids=(evidence.evidence_id,),provenance=dict(region.provenance)))
        if nodes:
            try:self._commit('visual',spatial_nodes=tuple(nodes))
            except (ValueError,TypeError):self.objects=previous;raise
        self.events.append({'kind':'object_registry','registry':self.objects.to_dict()})

    def _private_fitness_route(self, selected, mode):
        return self._route('verification', selected, mode)

    def _private_variation_context(self, parent, context, stage):
        # Optional private feedback belongs to the receiving expert's context.
        # The historical runtime retains its exact input and cost contract.
        return context

    def _expert(self,role,context,packets,selected):
        if role!='reasoning' or not self.mechanisms.evolution:return super()._expert(role,context,packets,selected)
        # Use the existing private engine; its callbacks are real sampled role turns.
        instruction=answer_instruction(self.task.answer_format)
        seed=answer_valid(self._call('reasoning','Form a private candidate from the selected evidence. '+instruction,packets,context,output_format='text'),self.task.answer_format)
        runtime=self
        private_context=context
        search_mode=getattr(self,'_active_private_fitness_mode',self.last_modes.get('reasoning','tokens'))
        class Variation:
            name='evidence_conditioned_revision'
            def vary(self,parent,*,context):
                cost_before=runtime.cost.to_dict()
                parent_value=parent.to_private_dict()['payload'] if runtime.mechanisms.search_type=='revision' else {'note':'independent candidate; parent answer hidden'}
                choice=runtime._call('reasoning','Choose how to revise the private candidate. request_view proposes a missing observation; revise re-examines selected evidence.',packets,runtime._private_variation_context(parent,{
                    'parent':parent_value,'missing':runtime.ledger.missing(selected) if runtime.ledger else [],
                    'legal_action_examples':[{'operation':'revise'},{'operation':'request_view'}]},'variation_action'))
                text=runtime._call('reasoning','Re-examine the evidence and produce a candidate, addressing the proposed revision. '+instruction,packets,
                    runtime._private_variation_context(parent,{'parent':parent_value,'operation':choice['operation'],'evidence':private_context['evidence']},'candidate_answer'),output_format='text')
                cost=CostVector(**{k:max(0,v-cost_before[k]) for k,v in runtime.cost.to_dict().items()})
                return VariationOutcome({'answer':answer_valid(text,runtime.task.answer_format),'acquisition':choice['operation']},cost=cost)
        class Fitness:
            name='source_grouped_support_coverage_contradiction_cost'
            def evaluate(self,candidate):
                cost_before=runtime.cost.to_dict()
                # Group visual evidence by original image/date identity. Repeated
                # crops cannot become extra independent votes in candidate fitness.
                closed=runtime._support_closure([e for e in runtime.state.evidence if e.evidence_id in selected])
                grouped={}
                for ev in closed:
                    if ev.kind!='workflow_visual':continue
                    source=','.join(sorted(f.descriptor()['sha256'] for f in runtime.task.frames if f.frame_id in ev.payload.get('frame_ids',())))
                    grouped.setdefault(source,[]).append(ev.evidence_id)
                verdicts={}
                for group,group_ids in grouped.items():
                    fitness_context,fitness_packets,_=runtime._private_fitness_route(group_ids,search_mode)
                    judgement=runtime._call('verification','Evaluate the support contributed by this ORIGINAL SOURCE GROUP to the private candidate. supported means this source supports the claim; uncertain means insufficient evidence; contradicted means conflict. Repeated views are one group, not votes.',fitness_packets,{
                        'candidate':candidate.to_private_dict()['payload'],'evidence':fitness_context['evidence'],'source_group':group,
                        'legal_action_examples':[{'verdict':v} for v in ('supported','uncertain','contradicted')]})
                    verdicts[group]=judgement['verdict']
                support=sum(SemanticLedger.scores[v] for v in verdicts.values())/max(1,len(verdicts))
                coverage=runtime._coverage([e for e in runtime.state.evidence if e.evidence_id in selected])
                contradiction=float(any(v=='contradicted' for v in verdicts.values()));cost=.01*candidate.generation
                measured=CostVector(**{k:max(0,v-cost_before[k]) for k,v in runtime.cost.to_dict().items()})
                runtime.events.append({'kind':'private_source_fitness','candidate_id':candidate.candidate_id,'source_verdicts':verdicts,'cost':measured.to_dict()})
                return FitnessEvaluation(support+coverage-contradiction-cost,{'support':support,'coverage':coverage,'contradiction':contradiction,'search_cost':cost,'source_groups':len(grouped)},cost=measured)
        variations=self.mechanisms.generations*(self.mechanisms.population_size-1)
        evaluations=1+variations
        needed=evaluations+2*variations+1
        if self.max_model_calls-self._model_calls_used()<needed:raise ValueError('insufficient budget for private search')
        engine=IntraAgentEvolution(Variation(),Fitness(),EvolutionConfig(enabled=True,population_size=self.mechanisms.population_size,max_generations=self.mechanisms.generations,
            budget=EvolutionBudget(max_candidate_evaluations=evaluations,max_variations=variations)))
        result=engine.run(Candidate(f'private-{self.step}',{'answer':seed,'acquisition':'none'}))
        trace=result.private_trace.to_dict()
        self.private_search.append(trace)
        winner=result.selected_candidate
        self.events.append({'kind':'private_evolution','trace':trace,'selected':winner.to_selected_dict()})
        if winner.payload.get('acquisition')=='request_view':
            self.acquisition_request={'candidate_id':winner.candidate_id,'missing':self.ledger.missing(selected) if self.ledger else []}
        self._commit('reasoning',answer_candidates=(AnswerCandidate(f'candidate-{self.step}',winner.payload['answer'],'reasoning',
            evidence_ids=selected,created_step=self.step),))

    def _after_role(self,role):
        super()._after_role(role)
        if role=='visual':self.acquisition_request=None
        elif self.acquisition_request is not None:
            self.feedback='Private winner requests additional evidence: '+json.dumps(self.acquisition_request)
