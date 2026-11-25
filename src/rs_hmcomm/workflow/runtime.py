"""Label-free, bounded workflow execution using the existing Workspace/Router.

All role decisions are model-generated and recorded. Ground truth is accepted
only by the external reward function, never by this runtime or its prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from rs_hmcomm.artifacts import ArtifactKind, VisualArtifact
from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.communication import ContextRouter, Inbox, ObservationBuilder
from rs_hmcomm.contracts import TaskContext, Evidence, AnswerCandidate, CostVector
from rs_hmcomm.spatial import SpatialNode, SpatialLevel
from rs_hmcomm.workspace import WorkspaceState, WorkspaceDelta, WorkspaceReducer
from rs_hmcomm.tools.base import ToolContext
from rs_hmcomm.tools.crop import CropTool, _normalized_bbox
from .evidence import WorkflowTask, crop_frame, paired_crops, image_digest, observation_encodable


def strict_json(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    def pairs(items):
        value = {}
        for key,item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    result = json.loads(text,object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(result,dict):
        raise ValueError("action must be a JSON object")
    return result


def schema(value, required, optional=()):
    if set(required)-set(value) or set(value)-set(required)-set(optional):
        raise ValueError(f"expected fields {sorted(required)}, optional {sorted(optional)}")


def refs(value, allowed):
    if not isinstance(value,list) or any(not isinstance(v,str) for v in value):
        raise ValueError("evidence_ids must be a list of IDs")
    if len(value) != len(set(value)) or set(value)-set(allowed):
        raise ValueError("unknown/duplicate evidence IDs")
    return tuple(value)


def answer_valid(answer, format):
    if not isinstance(answer,str) or not answer.strip() or re.search(r"#\s*\d",answer):
        raise ValueError("answer must be nonempty semantic text, not an internal region number")
    if format in {'yes_no','integer','choice'}:answer=answer.strip().rstrip('.').strip()
    if format=='choice':answer=answer.upper()
    if format == "yes_no" and answer.strip().lower() not in {"yes","no"}:
        raise ValueError("answer format requires Yes or No")
    if format == "integer" and not re.fullmatch(r"\d+",answer.strip()):
        raise ValueError("answer format requires a nonnegative integer")
    if format == 'choice' and not re.fullmatch(r'[A-Z]',answer.strip()):
        raise ValueError('answer format requires one uppercase option letter')
    if format == 'bbox':
        value=strict_json(answer);schema(value,{'bbox'});_normalized_bbox(value['bbox'])
    return answer.strip()


def answer_instruction(format):
    return {'yes_no':'Reply with exactly Yes or No.',
        'integer':'Reply with only a nonnegative integer.',
        'choice':'Reply with only the uppercase letter of the selected option.',
        'bbox':'Reply with JSON {"bbox":[x1,y1,x2,y2]} normalized in the ORIGINAL image.'}.get(
            format,'Reply with a concise semantic answer, without explanation or internal IDs.')


def question_choices(question):
    """Recover explicit MCQ options from the question, without reading labels."""
    return tuple(match.group(0).strip() for match in re.finditer(r'(?m)^\s*[A-Z][.)]\s*[^\n]+',question))


def rectangle_union_area(boxes):
    """Exact normalized observed area; overlapping crops never count twice."""
    boxes=[_normalized_bbox(b) for b in boxes]
    xs=sorted({x for b in boxes for x in (b[0],b[2])})
    area=0.0
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((b[1],b[3]) for b in boxes if b[0]<right and b[2]>left)
        end=0.0;height=0.0
        for bottom,top in intervals:
            height+=max(0.0,top-max(end,bottom));end=max(end,top)
        area+=(right-left)*height
    return min(1.0,area)


@dataclass
class WorkflowEpisode:
    sample_id: str
    answer: str | None
    stop_reason: str
    workspace: WorkspaceState
    events: list[dict]
    decisions: list[Any]
    cost: CostVector
    coverage: float

    def to_dict(self):
        return {"sample_id":self.sample_id,"answer":self.answer,"stop_reason":self.stop_reason,
                "workspace":self.workspace.to_dict() if self.workspace is not None else None,"events":self.events,
                "decisions":[x.descriptor() for x in self.decisions],
                "cost":self.cost.to_dict(),"coverage":self.coverage}


class WorkflowRuntime:
    def __init__(self, policy, *, max_steps=10, max_model_calls=24, max_tool_calls=8,
                 max_packet_bytes=16777216, max_visual_tokens=1024, min_pair_confidence=0.8,
                 role_coverage_curriculum=False):
        for value in (max_steps,max_model_calls,max_tool_calls,max_packet_bytes,max_visual_tokens):
            if not isinstance(value,int) or isinstance(value,bool) or value<=0:
                raise ValueError("workflow budgets must be positive integers")
        self.policy = policy
        self.max_steps,self.max_model_calls,self.max_tool_calls = max_steps,max_model_calls,max_tool_calls
        self.max_packet_bytes,self.max_visual_tokens = max_packet_bytes,max_visual_tokens
        self.min_pair_confidence = min_pair_confidence
        self.role_coverage_curriculum = role_coverage_curriculum

    def _initialize_task(self):pass

    def _controller_decide(self,context,legal):
        return self._call("controller",'Choose a useful next step and return exactly ONE JSON object from legal_action_examples. Never invent IDs or add fields. Revisit visual for missing details. Finish requires an accepted supported answer.',context=context)

    def _finish_allowed(self,candidate):
        return self._coverage([e for e in self.state.evidence if e.evidence_id in candidate.evidence_ids])>=1-1e-8

    def _candidate_terminal_allowed(self, candidate):
        return self.verdicts.get(candidate.candidate_id)=="accept"

    def _on_finish(self, candidate):
        pass

    def _commit(self, actor, **records):
        before = self.state.version
        self.state = WorkspaceReducer.apply(self.state,WorkspaceDelta(
            f"delta-{before}",self.state.episode_id,before,actor,**records))
        self.events.append({"kind":"commit","actor":actor,"decision_index":len(self.policy.records)-1,
                            "before":before,"after":self.state.version})

    def _model_calls_used(self):
        from .generation_failures import model_attempts
        return model_attempts(self.policy)

    def _call(self, role, objective, packet_ids=(), context=None, *, output_format="json",json_field_schema=None):
        if self._model_calls_used() >= self.max_model_calls:
            raise RuntimeError("model_budget_exhausted")
        packet_ids = tuple(packet_ids)
        tokens = sum(self.policy.packets[p].artifact.shape[0] for p in packet_ids)
        if tokens > self.max_visual_tokens:
            raise ValueError("visual token budget exceeded")
        context = dict(context or {})
        feasibility = context.pop('_pointer_feasibility', None)
        prompt = "TASK INFORMATION (context for your assigned role):\n"+json.dumps(
            {"question":self.task.question,"answer_format":self.task.answer_format,
             "context":context or {}},ensure_ascii=False)+"\n\nYOUR REQUIRED ACTION:\n"+objective
        before = self.state.version
        choices = (context or {}).get("legal_action_examples")
        pointer = (context or {}).get('evidence_pointer_constraint')
        options = {} if pointer is None else {'evidence_pointer_constraint': pointer}
        if feasibility is not None:options['pointer_feasibility']=feasibility
        if json_field_schema is not None:options['json_field_schema']=json_field_schema
        try:
            response = self.policy.decide(role,prompt,packet_ids,allowed_actions=choices,output_format=output_format,**options)
        except Exception as error:
            failure=getattr(error,'evors_generation_failure',None)
            if failure is not None:
                from .generation_failures import failure_cost
                self.cost+=failure_cost(failure,role!="controller")
                self.events.append({"kind":"policy_generation_failure","workspace_version":before,
                    "decision_index":None,**failure})
            raise
        record = self.policy.records[-1]
        self.cost += CostVector(agent_calls=int(role!="controller"),model_calls=1,
            model_input_tokens=int(record.input_ids.numel()),model_output_tokens=int(record.output_ids.numel()),
            billed_compute_ms=record.latency_ms)
        self.events.append({"kind":"policy_decision","role":role,"decision_index":len(self.policy.records)-1,
                            "workspace_version":before,"packet_ids":list(packet_ids),"visual_tokens":tokens,
                            "raw_response":response})
        if output_format=="text":
            if not response.strip():
                raise ValueError("empty expert output")
            return response.strip()
        return strict_json(response)

    def _coverage(self, evidence=None):
        evidence = self._support_closure(self.state.evidence if evidence is None else evidence)
        observed = {r:[] for r in self.requirements}
        fingerprints = set()
        for e in evidence:
            fingerprint = e.payload.get("source_fingerprint")
            # Knowledge and self-written revisions are not independent visual confirmation.
            if e.kind != "workflow_visual" or fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            for fid,boxes in e.payload.get("observed_bboxes",{}).items():
                observed.get("frame:"+fid,[]).extend(boxes)
            if e.payload.get("pair_id"):
                observed.get("pair:registered",[]).append(e.payload["reference_bbox"])
        return sum(rectangle_union_area(boxes) for boxes in observed.values())/max(1,len(self.requirements))

    def _support_closure(self, selected):
        """Resolve derived guidance to original observations without creating new evidence."""
        lookup = {e.evidence_id:e for e in self.state.evidence}
        result = {};active = set()
        def visit(eid):
            if eid in active:
                raise ValueError("cyclic evidence provenance")
            if eid in result:
                return
            if eid not in lookup:
                raise ValueError("missing evidence provenance")
            active.add(eid)
            item = lookup[eid]
            for source in item.payload.get("source_evidence_ids",[]):
                visit(source)
            active.remove(eid);result[eid]=item
        for item in selected:
            visit(item.evidence_id)
        return tuple(result.values())

    def _route(self, role, selected, mode):
        evidence = {e.evidence_id:e for e in self.state.evidence}
        selected = refs(selected,evidence)
        if mode not in {"text","tokens"}:
            raise ValueError("mode must be text or tokens")
        disclosed = self._support_closure([evidence[eid] for eid in selected])
        payload_ids = tuple(dict.fromkeys(p for e in disclosed for p in e.artifact_refs)) if mode=="tokens" else ()
        count = sum(self.policy.packets[p].artifact.byte_size for p in payload_ids)
        if self.cost.message_latent_bytes+count > self.max_packet_bytes:
            raise ValueError("communication budget exceeded")
        envelope,grant = self.router.route(self.state,"controller",role,(self.state.task.task_id,*(e.evidence_id for e in disclosed),*payload_ids),
            ("text","structure","latent") if payload_ids else ("text","structure"),self.step)
        inbox = Inbox();inbox.append(envelope)
        view = ObservationBuilder.build(self.state,role,inbox)
        self.cost += grant.cost
        actual_packets = tuple(a.artifact_id for a in view.artifacts if a.kind==ArtifactKind.LATENT)
        context = {"evidence":[e.to_dict() for e in view.evidence],"requirements":sorted(self.requirements)}
        self.events.append({"kind":"route","role":role,"decision_index":len(self.policy.records)-1,
                            "message":envelope.to_dict(),"bytes":count,"packet_ids":list(actual_packets)})
        return context,actual_packets,selected

    def _dispatch_key(self, role, selected, mode):
        if role=="knowledge":
            lookup={e.evidence_id:e for e in self.state.evidence}
            selected=sorted(e.evidence_id for e in self._support_closure([lookup[eid] for eid in selected]) if e.kind=="workflow_visual")
        target=self.state.answer_candidates[-1].candidate_id if role=="verification" and self.state.answer_candidates else None
        return role,tuple(sorted(selected)),mode,target

    def _crop_observation_inputs(self, crop_ids, context):
        """Default historical observation; extensions must preserve provenance."""
        objective = ('Observe ONLY the attached crops. Describe facts relevant to the question in at most two sentences, '
                     'including date and location where useful. Be explicit about missing or ambiguous evidence.')
        return objective, tuple(crop_ids), context, ()

    def _visual(self, context, packets):
        frames = {f.frame_id:f for f in self.task.frames}
        # A perception role may access original views to choose a new ROI.
        originals = tuple(self.global_packets.values())
        boxes = {(0.,0.,1.,1.)}
        # Bounded, progressively refinable windows; no target labels or detections.
        parents = [(0.,0.,1.,1.)]+[n.bbox for n in self.state.spatial_nodes if n.bbox is not None][-4:]
        for x1,y1,x2,y2 in parents:
            xm,ym=(x1+x2)/2,(y1+y2)/2
            boxes.update(((x1,y1,xm,ym),(xm,y1,x2,ym),(x1,ym,xm,y2),(xm,ym,x2,y2)))
        legal=[]
        remaining = self.max_tool_calls-self.cost.tool_calls
        for fid in frames:
            for bbox in sorted(boxes):
                if remaining>=1:
                    legal.append({"op":"crop","frame_id":fid,"bbox":list(bbox)})
        if len(frames)>1 and remaining>=len(frames) and all(f.reference_to_frame is not None and f.alignment_confidence>=self.min_pair_confidence for f in self.task.frames):
            for bbox in sorted(boxes):
                legal.append({"op":"paired_crop","frame_id":self.task.frames[0].frame_id,"bbox":list(bbox)})
        if remaining>=4:
            for fid in frames:
                legal.append({"op":"grid","frame_id":fid,"bbox":[0.,0.,1.,1.]})
        legal = [a for a in legal if json.dumps(a,sort_keys=True) not in self.observed_actions and
                 observation_encodable(self.task.frames,a)]
        if self.training and self.role_coverage_curriculum and len(frames)>1 and self._coverage()<1:
            paired = [a for a in legal if a["op"]=="paired_crop"]
            if paired:
                legal = paired
        if self.training and self.role_coverage_curriculum and not self.state.evidence:
            # Warm-up starts with a complete observation. Adaptive episodes retain
            # all progressive ROIs and must learn to obtain enough spatial coverage.
            legal = [a for a in legal if a["bbox"]==[0.,0.,1.,1.]]
        if not legal:
            raise ValueError("no observation action fits remaining tool budget")
        action = self._call("visual",'Choose a new observation. JSON: {"op":"crop" or "paired_crop" or "grid", "frame_id":"ID", "bbox":[x1,y1,x2,y2]}. Coordinates are normalized [0,1]. paired_crop uses the common reference coordinate frame. grid uses its bbox split into 2x2. Select detail relevant to the question; do not invent an answer.',
            originals,{**context,"frames":[f.descriptor() for f in self.task.frames],"feedback":self.feedback,
                       "legal_action_examples":legal})
        schema(action,{"op","frame_id","bbox"})
        if json.dumps(action,sort_keys=True) in self.observed_actions:
            raise ValueError("identical observation has no new evidence")
        if action["frame_id"] not in frames:
            raise ValueError("unknown frame")
        bbox = _normalized_bbox(action["bbox"])
        if action["op"] == "paired_crop":
            crops = paired_crops(self.task.frames,bbox,min_confidence=self.min_pair_confidence)
        elif action["op"] == "crop":
            f = frames[action["frame_id"]]
            call = CallTool(f"crop-{self.step}","crop",{"image_ref":f.frame_id,"bbox":list(bbox)})
            result = CropTool().execute(call,ToolContext("visual",self.step,
                {f.frame_id:VisualInput(f.frame_id,f.frame_id,f.image)}))
            crops = ((f,result.visuals[0].image,bbox),)
        elif action["op"] == "grid":
            f = frames[action["frame_id"]]
            x1,y1,x2,y2 = bbox;xm,ym = (x1+x2)/2,(y1+y2)/2
            crops = tuple((f,crop_frame(f,b),b) for b in ((x1,y1,xm,ym),(xm,y1,x2,ym),(x1,ym,xm,y2),(xm,ym,x2,y2)))
        else:
            raise ValueError("unknown observation op")
        if self.cost.tool_calls+len(crops) > self.max_tool_calls:
            raise ValueError("tool budget exceeded")
        self.cost += CostVector(tool_calls=len(crops))
        crop_ids = [];artifacts = [];nodes = []
        for index,(frame,image,native_bbox) in enumerate(crops):
            pid = f"tokens-{self.step}-{index}"
            descriptor = self.policy.encode(image,artifact_id=pid,source_image_ref=self.task_ref,
                roi=native_bbox,label=f"{frame.frame_id} / {frame.timestamp} / ROI {native_bbox}")
            artifacts.append(descriptor);crop_ids.append(pid)
            nodes.append(SpatialNode(f"region-{self.step}-{index}",SpatialLevel.REGION,"question_conditioned_roi",
                bbox=native_bbox,parent_id="scene",visual_artifact_refs=(pid,),
                provenance={"frame_id":frame.frame_id,"timestamp":frame.timestamp,"source":frame.source,
                            "reference_bbox":list(bbox),"paired":action["op"]=="paired_crop"}))
        # Visual payloads are committed before model calls, so interrupted observations remain auditable.
        self._commit("visual",artifacts=tuple(artifacts),spatial_nodes=tuple(nodes))
        self.events.append({"kind":"tool","decision_index":len(self.policy.records)-1,
                            "operation":action,"outputs":crop_ids})
        objective, observation_packets, observation_context, contextual_sources = self._crop_observation_inputs(
            crop_ids, {"requirements":sorted(self.requirements),"pair":action["op"]=="paired_crop"})
        observation = self._call("visual", objective, observation_packets, observation_context, output_format="text")
        # Structural coverage comes from executed observations, not model-authored IDs.
        supports = set()
        actual_frames = {frame.frame_id for frame,_,_ in crops}
        for fid in actual_frames:
            supports.add("frame:"+fid)
        if action["op"]=="paired_crop":
            supports.add("pair:registered")
        fingerprint = json.dumps(sorted((image_digest(f.image),f.timestamp,list(b)) for f,_,b in crops))
        evidence = Evidence(f"evidence-{self.step}","workflow_visual",observation,"visual",self.step,
            spatial_node_ids=tuple(n.node_id for n in nodes),artifact_refs=tuple(crop_ids),
            payload={**({"source_evidence_ids":list(contextual_sources)} if contextual_sources else {}),
                     "supports":sorted(supports),"source_fingerprint":fingerprint,
                     "frame_ids":sorted(actual_frames),"reference_bbox":list(bbox),
                     "observed_bboxes":{fid:[list(b) for f,_,b in crops if f.frame_id==fid] for fid in actual_frames},
                     "pair_id":f"pair-{self.step}" if action["op"]=="paired_crop" else None})
        self._commit("visual",evidence=(evidence,))
        self.observed_actions.add(json.dumps(action,sort_keys=True))

    def _expert(self, role, context, packets, selected):
        if not selected:
            raise ValueError("expert needs selected evidence")
        if role=="knowledge":
            summary = self._call(role,'Provide at most two sentences of interpretation guidance grounded in the selected evidence. Do not invent additional visual observations.',packets,context,output_format="text")
            self._commit(role,evidence=(Evidence(f"guidance-{self.step}","workflow_guidance",summary,role,self.step,
                                          payload={"source_evidence_ids":list(selected)}),))
        elif role=="reasoning":
            instruction = answer_instruction(self.task.answer_format)
            text = self._call(role,'Answer the task using only the selected evidence. '+instruction,packets,context,output_format="text")
            answer = answer_valid(text,self.task.answer_format)
            candidate = AnswerCandidate(f"candidate-{self.step}",answer,role,evidence_ids=selected,created_step=self.step)
            self._commit(role,answer_candidates=(candidate,))
        elif role=="verification":
            if not self.state.answer_candidates:
                raise ValueError("verification requires an answer")
            candidate = self.state.answer_candidates[-1]
            if not set(candidate.evidence_ids) <= set(selected):
                raise ValueError("verification must receive all candidate support evidence")
            context = {**context,"candidate":candidate.to_dict()}
            context["legal_action_examples"] = [{"decision":d,"evidence_ids":list(selected)} for d in ("accept","reject","needs_evidence")]
            action = self._call(role,'Independently check the candidate against the selected evidence. Return one legal decision. accept means its answer is supported; reject means it is wrong; needs_evidence means an observation is missing. Do not substitute an answer.',packets,context)
            schema(action,{"decision","evidence_ids"},{"reason","revised_answer"})
            sources = refs(action["evidence_ids"],selected)
            if not sources:
                raise ValueError("verification requires evidence")
            decision = action["decision"]
            if decision not in {"accept","reject","needs_evidence"}:
                raise ValueError("invalid verification decision")
            revision = action.get("revised_answer")
            if decision=="accept" and revision and revision.strip().casefold()!=candidate.answer.strip().casefold():
                raise ValueError("contradictory accept/revised_answer; reinspection required")
            reason = action.get("reason") or self._call(role,'Explain this verification decision in one sentence. Identify missing evidence or an incorrect inference if present.',packets,
                {"candidate":candidate.to_dict(),"evidence":context["evidence"],"decision":decision},output_format="text")
            self.verdicts[candidate.candidate_id] = decision
            self.events.append({"kind":"verification","decision_index":len(self.policy.records)-1,
                                "target":candidate.candidate_id,"decision":decision,"reason":reason,
                                "evidence_ids":list(sources)})
            if decision!="accept":
                self.feedback = reason
            if revision and decision!="accept":
                answer = answer_valid(revision,self.task.answer_format)
                self._commit(role,answer_candidates=(AnswerCandidate(f"revision-{self.step}",answer,role,
                    evidence_ids=sources,created_step=self.step),))
        else:
            raise ValueError("unknown expert")

    def run(self, task: WorkflowTask, *, training=False):
        self.policy.begin_episode(training=training)
        self.training = training
        self.task = task;self.task_ref = "workflow://"+task.sample_id
        self.state = WorkspaceState.initial("workflow-"+task.sample_id,TaskContext(task.sample_id,self.task_ref,
            task.question,answer_format=task.answer_format,choices=question_choices(task.question) if task.answer_format=='choice' else (),
            metadata={"frames":[f.descriptor() for f in task.frames],"scene_id":task.scene_id}))
        self.events = [];self.cost = CostVector();self.verdicts = {};self.feedback = ""
        self.dispatched = set();self.observed_actions = set()
        self.requirements = {"frame:"+f.frame_id for f in task.frames}
        if len(task.frames)>1:
            self.requirements.add("pair:registered")
        self.router = ContextRouter();self.global_packets = {};self.step = 0
        artifacts = []
        for f in task.frames:
            pid = "global-"+f.frame_id
            descriptor = self.policy.encode(f.image,artifact_id=pid,source_image_ref=self.task_ref,
                roi=None,label=f"Overview: {f.frame_id} / {f.timestamp}")
            artifacts.append(descriptor);self.global_packets[f.frame_id] = pid
        self._commit("visual",artifacts=tuple(artifacts),spatial_nodes=(SpatialNode("scene",SpatialLevel.SCENE,"image_collection"),))
        try:self._initialize_task()
        except (ValueError,KeyError,TypeError) as exc:
            self.cost+=CostVector(invalid_actions=1,errors=1)
            self.events.append({'kind':'invalid','reason':str(exc),'decision_index':len(self.policy.records)-1})
            return WorkflowEpisode(task.sample_id,None,'task_initialization_failed',self.state,self.events,list(self.policy.records),self.cost,0.0)
        answer = None;stop = "step_budget_exhausted"
        for self.step in range(self.max_steps):
            try:
                context = {"frames":[f.descriptor() for f in task.frames],
                    "evidence":[{"id":e.evidence_id,"kind":e.kind,"summary":e.summary,"supports":e.payload.get("supports",[])} for e in self.state.evidence],
                    "candidates":[c.to_dict() for c in self.state.answer_candidates],"verdicts":self.verdicts,
                    "coverage":self._coverage(),"requirements":sorted(self.requirements),"feedback":self.feedback,
                    "remaining_model_calls":self.max_model_calls-self._model_calls_used(),
                    "remaining_tool_calls":self.max_tool_calls-self.cost.tool_calls}
                evidence_ids = [e.evidence_id for e in self.state.evidence]
                subsets = [[eid] for eid in evidence_ids]
                if evidence_ids and evidence_ids not in subsets:
                    subsets.append(evidence_ids)
                remaining_calls = self.max_model_calls-self._model_calls_used()-1
                legal = [{"role":"visual","evidence_ids":[],"mode":"tokens"}] if self.cost.tool_calls<self.max_tool_calls and remaining_calls>=2 else []
                for expert in ("knowledge","reasoning","verification"):
                    if remaining_calls < (2 if expert=="verification" else 1):
                        continue
                    if expert=="verification" and (not self.state.answer_candidates or self.state.answer_candidates[-1].candidate_id in self.verdicts):
                        continue
                    for selected in subsets:
                        if expert=="verification" and not set(self.state.answer_candidates[-1].evidence_ids)<=set(selected):
                            continue
                        for mode in ("text","tokens"):
                            signature = self._dispatch_key(expert,selected,mode)
                            if signature in self.dispatched:
                                continue
                            legal.append({"role":expert,"evidence_ids":selected,"mode":mode})
                legal += [{"role":"finish","candidate_id":c.candidate_id} for c in self.state.answer_candidates
                          if self._candidate_terminal_allowed(c) and self._finish_allowed(c)]
                legal.append({"role":"abort"})
                # Optional initial curriculum exercises every role. It never uses labels
                # and is disabled for subsequent adaptive training and evaluation.
                if training and self.role_coverage_curriculum:
                    visited = {r.role for r in self.policy.records}
                    desired = "visual" if not evidence_ids else next(
                        (r for r in ("knowledge","reasoning","verification") if r not in visited),None)
                    if desired is None and any(v=="accept" for v in self.verdicts.values()):
                        desired = "finish"
                    candidates = [a for a in legal if a["role"]==desired]
                    if candidates:
                        legal = candidates
                context["legal_action_examples"] = legal
                action = self._controller_decide(context,legal)
                role = action.get("role")
                if role=="abort":
                    schema(action,{"role"});stop="policy_abort";break
                if role=="finish":
                    schema(action,{"role","candidate_id"})
                    candidate = next((c for c in self.state.answer_candidates if c.candidate_id==action["candidate_id"]),None)
                    if candidate is None or not self._candidate_terminal_allowed(candidate):
                        raise ValueError("finish requires an accepted candidate")
                    support = [e for e in self.state.evidence if e.evidence_id in candidate.evidence_ids]
                    if not self._finish_allowed(candidate):
                        raise ValueError("finish blocked: incomplete visual evidence")
                    self._on_finish(candidate)
                    answer = candidate.answer;stop = "finished";break
                schema(action,{"role","evidence_ids","mode"},{"requirements"})
                if "requirements" in action:
                    if not isinstance(action["requirements"],list) or action["requirements"] and self.step != 0 or any(not isinstance(x,str) or not x.strip() or x.startswith("frame:") for x in action["requirements"]):
                        raise ValueError("target requirements must be nonempty strings specified on first step")
                    self.requirements.update(action["requirements"])
                ctx,packets,selected = self._route(role,action["evidence_ids"],action["mode"])
                if role=="visual":
                    self._visual(ctx,packets)
                else:
                    signature=self._dispatch_key(role,selected,action["mode"])
                    if signature in self.dispatched:
                        raise ValueError("identical expert dispatch has no new evidence")
                    self._expert(role,ctx,packets,selected)
                    self.dispatched.add(signature)
                self._after_role(role)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                self.feedback = str(exc)
                self.cost += CostVector(invalid_actions=1,errors=1,retries=1)
                self.events.append({"kind":"invalid","decision_index":len(self.policy.records)-1,
                                    "reason":self.feedback,"workspace_version":self.state.version})
            except RuntimeError as exc:
                if str(exc)!="model_budget_exhausted":
                    raise
                stop = str(exc);break
        return WorkflowEpisode(task.sample_id,answer,stop,self.state,self.events,list(self.policy.records),self.cost,self._coverage())

    def _after_role(self,role):
        self.feedback = "" if role!='verification' else self.feedback
