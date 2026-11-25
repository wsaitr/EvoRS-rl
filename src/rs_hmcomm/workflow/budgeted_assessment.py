"""Bounded, incremental native verification without credit for omitted views."""
import json
from rs_hmcomm.contracts import Evidence, CostVector
from .context_anchor import ContextAnchoredRuntime
from .runtime import rectangle_union_area


def assessment_subset(evidence, packets, latest_id, *, max_tokens, byte_allowance):
    """Prefer whole-scene reference and newest observation; no answer access."""
    views = [e for e in evidence if e.kind == 'workflow_visual']
    by_id = {e.evidence_id:e for e in views}
    preferred = [by_id[eid] for eid in ('overview', latest_id) if eid in by_id]
    preferred += list(reversed(views))
    selected = []; ids = set(); packet_ids = set()
    for item in preferred:
        if item.evidence_id in ids: continue
        prospective = packet_ids | set(item.artifact_refs)
        native_bytes = sum(packets[p].artifact.byte_size for p in prospective)
        tokens = sum(packets[p].artifact.shape[0] for p in prospective)
        # Reserve room for the actual evidence payload as well as envelope metadata.
        records = [*selected,item]
        metadata = len(json.dumps([e.to_dict() for e in records],ensure_ascii=False).encode())
        if tokens > max_tokens or native_bytes+metadata+65536 > byte_allowance: continue
        selected.append(item); ids.add(item.evidence_id); packet_ids = prospective
    return tuple(selected)


class BudgetedAssessmentRuntime(ContextAnchoredRuntime):
    def _assessment_requirements(self, selected, latest):
        return self.ledger.requirements

    def _assess_visual(self, latest):
        if self.mechanisms.objects:
            try: self._extract_objects(latest)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
                self.cost += CostVector(invalid_actions=1,errors=1)
                self.events.append({'kind':'optional_component_failure','component':'objects',
                    'reason':str(error),'evidence_id':latest.evidence_id})
        if self.ledger is None:return
        remaining = self.max_packet_bytes-self.cost.message_latent_bytes-self.cost.message_structure_bytes
        # Keep half the remaining communication for reasoning and final verification.
        selected = assessment_subset(self.state.evidence,self.policy.packets,latest.evidence_id,
            max_tokens=self.max_visual_tokens,byte_allowance=max(0,remaining/2))
        if not selected:
            self.events.append({'kind':'assessment_deferred','reason':'no native evidence fits reserved budget',
                                'latest_evidence_id':latest.evidence_id})
            return
        requirements = self._assessment_requirements(selected, latest)
        if not requirements:return
        # Disable implicit additions: the selection already contains its exact
        # feasible overview when present. Only these actual views may get credit.
        anchor = self.anchor_global; self.anchor_global = False
        try:
            context, packets, _ = self._route('verification',[e.evidence_id for e in selected],'tokens')
        except ValueError as error:
            if str(error) != 'communication budget exceeded':raise
            self.events.append({'kind':'assessment_deferred','reason':str(error),'latest_evidence_id':latest.evidence_id})
            return
        finally:self.anchor_global = anchor
        observed = {frame.frame_id:[] for frame in self.task.frames}
        for item in selected:
            for fid,boxes in item.payload.get('observed_bboxes',{}).items():observed[fid].extend(boxes)
        assessment_id = 'assessment-'+str(len(self.ledger.assessments))
        for requirement in requirements:
            choices = [{'verdict':value} for value in ('supported','uncertain','contradicted')]
            if all(observed[fid] for fid in requirement.frame_ids):
                action = self._call('verification',
                    'Assess whether ONLY the attached actual views establish this requirement. Omitted previous views '
                    'are not evidence for this assessment. A supported negative answer is valid; missing views are uncertain.',
                    packets,{'requirement':requirement.to_dict(),'observations':context['evidence'],
                             'legal_action_examples':choices})
            else:
                action = {'verdict':'uncertain'}
                self.events.append({'kind':'assessment_missing_required_frame',
                    'requirement_id':requirement.requirement_id,'actual_evidence_ids':[e.evidence_id for e in selected]})
            global_valid = all(rectangle_union_area(observed[fid])>=1-1e-8 for fid in requirement.frame_ids)
            pair_valid = any(item.payload.get('pair_id') and set(requirement.frame_ids)<=set(item.payload['frame_ids']) for item in selected)
            sources = ','.join(sorted(frame.descriptor()['sha256'] for frame in self.task.frames
                                     if frame.frame_id in requirement.frame_ids))
            self.ledger.record(requirement.requirement_id,assessment_id,sources,action['verdict'],
                global_valid=global_valid,pair_valid=pair_valid,
                observed_evidence_ids=[item.evidence_id for item in selected])
        ratings = self.ledger.assessments[assessment_id]
        self._commit('verification',evidence=(Evidence(assessment_id,'workflow_guidance',json.dumps(ratings),
            'verification',self.step,payload={'source_evidence_ids':[item.evidence_id for item in selected],
                                            'semantic_assessment':ratings}),))
        self.events.append({'kind':'incremental_semantic_assessment','evidence_id':assessment_id,
                            'actual_evidence_ids':[item.evidence_id for item in selected],
                            'omitted_views_receive_credit':False,'ratings':ratings,'coverage':self.ledger.coverage()})
