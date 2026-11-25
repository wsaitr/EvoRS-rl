"""Learned submission, keeping factual endorsement separate from source validity."""
from .runtime import answer_valid,rectangle_union_area
from .early_candidate_closing import EarlyCandidateClosingRuntime
from .feasible_pointer_workflow import FeasiblePointerWorkflowRuntime


class PolicyTerminalMixin:
    def _historical_terminal(self):
        return self.mechanisms.fixed_schedule or (self.training and self.role_coverage_curriculum)

    def _finish_allowed(self,candidate):
        if self._historical_terminal():return super()._finish_allowed(candidate)
        try:answer_valid(candidate.answer,self.task.answer_format)
        except (ValueError,TypeError):return False
        lookup={e.evidence_id:e for e in self.state.evidence}
        if not candidate.evidence_ids or not set(candidate.evidence_ids)<=set(lookup):return False
        closed=self._support_closure([lookup[eid] for eid in candidate.evidence_ids])
        visual=[e for e in closed if e.kind in {'workflow_visual','workflow_measurement'}]
        if not visual or not any(e.artifact_refs for e in visual):return False
        requirements=self.ledger.requirements if self.ledger else ()
        for requirement in requirements:
            observed={fid:[] for fid in requirement.frame_ids}
            for e in visual:
                for fid,boxes in e.payload.get('observed_bboxes',{}).items():
                    if fid in observed:observed[fid].extend(boxes)
            if requirement.scope=='global':
                if not all(rectangle_union_area(observed[fid])>=1-1e-8 for fid in observed):return False
            elif not all(observed[fid] for fid in observed):return False
            if requirement.kind=='change':
                if not any(e.payload.get('pair_id') and set(requirement.frame_ids)<=set(e.payload.get('frame_ids',())) for e in visual):return False
        return True

    def _candidate_terminal_allowed(self,candidate):
        if self._historical_terminal():return super()._candidate_terminal_allowed(candidate)
        return self._finish_allowed(candidate)

    def _pointer_progress_choices(self,choices,ids,remaining,finish):
        if self._historical_terminal():return super()._pointer_progress_choices(choices,ids,remaining,finish)
        if not self.mechanisms.progress_guard:return choices
        pointer=self._pointer_call_bound(ids)
        result=[c for c in choices if c['role'] not in {'knowledge','verification'} or remaining>=pointer+(4 if c['role']=='verification' else 1)]
        closing=self._closing_call_bound(ids,self._reasoning_call_bound())
        result=[c for c in result if c['role']!='reasoning' or remaining>=closing]
        visual=3+self._mandatory_visual_assessment_calls()+int(self.mechanisms.objects)
        result=[c for c in result if c['role']!='visual' or remaining>=visual+closing+4]
        if self.max_steps-self.step<=1:
            result=[c for c in result if c['role'] in {'finish','abort'}]
        return result

    def _controller_role_choices(self,choices,context):
        if self._historical_terminal():return super()._controller_role_choices(choices,context)
        # Keep learned finish-versus-review choices; no priority rule may silently
        # turn a tentative candidate into a mandatory verifier transaction.
        if any(c['role']=='finish' for c in choices):return choices
        return super()._controller_role_choices(choices,context)

    def _controller_decide(self,context,legal):
        if self._historical_terminal():return super()._controller_decide(context,legal)
        return super()._controller_decide({**context,'submission_rule':
            'Finish submits a source-linked candidate for external task evaluation. '
            'It does not assert that a model verifier or semantic rating is ground truth. '
            'Consider disagreement, missing facts and remaining resources. You can acquire '
            'evidence, revise, verify, submit a tentative answer or abort. '
            'Do not assume finishable means correct.'},legal)

    def _on_finish(self,candidate):
        super()._on_finish(candidate)
        if not self._historical_terminal():
            self.events.append({'kind':'policy_terminal_submission',
                'candidate_id':candidate.candidate_id,'source_evidence_ids':list(candidate.evidence_ids),
                'verifier_decision':self.verdicts.get(candidate.candidate_id),
                'semantic_coverage_estimate':self._coverage(),
                'factual_endorsement_claimed':False,'external_reference_used':False,
                'decision_index':len(self.policy.records)-1})


class PolicyTerminalRuntime(PolicyTerminalMixin,EarlyCandidateClosingRuntime):
    pass


class PolicyTerminalMeasurementRuntime(PolicyTerminalMixin,FeasiblePointerWorkflowRuntime):
    pass


from .feasible_pointer_workflow import FeasiblePointerMixin
from .task_grounded_domain import TaskGroundedDomainRuntime


class PolicyTerminalDomainRuntime(PolicyTerminalMixin,FeasiblePointerMixin,TaskGroundedDomainRuntime):
    pass
