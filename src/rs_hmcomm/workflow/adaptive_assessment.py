"""Paid policy selection of visual requirement checks; no credit for omission."""
from .adaptive_private_search import AdaptivePrivateSearchRuntime
from .runtime import schema


class AdaptiveAssessmentRuntime(AdaptivePrivateSearchRuntime):
    adaptive_visual_assessment = True

    def _mandatory_visual_assessment_calls(self):
        # Assessments can be deferred without creating support. Admit the
        # affordable observation branch; choose checks afterwards from the
        # actual remaining budget, preserving the full closing reserve.
        return 0

    def _assessment_requirements(self, selected, latest):
        observed = {frame.frame_id: [] for frame in self.task.frames}
        for item in selected:
            for fid, boxes in item.payload.get('observed_bboxes', {}).items():
                if fid in observed:observed[fid].extend(boxes)
        eligible = tuple(r for r in self.ledger.requirements
                         if all(observed[fid] for fid in r.frame_ids))
        ids = [item.evidence_id for item in selected]
        remaining = self.max_model_calls-self._model_calls_used()
        reserve = self._closing_call_bound(
            [item.evidence_id for item in self.state.evidence], self._reasoning_call_bound())
        maximum = min(len(eligible), max(0, remaining-reserve-1))
        if not maximum:
            self.events.append({'kind':'adaptive_visual_assessment_deferred',
                'latest_evidence_id':latest.evidence_id,'selected_evidence_ids':ids,
                'remaining_model_calls':remaining,'closing_call_reserve':reserve,
                'reason':'no observable requirement or no budget beyond closing reserve',
                'omitted_requirements_receive_credit':False,'selection_model_calls':0})
            return ()
        action = self._call('controller',
            'Choose which task requirements deserve an additional native-view check now. '
            'Return requirement_ids, an array of distinct eligible IDs, or [] to defer all checks. '
            'Checking costs one model call per ID, plus this selection call and communication. '
            'Use checks when the current views can resolve uncertainty or contradictions. '
            'Skipping does not establish support and does not submit an answer.',
            context={'latest_evidence_id':latest.evidence_id,'selected_evidence_ids':ids,
                'eligible_requirements':[r.to_dict() for r in eligible],
                'existing_support':self.ledger.support(),
                'maximum_checks':maximum,'remaining_model_calls':remaining,
                'closing_call_reserve':reserve},
            json_field_schema=[['requirement_ids',{'array':{'enum':[r.requirement_id for r in eligible]},
                                                  'maximum':maximum}]])
        schema(action, {'requirement_ids'})
        chosen = action['requirement_ids']
        allowed = {r.requirement_id for r in eligible}
        if (not isinstance(chosen,list) or len(chosen)>maximum
                or any(not isinstance(key,str) or key not in allowed for key in chosen)
                or len(set(chosen))!=len(chosen)):
            raise ValueError('unknown, duplicate or unaffordable assessment requirements')
        self.events.append({'kind':'adaptive_visual_assessment_dispatch',
            'latest_evidence_id':latest.evidence_id,'selected_evidence_ids':ids,
            'requirement_ids':list(chosen),'eligible_requirement_ids':sorted(allowed),
            'decision_index':len(self.policy.records)-1,'selection_model_calls':1,
            'maximum_checks':maximum,'closing_call_reserve':reserve,'reference_used':False,
            'omitted_requirements_receive_credit':False})
        lookup={r.requirement_id:r for r in eligible}
        return tuple(lookup[key] for key in chosen)
