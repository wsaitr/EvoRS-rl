"""Persist measured verification diagnostics as routed, source-linked guidance."""
from rs_hmcomm.contracts import Evidence
from .bounded_pointer import BoundedPointerRuntime


class RejectionFeedbackRuntime(BoundedPointerRuntime):
    def _expert(self, role, context, packets, selected):
        start = len(self.events)
        result = super()._expert(role,context,packets,selected)
        if role != 'verification':return result
        verdicts = [event for event in self.events[start:] if event['kind']=='verification']
        if not verdicts or verdicts[-1]['decision']=='accept':return result
        verdict = verdicts[-1]
        sources = tuple(verdict['evidence_ids'])
        if not sources:return result
        # The original verdict is retained. Guidance is not a visual observation,
        # an independent vote, a revised answer, or a permission to finish.
        eid = 'verification-feedback-'+str(self.step)
        summary = ('Untrusted model verification diagnostic for '+verdict['target']+': '+verdict['decision']+'. '
            +verdict['reason']+' Re-evaluate against the actual source evidence; this diagnostic may be wrong.')
        self._commit('verification',evidence=(Evidence(eid,'workflow_guidance',summary,'verification',self.step,
            payload={'source_evidence_ids':list(sources),'target_candidate_id':verdict['target'],
                     'decision':verdict['decision'],'diagnostic_only':True,'reference_used':False}),))
        self.events.append({'kind':'verification_feedback_persisted','evidence_id':eid,
                            'source_evidence_ids':list(sources),'target_candidate_id':verdict['target'],
                            'automatically_revised_answer':False,'independent_visual_observation':False})
        return result
