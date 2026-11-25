"""Candidate-hidden private fitness; model agreement is never ground truth."""
from .contextual_observation import ContextualObservationRuntime
from .runtime import question_choices, schema


class IndependentFitnessRuntime(ContextualObservationRuntime):
    def __init__(self,*args,independent_fitness=False,fitness_native_pixels_only=False,**kwargs):
        if fitness_native_pixels_only and not independent_fitness:
            raise ValueError('native-only fitness requires independent fitness')
        super().__init__(*args,**kwargs)
        self.independent_fitness=independent_fitness
        self.fitness_native_pixels_only=fitness_native_pixels_only

    def _call(self,role,objective,packet_ids=(),context=None,**kwargs):
        marker='Evaluate the support contributed by this ORIGINAL SOURCE GROUP to the private candidate.'
        if not (self.independent_fitness and role=='verification' and objective.startswith(marker)
                and self.task.answer_format in {'choice','yes_no'}):
            return super()._call(role,objective,packet_ids,context,**kwargs)
        context=dict(context or {})
        candidate=context['candidate']['answer']
        letters=[value.split('.',1)[0].strip() for value in question_choices(self.task.question)]
        alternatives=letters if self.task.answer_format=='choice' else ['Yes','No']
        if not alternatives or candidate.casefold() not in {a.casefold() for a in alternatives}:
            raise ValueError('private fitness candidate outside public answer space')
        # Full source records are authorized. Remove the candidate in every mode;
        # native-only additionally excludes inherited summaries, but text MAS
        # still needs its authorized descriptions to have an observation.
        pixels_only=(bool(packet_ids) and self.fitness_native_pixels_only and
                     all(self.policy.packets[pid].recipe is None for pid in packet_ids))
        evidence=context['evidence']
        if pixels_only:
            evidence=[{'evidence_id':e['evidence_id'],'artifact_refs':e['artifact_refs'],
                'frame_ids':e.get('payload',{}).get('frame_ids',[]),
                'observed_bboxes':e.get('payload',{}).get('observed_bboxes',{})} for e in evidence]
        visible={'source_group':context['source_group'],'evidence':evidence,
            'candidate_answer_disclosed':False,'prior_summaries_disclosed':not pixels_only,
            'legal_action_examples':[{'independent_answer':a} for a in alternatives]+[{'independent_answer':'uncertain'}]}
        action=super()._call(role,
            'Independently answer the ORIGINAL task using only this authorized ORIGINAL SOURCE GROUP. '
            'No proposed candidate answer is provided. Choose independent_answer from the public options, '
            'or uncertain when this source lacks required detail, frames or visible properties. '
            'Earlier descriptions, if supplied, are fallible model interpretations. '
            'When native views are attached, inspect their actual pixels. Repeated views are one source, not votes.',
            packet_ids,visible,**kwargs)
        schema(action,{'independent_answer'})
        answer=action['independent_answer']
        if answer not in alternatives+['uncertain']:
            raise ValueError('unknown independent private-fitness answer')
        verdict='uncertain' if answer=='uncertain' else 'supported' if answer.casefold()==candidate.casefold() else 'contradicted'
        self.events.append({'kind':'independent_private_fitness','source_group':context['source_group'],
            'candidate_answer':candidate,'independent_answer':answer,'mapped_verdict':verdict,
            'decision_index':len(self.policy.records)-1,'candidate_disclosed_in_prompt':False,
            'prior_summaries_disclosed':not pixels_only,'packet_ids':list(packet_ids),
            'reference_used':False,'model_agreement_is_ground_truth':False})
        return {'verdict':verdict}
