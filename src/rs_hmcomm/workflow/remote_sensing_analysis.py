"""Source-linked domain analysis, sampled in the existing expert completion."""
from dataclasses import replace
from .remote_sensing_specialists import RemoteSensingSpecialistRuntime
from .runtime import schema, refs


def validate_domain_analysis(value, frame_ids):
    schema(value, {'task_inference', 'constraints', 'missing_observations'})
    inference = value['task_inference']
    if not isinstance(inference, str) or not inference.strip() or len(inference) > 1024:
        raise ValueError('bounded task inference required')
    constraints = value['constraints']
    if not isinstance(constraints, list) or len(constraints) > 3 or any(
            not isinstance(x, str) or not x.strip() or len(x) > 512 for x in constraints):
        raise ValueError('bounded domain constraints required')
    observations = value['missing_observations']
    if not isinstance(observations, list) or len(observations) > 3:
        raise ValueError('bounded missing observations required')
    for observation in observations:
        schema(observation, {'need', 'frame_ids'})
        if not isinstance(observation['need'], str) or not observation['need'].strip() or len(observation['need']) > 512:
            raise ValueError('bounded observation need required')
        if not refs(observation['frame_ids'], frame_ids):
            raise ValueError('observation request needs actual frame IDs')
    return value


class RemoteSensingAnalysisRuntime(RemoteSensingSpecialistRuntime):
    def _controller_decide(self, context, legal):
        # The base scheduler summarizes evidence and omits payload fields.
        # Disclose analysis only alongside records already in its state view.
        by_id = {e.evidence_id: e.to_dict()['payload'] for e in self.state.evidence}
        evidence = []
        for row in context.get('evidence', ()):
            payload = by_id.get(row.get('id'), {})
            if 'domain_analysis' in payload:
                row = {**row, 'domain_analysis': payload['domain_analysis'],
                    'remote_sensing_specialist': payload['remote_sensing_specialist'],
                    'source_evidence_ids': payload['source_evidence_ids'],
                    'analysis_is_model_inference': True, 'missing_observations_executed': False}
            evidence.append(row)
        return super()._controller_decide({**context, 'evidence': evidence}, legal)

    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        if role != 'knowledge' or not getattr(self, 'active_domain_specialist', None):
            return super()._call(role, objective, packet_ids, context, **kwargs)
        self.pending_domain_analysis = None
        objective = objective.replace(
            'Return JSON with exactly guidance (string) and calculations (list).',
            'Return JSON with exactly guidance (string), calculations (list), and domain_analysis (object).')
        objective += (
            ' domain_analysis has exactly task_inference (concise answer to the missing inference, '
            'or explicit reason it cannot be established), constraints (at most three concise '
            'task-specific physical/geographic/temporal/observation constraints), and '
            'missing_observations (at most three objects with exactly need and frame_ids). '
            'Use actual frame IDs only. State what new view or measurement is needed, without '
            'claiming it was executed. Distinguish pedestrian network distance from straight-line '
            'distance when relevant. Do not simply repeat the image caption. An empty list is '
            'valid when nothing further is needed; avoid inventing constraints or measurements.')
        # Keep the caller's arithmetic budget/source contract inside the domain
        # completion instead of replacing its calculations schema with a free array.
        fields=kwargs.get('json_field_schema')
        calculations=dict(fields)['calculations'] if fields else 'array'
        kwargs['json_field_schema']=[['guidance','string'],['calculations',calculations],
            ['domain_analysis',{'object':[['task_inference','string'],
                ['constraints',{'array':'string','maximum':3}],
                ['missing_observations',{'array':{'object':[['need','string'],
                    ['frame_ids',{'array':{'enum':[f.frame_id for f in self.task.frames]},'minimum':1,'maximum':len(self.task.frames)}]]},'maximum':3}]]}]]
        value = super()._call(role, objective, packet_ids, context, **kwargs)
        schema(value, {'guidance', 'calculations', 'domain_analysis'})
        self.pending_domain_analysis = validate_domain_analysis(
            value['domain_analysis'], [f.frame_id for f in self.task.frames])
        # Existing arithmetic validation and billing remain the sole execution path.
        return {'guidance': value['guidance'], 'calculations': value['calculations']}

    def _commit(self, actor, **records):
        analysis = getattr(self, 'pending_domain_analysis', None)
        if actor == 'knowledge' and getattr(self, 'active_domain_specialist', None) and analysis is not None:
            records['evidence'] = tuple(replace(e, payload={**e.payload,
                'domain_analysis': analysis, 'analysis_is_model_inference': True,
                'missing_observations_executed': False}) for e in records.get('evidence', ()))
            self.events.append({'kind': 'remote_sensing_domain_analysis',
                'specialist': self.active_domain_specialist, 'analysis': analysis,
                'source_evidence_ids': list(records['evidence'][0].payload['source_evidence_ids']),
                'decision_index': len(self.policy.records)-1, 'extra_model_calls': 0,
                'reference_used': False, 'new_observation_claimed': False})
        result = super()._commit(actor, **records)
        if actor == 'knowledge':
            self.pending_domain_analysis = None
        return result
