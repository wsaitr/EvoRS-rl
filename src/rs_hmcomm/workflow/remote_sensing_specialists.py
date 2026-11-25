"""Sampled domain-expert dispatch inside the trainable knowledge stage."""
from dataclasses import replace
from .source_certificate import SourceCertificateRuntime
from .runtime import schema


SPECIALISTS = {
    'physical_measurement': {
        'name': 'Remote sensing physics and measurement expert',
        'instruction': 'Check pixel coordinates, spatial resolution, physical dimensions and units. '
            'Convert pixel lengths or areas to physical quantities only with an explicit public '
            'scale and source-linked measurements. Distinguish native pixels from processor '
            'canvas coordinates. RGB appearance alone does not establish calibrated reflectance, '
            'temperature, elevation or SAR scattering. State missing calibration rather than guess.',
    },
    'geography': {
        'name': 'Geography and land-use expert',
        'instruction': 'Interpret land cover, land use, settlement morphology and relationships '
            'among roads, water, buildings and terrain using addressed spatial evidence. '
            'Separate visible land cover from inferred function. Do not invent geographic '
            'coordinates, named locations, elevation, accessibility or causal explanations.',
    },
    'temporal_change': {
        'name': 'Multi-temporal change expert',
        'instruction': 'Compare corresponding regions in the ordered before and after views. '
            'Check object additions/removals, land-cover conversion, shape or extent, and '
            'surface or construction-state changes, including pavement, markings and unfinished '
            'ground. An unchanged footprint or alignment does not establish an unchanged scene. '
            'For each reported change, state the visible before-to-after difference and location; '
            'separate an uncertain material or functional label from the visible change itself. '
            'Check illumination, seasonal appearance and misregistration as alternative '
            'explanations. Quantitative change requires comparable scale and correspondence; '
            'state missing registration or dates. A single frame cannot prove change.',
    },
    'observation_quality': {
        'name': 'Remote sensing observation-quality expert',
        'instruction': 'Assess whether spatial resolution, crop extent, occlusion, clouds, shadows '
            'or missing bands prevent the requested inference. Explain which actual observation '
            'would resolve uncertainty. Do not assume unseen spectral bands, revisit times or '
            'sensor parameters. Never create image detail by inventing a measurement.',
    },
}


def domain_dispatch_summary(events):
    """Count actual sampled dispatches; an available expert is not an invocation."""
    sequence = [event['specialist'] for event in events
                if event.get('kind') == 'remote_sensing_specialist_dispatch']
    return {'dispatch_sequence': sequence,
            'dispatch_counts': {key: sequence.count(key) for key in SPECIALISTS},
            'distinct_experts': len(set(sequence)),
            'domain_stage_skipped': not sequence}


class RemoteSensingSpecialistRuntime(SourceCertificateRuntime):
    def _controller_role_choices(self, choices, context):
        result = super()._controller_role_choices(choices, context)
        if not self.mechanisms.progress_guard:
            return result
        pending = bool(self.state.answer_candidates and
                       self.state.answer_candidates[-1].candidate_id not in self.verdicts)
        closing = self._closing_call_bound([], self._reasoning_call_bound())
        remaining = self.max_model_calls-self._model_calls_used()
        # Three pointer calls, one specialist selection, one knowledge completion.
        return [c for c in result if c['role'] != 'knowledge' or pending or remaining >= 5+closing]

    def _expert(self, role, context, packets, selected):
        if role != 'knowledge':
            return super()._expert(role, context, packets, selected)
        if not selected:
            raise ValueError('domain specialist requires addressed evidence')
        action = self._call('controller',
            'Choose ONE remote sensing specialist for the current missing inference. '
            'Return one legal specialist action. Choose expertise from the task and '
            'available evidence; no reference answer is provided.',
            context={**context, 'specialists': SPECIALISTS,
                     'legal_action_examples': [{'specialist': k} for k in SPECIALISTS]})
        schema(action, {'specialist'})
        key = action['specialist']
        if not isinstance(key, str) or key not in SPECIALISTS:
            raise ValueError('unknown remote sensing specialist')
        self.events.append({'kind': 'remote_sensing_specialist_dispatch', 'specialist': key,
                            'source_evidence_ids': list(selected),
                            'decision_index': len(self.policy.records)-1,
                            'reference_used': False, 'selection_model_calls': 1})
        self.active_domain_specialist = key
        try:
            return super()._expert(role, {**context, 'remote_sensing_specialist': SPECIALISTS[key],
                'frame_metadata': [f.descriptor() for f in self.task.frames],
                'radiometric_calibration': 'not supplied by RGB frame descriptors'}, packets, selected)
        finally:
            self.active_domain_specialist = None

    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        key = getattr(self, 'active_domain_specialist', None)
        if role == 'knowledge' and key:
            objective = SPECIALISTS[key]['instruction']+' '+objective
        return super()._call(role, objective, packet_ids, context, **kwargs)

    def _commit(self, actor, **records):
        key = getattr(self, 'active_domain_specialist', None)
        if actor == 'knowledge' and key and 'evidence' in records:
            records['evidence'] = tuple(replace(e, payload={**e.payload,
                'remote_sensing_specialist': key,
                'source_evidence_ids_preserved': True}) for e in records['evidence'])
        return super()._commit(actor, **records)
