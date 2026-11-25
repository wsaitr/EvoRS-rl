"""Bound crop descriptions to actual same-frame context, without semantic credit."""
from .adaptive_assessment import AdaptiveAssessmentRuntime


class ContextualObservationRuntime(AdaptiveAssessmentRuntime):
    def __init__(self, *args, contextual_observation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.contextual_observation = contextual_observation

    def _crop_observation_inputs(self, crop_ids, context):
        historical = super()._crop_observation_inputs(crop_ids, context)
        if not self.contextual_observation:
            return historical
        overview = next((e for e in self.state.evidence if e.evidence_id == 'overview'), None)
        if overview is None:
            return historical
        # Bind frame identity by executed spatial provenance, never model prose.
        nodes = {pid: node for node in self.state.spatial_nodes
                 for pid in node.visual_artifact_refs}
        crops = []
        for pid in crop_ids:
            node = nodes[pid]
            crops.append({'packet_id':pid, 'frame_id':node.provenance['frame_id'],
                          'timestamp':node.provenance['timestamp'], 'roi':list(node.bbox)})
        frame_ids = {row['frame_id'] for row in crops}
        selected = []
        inputs = list(crop_ids)
        tokens = sum(self.policy.packets[pid].artifact.shape[0] for pid in inputs)
        for pid in overview.artifact_refs:
            node = nodes[pid]
            if node.provenance['frame_id'] not in frame_ids or pid in inputs:
                continue
            packet = self.policy.packets[pid]
            if packet.recipe is not None:
                raise ValueError('contextual observation needs an actual native overview')
            count = packet.artifact.shape[0]
            if tokens + count > self.max_visual_tokens:
                continue
            inputs.append(pid); tokens += count
            selected.append({'packet_id':pid, 'frame_id':node.provenance['frame_id'],
                             'timestamp':node.provenance['timestamp'], 'roi':list(node.bbox)})
        self.events.append({'kind':'contextual_crop_observation', 'crop_packet_ids':list(crop_ids),
            'overview_packet_ids':[r['packet_id'] for r in selected],
            'actual_input_packet_ids':inputs, 'visual_token_count':tokens,
            'source_evidence_ids':['overview'] if selected else [],
            'reference_used':False, 'semantic_support_granted':False,
            'extra_image_encodings':0, 'internal_cached_visual_context':True})
        if not selected:
            return historical
        context = {**context, 'actual_view_bindings':{'detail_views':crops, 'reference_views':selected},
                   'contextual_source_evidence_ids':['overview'], 'interpretation_verified':False}
        objective = ('Observe the attached detail crops together with their declared same-frame whole-image references. '
            'Describe question-relevant visible facts in at most two sentences. Use actual_view_bindings to connect '
            'a detail to its surrounding labels, coordinates, orientation and neighbors; do not transfer facts across '
            'dates or frames. Distinguish what the crop shows from what only its whole-image reference shows. '
            'Reinspect the actual pixels instead of repeating an earlier summary. State missing or ambiguous evidence.')
        return objective, tuple(inputs), context, ('overview',)
