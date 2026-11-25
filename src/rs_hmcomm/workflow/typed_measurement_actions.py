"""Hierarchical sampled acquisition with actual tool/asset address constraints.

Semantic labels, thresholds and object choice remain policy decisions. No
annotation-derived values, fixed tool chain or automatic support is supplied.
"""
import json
from .measurement_workflow import MeasurementWorkflowRuntime, measurement_context
from .runtime import schema


def acquisition_choices(toolbox, frame_ids, executed, *, blocked=False):
    choices = [{'kind': 'observation'}]
    if blocked:
        return choices
    choices += [{'kind': 'measurement', 'tool': 'detect_objects', 'asset': fid}
                for fid in frame_ids]
    for key, detection in toolbox.detections.items():
        if detection['objects']:
            request = {'tool': 'segment_detected_objects', 'detection_id': key}
            if json.dumps(request, sort_keys=True) not in executed:
                choices.append({'kind': 'measurement', **request})
        if len(detection['objects']) >= 2:
            choices.append({'kind': 'measurement', 'tool': 'measure_center_distance', 'detection_id': key})
    choices += [{'kind': 'measurement', 'tool': 'summarize_mask_areas', 'segmentation_id': key}
                for key in toolbox.segmentations]
    return choices


class TypedMeasurementWorkflowRuntime(MeasurementWorkflowRuntime):
    # Address selection plus optional free semantic parameters. Segmentation is
    # fully addressed by the first call; no artificial second completion.
    acquisition_selector_call_bound = 2

    def _acquisition_action(self, context, toolbox, remaining):
        blocked = getattr(self, 'measurement_no_progress', 0) >= 2
        frame_ids = [f.frame_id for f in self.task.frames]
        choices = acquisition_choices(toolbox, frame_ids, self.executed_measurements, blocked=blocked)
        metadata = measurement_context(toolbox, frame_ids, self.executed_measurements)
        action = self._call('controller',
            'Choose ONE currently legal visual acquisition. Observation asks the existing '
            'crop/grid/date planner for new visual evidence. Measurement chooses only the '
            'tool and existing source address in a legal action; semantic parameters are '
            'requested separately when needed. Select only a task-relevant action. '
            'No mandatory tool chain and no guarantee of detection classification or coverage.',
            context={**context, **metadata, 'remaining_tool_calls': remaining,
                     'measurement_retries_blocked_until_actual_observation': blocked,
                     'legal_action_examples': choices})
        if action not in choices:
            raise ValueError('acquisition is outside current addressed choices')
        self.events.append({'kind': 'typed_visual_acquisition', 'action': action,
            'decision_index': len(self.policy.records)-1, 'reference_used': False})
        if action['kind'] == 'observation':
            return action
        request = {key: value for key, value in action.items() if key != 'kind'}
        tool = request['tool']
        if tool == 'segment_detected_objects':
            return {'kind': 'measurement', 'action': request}
        fields = {'detect_objects': {'labels'}, 'summarize_mask_areas': {'threshold_pixels'},
                  'measure_center_distance': {'object_indices'}}[tool]
        parameters = self._call('controller',
            'Fill ONLY the semantic parameter fields '+str(sorted(fields))+' for the '
            'chosen acquisition. Return a JSON object with exactly those fields. '
            'Do not repeat tool, kind, action or source IDs. Choose concrete visually meaningful '
            'labels from the public question, not invented sensor properties. A detector '
            'label is not a validated damage class. Use the public requested threshold '
            'and actual distinct zero-based object indices when applicable. '
            'Do not repeat an already executed request or invent measurements.',
            context={**context, **metadata, 'chosen_request': request,
                     'required_parameter_fields': sorted(fields)})
        schema(parameters, fields)
        return {'kind': 'measurement', 'action': {**request, **parameters}}
