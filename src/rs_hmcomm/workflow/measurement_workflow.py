"""Optional sampled detector/SAM actions inside the whole trainable workflow."""
import hashlib
import json
from pathlib import Path
import time

from rs_hmcomm.contracts import CostVector, Evidence
from .evidence import image_digest
from .runtime import schema
from .segmentation_tools import SegmentationMeasurementToolbox
from .task_grounded_domain import TaskGroundedDomainRuntime


MEASUREMENT_SCHEMAS = {
    'detect_objects': ['tool', 'asset', 'labels'],
    'segment_detected_objects': ['tool', 'detection_id'],
    'summarize_mask_areas': ['tool', 'segmentation_id', 'threshold_pixels'],
    'measure_center_distance': ['tool', 'detection_id', 'object_indices']}


MEASUREMENT_CAPABILITIES = {
    'detect_objects': 'Predict original-frame boxes for explicit target classes chosen from the question. Prefer concrete visually meaningful class names; broad superclass labels can miss objects. Predictions may omit or misclassify targets.',
    'segment_detected_objects': 'Use an actual earlier detection_id to predict original-pixel instance masks, rather than treating bounding boxes as object areas.',
    'summarize_mask_areas': 'Use an actual segmentation_id to calculate mask pixel areas and the fraction of distinct nonempty predicted masks strictly below threshold_pixels. This does not certify complete scene coverage.',
    'measure_center_distance': 'Measure the original-pixel center distance of two actual detected objects. Convert to meters only using GSD explicitly supplied in the public question.'}


def measurement_observation(value):
    """Deterministic semantic view; raw results stay in the execution audit.

    Tool IDs and content hashes address masks. A per-run filesystem location
    and wall-clock timings carry no geometric or physical evidence.
    """
    if isinstance(value, dict):
        return {key: measurement_observation(item) for key, item in value.items()
                if key not in {'latency_ms', 'actual_execution_ms', 'mask_artifact'}}
    if isinstance(value, list):
        return [measurement_observation(item) for item in value]
    return value


def frame_measurement_toolbox(task, policy, *, detector, segmenter, output, classifier=None):
    """The same original-pixel tool session for a single or multi-Agent."""
    import tempfile
    from PIL import Image
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='episode-', dir=output))
    assets = {}
    for frame in task.frames:
        name = hashlib.sha256(frame.frame_id.encode()).hexdigest()+'.png'
        path = folder/name
        image = frame.image.convert('RGB')
        image.save(path)
        with Image.open(path) as saved:
            if image_digest(saved) != image_digest(image):
                raise ValueError('measurement input pixel mismatch')
        assets[frame.frame_id] = path
    return SegmentationMeasurementToolbox(assets, policy, output=folder/'artifacts',
        detector=detector, segmenter=segmenter, classifier=classifier,
        public_question=task.question)


def measurement_context(toolbox, frame_ids, executed):
    detections = {key: {'object_count': len(value['objects']),
                      'valid_object_indices': list(range(len(value['objects'])))}
                  for key, value in toolbox.detections.items()}
    ready = ['detect_objects']
    if detections:
        ready.append('segment_detected_objects')
    if toolbox.segmentations:
        ready.append('summarize_mask_areas')
    distance_ids = [key for key, value in detections.items() if value['object_count'] >= 2]
    if distance_ids:
        ready.append('measure_center_distance')
    parameter_types = {
        'detect_objects': {'asset': 'one authorized_frame_id string',
                           'labels': 'nonempty list of concrete class-name strings chosen from the public task'},
        'segment_detected_objects': {'detection_id': 'one actual detection_ids string'},
        'summarize_mask_areas': {'segmentation_id': 'one actual segmentation_ids string',
                               'threshold_pixels': 'positive finite number for the threshold requested in the task'},
        'measure_center_distance': {'detection_id': 'one distance_ready_detection_ids string',
            'object_indices': 'exactly TWO DISTINCT integer indices from that detection valid_object_indices; zero-based. Never strings or detection IDs.'}}
    examples = [{'tool':'segment_detected_objects','detection_id':key} for key in detections]
    examples += [{'tool':'measure_center_distance','detection_id':key,'object_indices':[0,1]}
                 for key in distance_ids]
    return {'authorized_frame_ids': list(frame_ids),
            'tool_schemas': {name:MEASUREMENT_SCHEMAS[name] for name in ready},
            'tool_capabilities': {name:MEASUREMENT_CAPABILITIES[name] for name in ready},
            'parameter_types': {name:parameter_types[name] for name in ready},
            'currently_ready_tools': ready,
            'dependency_rule': 'Choose only currently_ready_tools. If no detection exists, detect first. '
                'If fewer than two objects were detected, center distance is unavailable. '
                'An ID exists only when returned by a successful executed tool; never invent an ID.',
            'detection_ids': list(detections), 'actual_detections': detections,
            'distance_ready_detection_ids': distance_ids,
            'segmentation_ids': list(toolbox.segmentations),
            'real_ID_request_examples': examples,
            'example_rule': 'Examples demonstrate valid parameter types and existing IDs; choose task-relevant objects, not necessarily indices 0 and 1.',
            'already_executed_requests': [json.loads(s) for s in sorted(executed)],
            'measurement_limitations': 'Use actual earlier tool IDs and original pixels. '
                'Bounding-box area is not object mask area. Exhaustive coverage and class '
                'correctness are unverified. Metric values require explicit public GSD.'}


class MeasurementWorkflowRuntime(TaskGroundedDomainRuntime):
    def __init__(self, *args, detector, segmenter, measurement_output, classifier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = detector
        self.segmenter = segmenter
        self.classifier = classifier
        self.measurement_output = Path(measurement_output)

    def _initialize_task(self):
        # Never retain predictions from a previous episode, even for the same scene.
        self.measurement_toolbox = None
        self.measurement_source_records = {}
        self.measurement_count = 0
        self.measurement_no_progress = 0
        self.executed_measurements = set()
        super()._initialize_task()

    def _measurement_session(self):
        if self.measurement_toolbox is not None:
            return self.measurement_toolbox
        self.measurement_toolbox = frame_measurement_toolbox(self.task, self.policy,
            output=self.measurement_output, detector=self.detector,
            segmenter=self.segmenter, classifier=self.classifier,
        )
        return self.measurement_toolbox

    def _controller_decide(self, context, legal):
        return super()._controller_decide({**context,
            'optional_visual_measurements': MEASUREMENT_CAPABILITIES,
            'measurement_rule': 'Request these through the visual stage only when needed. '
                'Detection coverage and class predictions are uncertain; do not require all tools.'}, legal)

    def _controller_role_choices(self, choices, context):
        choices = super()._controller_role_choices(choices, context)
        if not self.mechanisms.progress_guard:
            return choices
        # The optional acquisition selector is one additional sampled call.
        # Retain the parent's closing reserve, including candidate verification.
        pending = bool(self.state.answer_candidates and
            self.state.answer_candidates[-1].candidate_id not in self.verdicts)
        ids = [e.evidence_id for e in self.state.evidence]
        closing = self._closing_call_bound(ids, self._reasoning_call_bound())
        reserve = len(ids)+8 if pending else closing+4
        observation = 3+(len(self.ledger.requirements) if self.ledger else 0)+int(self.mechanisms.objects)
        remaining = self.max_model_calls-self._model_calls_used()
        return [c for c in choices if c['role'] != 'visual' or remaining >= observation+reserve+getattr(self, 'acquisition_selector_call_bound', 1)]

    def _dispatch_key(self, role, selected, mode):
        if role != 'knowledge':
            return super()._dispatch_key(role, selected, mode)
        lookup = {e.evidence_id:e for e in self.state.evidence}
        sources = self._support_closure([lookup[eid] for eid in selected])
        # A new measured value is not the same Knowledge input as an earlier caption.
        # Guidance paraphrases still collapse to the same underlying observations.
        selected = sorted(e.evidence_id for e in sources
                          if e.kind in {'workflow_visual', 'workflow_measurement'})
        return role, tuple(selected), mode, None

    def _assess_measurement(self, evidence_id):
        if self.ledger is None:
            return
        if self.max_model_calls-self._model_calls_used() < len(self.ledger.requirements):
            self.events.append({'kind':'measurement_assessment_deferred',
                                'reason':'remaining model budget', 'evidence_id':evidence_id})
            return
        try:
            context, packets, _ = self._route('verification', [evidence_id], 'tokens')
        except ValueError as error:
            if 'budget exceeded' not in str(error):
                raise
            self.events.append({'kind':'measurement_assessment_deferred',
                                'reason':str(error), 'evidence_id':evidence_id})
            return
        actual = [e for e in context['evidence']
                  if e['kind'] in {'workflow_visual', 'workflow_measurement'}]
        actual_ids = [e['evidence_id'] for e in actual]
        full_frames = {fid for fid, pid in self.global_packets.items() if pid in packets}
        aid = 'assessment-'+str(len(self.ledger.assessments))
        for requirement in self.ledger.requirements:
            if set(requirement.frame_ids) <= full_frames:
                action = self._call('verification',
                    'Assess this requirement using ONLY the actual attached source views and '
                    'source-linked executed measurement results. Check the requested quantity '
                    'and units. Predicted masks and boxes do not guarantee complete target '
                    'coverage or correct classification. A percentage over detected masks '
                    'does not by itself establish a whole-scene percentage. Use uncertain '
                    'when coverage, calibration, dates or measurements remain insufficient. '
                    'Earlier verbal claims or semantic ratings are not independent evidence.',
                    packets, {'requirement':requirement.to_dict(), 'observations':actual,
                        'legal_action_examples':[{'verdict':v} for v in
                                                 ('supported','uncertain','contradicted')]})
                schema(action, {'verdict'})
            else:
                action = {'verdict':'uncertain'}
            pair_valid = any(e['payload'].get('pair_id') and
                set(requirement.frame_ids) <= set(e['payload'].get('frame_ids', ())) for e in actual)
            group = ','.join(sorted(f.descriptor()['sha256'] for f in self.task.frames
                                   if f.frame_id in requirement.frame_ids))
            self.ledger.record(requirement.requirement_id, aid, group, action['verdict'],
                global_valid=set(requirement.frame_ids) <= full_frames,
                pair_valid=pair_valid, observed_evidence_ids=actual_ids)
        ratings = self.ledger.assessments[aid]
        self._commit('verification', evidence=(Evidence(aid, 'workflow_guidance',
            json.dumps(ratings), 'verification', self.step,
            payload={'source_evidence_ids':actual_ids, 'semantic_assessment':ratings,
                     'measurement_verifier_is_model_estimate':True}),))
        self.events.append({'kind':'measurement_semantic_assessment','evidence_id':aid,
            'actual_evidence_ids':actual_ids,'ratings':ratings,
            'actual_attached_whole_frames':sorted(full_frames),
            'host_forced_acceptance':False, 'reference_used':False})

    def _visual(self, context, packets):
        remaining = self.max_tool_calls-self.cost.tool_calls
        if remaining <= 0:
            raise ValueError('measurement/observation tool budget exhausted')
        toolbox = self._measurement_session()
        action = self._acquisition_action(context, toolbox, remaining)
        if action.get('kind') == 'observation':
            schema(action, {'kind'})
            before = self.cost.tool_calls
            result = super()._visual(context, packets)
            if self.cost.tool_calls > before:self.measurement_no_progress = 0
            return result
        schema(action, {'kind', 'action'})
        if action['kind'] != 'measurement' or not isinstance(action['action'], dict):
            raise ValueError('invalid visual measurement action')
        request = action['action']
        if request.get('tool') not in MEASUREMENT_SCHEMAS:
            self.measurement_no_progress = getattr(self, 'measurement_no_progress', 0)+1
            raise ValueError('unsupported workflow measurement tool')
        signature = json.dumps(request, sort_keys=True)
        if signature in self.executed_measurements:
            self.measurement_no_progress = getattr(self, 'measurement_no_progress', 0)+1
            raise ValueError('identical frozen measurement already executed')
        self.cost += CostVector(tool_calls=1)
        started = time.perf_counter()
        event = {'kind': 'workflow_measurement', 'request': request,
                 'decision_index': len(self.policy.records)-1, 'reference_used': False}
        try:
            result = toolbox.execute(request)
            event.update(success=True, result=result)
            self.measurement_no_progress = 0
        except Exception as error:
            self.measurement_no_progress = getattr(self, 'measurement_no_progress', 0)+1
            event.update(success=False, error=str(error))
            raise
        finally:
            elapsed = (time.perf_counter()-started)*1000
            self.cost += CostVector(billed_compute_ms=elapsed)
            event['actual_execution_ms'] = elapsed
            self.events.append(event)
        parent = request.get('segmentation_id', request.get('detection_id'))
        sources = [] if parent is None else [self.measurement_source_records[parent]]
        asset = result.get('asset', result.get('source_asset'))
        frame = next(f for f in self.task.frames if f.frame_id == asset)
        eid = 'measurement-'+str(self.measurement_count)
        self.measurement_count += 1
        observed_result = measurement_observation(result)
        self._commit('visual', evidence=(Evidence(eid, 'workflow_measurement',
            json.dumps(observed_result, ensure_ascii=False), 'visual', self.step,
            artifact_refs=(self.global_packets[asset],) if parent is None else (),
            payload={'source_evidence_ids': sources, 'frame_ids': [asset],
                'source_frame': frame.descriptor(), 'measurement_result': observed_result,
                'ground_truth_claim': False, 'global_inventory_verified': False,
                'semantic_support_automatically_granted': False}),))
        for key in ('detection_id', 'segmentation_id'):
            # Preserve the producing record; subsequent summaries must not replace it.
            if key in result:
                self.measurement_source_records.setdefault(result[key], eid)
        self.executed_measurements.add(signature)
        # Assess through a sampled verifier and actual routed views, not mask existence.
        self._assess_measurement(eid)

    def _acquisition_action(self, context, toolbox, remaining):
        blocked = getattr(self, 'measurement_no_progress', 0) >= 2
        guard = {'legal_action_examples':[{'kind':'observation'}]} if blocked else {}
        return self._call('controller',
            'Choose ONE visual acquisition action. Return exactly kind=observation to use '
            'the existing crop/grid/date observation planner, or kind=measurement and '
            'action containing one tool request. Measurement tools are optional, not a '
            'mandatory chain. Detect original-frame objects before segmenting; use actual '
            'detection and segmentation IDs. Do not invent pixel measurements. '
            'A detected subset is not a complete inventory.',
            context={**context, **measurement_context(toolbox,
                [f.frame_id for f in self.task.frames], self.executed_measurements),
                'remaining_tool_calls': remaining,
                'measurement_retries_blocked_until_actual_observation': blocked,
                **guard,
                'action_schema': {'observation': {'kind': 'observation'},
                    'measurement': {'kind': 'measurement', 'action': 'one tool request'}},
            })
