"""Auxiliary detection and geometry use real outputs, never benchmark traces."""
from .public_scale import public_pixel_scale
from PIL import Image
from rs_hmcomm.contracts import CostVector
from .native_benchmark_tools import NativeDetectionToolbox
from .benchmark_budget import BudgetTaskBoundRuntime
from .grounding_detector import center_distance
from .benchmark_tools import calculate


def public_gsd(question):
    return public_pixel_scale(question)


class ExpertMeasurementToolbox(NativeDetectionToolbox):
    def __init__(self, *args, detector, public_question, classifier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = detector; self.question = public_question
        self.classifier = classifier
        self.detections = {}

    def execute(self, action):
        tool = action.get('tool')
        if tool == 'calculate':
            if set(action) != {'tool', 'expression'}:
                raise ValueError('calculate requires exactly tool and expression')
            result = {'value': calculate(action['expression'], rounding=True)}
        elif tool == 'detect_objects':
            if set(action) != {'tool', 'asset', 'labels'}:
                raise ValueError('detect_objects requires exactly tool, asset and labels')
            with Image.open(self.path(action['asset'])) as image:
                if image.mode not in {'RGB', 'RGBA', 'L', 'P'}:
                    raise ValueError('scientific raster needs native numeric tools')
                result = self.detector.detect(image.convert('RGB'), action['labels'])
            detection_id = 'detection-'+str(len(self.detections))
            result = {**result, 'asset': action['asset'], 'detection_id': detection_id,
                      'frozen_auxiliary_inferences': 1}
            self.detections[detection_id] = result
        elif tool == 'classify_detected_objects':
            if self.classifier is None:
                raise ValueError('no frozen crop classifier configured')
            if set(action) != {'tool', 'detection_id', 'labels'}:
                raise ValueError('classification requires exactly tool, detection_id and labels')
            from .remoteclip_expert import validate_labels
            validate_labels(action['labels'])
            if action['detection_id'] not in self.detections:
                raise ValueError('classification requires actual earlier detection')
            detection = self.detections[action['detection_id']]
            rows = []; elapsed = 0.
            with Image.open(self.path(detection['asset'])) as image:
                image = image.convert('RGB')
                for index, obj in enumerate(detection['objects']):
                    x1, y1, x2, y2 = obj['original_bbox_pixels']
                    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                        raise ValueError('classification box outside actual image')
                    # Fixed 10% context, identical to the diagnostic protocol.
                    dx = (x2-x1)*.1; dy = (y2-y1)*.1
                    crop_box = [max(0, int(x1-dx)), max(0, int(y1-dy)),
                                min(image.width, int(x2+dx+1)), min(image.height, int(y2+dy+1))]
                    classified = self.classifier.classify(image.crop(crop_box), action['labels'])
                    elapsed += classified['latency_ms']
                    rows.append({'object_index': index, 'crop_bbox_pixels': crop_box,
                                 'original_bbox_pixels': obj['original_bbox_pixels'], 'classification': classified})
            result = {'detection_id': action['detection_id'], 'source_asset': detection['asset'],
                      'objects': rows, 'all_detection_objects_preserved': True,
                      'latency_ms': elapsed, 'frozen_auxiliary_inferences': len(rows),
                      'reference_used': False, 'ground_truth_claim': False,
                      'scores_are_calibrated_probabilities': False}
        elif tool == 'measure_center_distance':
            if set(action) != {'tool', 'detection_id', 'object_indices'}:
                raise ValueError('measure_center_distance requires exactly tool, detection_id and object_indices')
            indices = action['object_indices']
            if not isinstance(indices, list) or len(indices) != 2 or any(type(i) != int or i < 0 for i in indices) or indices[0] == indices[1]:
                raise ValueError('two distinct nonnegative object indices required')
            if action['detection_id'] not in self.detections:
                raise ValueError('measurement requires an actual earlier successful detection')
            detection = self.detections[action['detection_id']]
            objects = detection['objects']
            if max(indices) >= len(objects):
                raise ValueError('object index outside actual detection')
            result = center_distance(objects[indices[0]], objects[indices[1]],
                                     meters_per_pixel=public_gsd(self.question))
            result.update(detection_id=action['detection_id'], object_indices=indices,
                          source_asset=detection['asset'], public_gsd_source='initial task question only',
                          reference_used=False, ground_truth_claim=False)
        else:
            return super().execute(action)
        self.events.append({'kind': 'executed_tool', 'action': action, 'output': result,
                            'origin': 'actual frozen detector / deterministic geometry; never reference trace'})
        return result


class ExpertMeasurementRuntime(BudgetTaskBoundRuntime):
    def __init__(self, *args, detector, classifier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = detector
        self.classifier = classifier

    def create_toolbox(self, task):
        return ExpertMeasurementToolbox(task['assets'], self.policy, output=self.output,
                                       detector=self.detector, classifier=self.classifier, public_question=task['question'])

    def public_task(self, task):
        public = super().public_task(task)
        classification_tools = [] if self.classifier is None else [{
            'tool': 'classify_detected_objects',
            'required_fields': {'detection_id': 'actual earlier detection_id',
                                'labels': 'explicit competing class names appropriate to the task'},
            'output': 'all predicted object indices with frozen RemoteCLIP crop cosine scores; no automatic pruning or calibrated probabilities'}]
        return {**public, 'available_tools': [*public['available_tools'], *classification_tools,
            {'tool': 'detect_objects', 'asset': next(iter(task['assets']), ''),
             'required_fields': {'labels': 'list of actual target class names, chosen from the question'},
             'output': 'detection_id and predicted boxes in ORIGINAL pixels; frozen CPU expert, not ground truth'},
            {'tool': 'measure_center_distance',
             'required_fields': {'detection_id': 'actual earlier detection_id',
                                 'object_indices': 'two distinct indices into that detection objects list'},
             'output': 'pixel center distance; meters only if initial question explicitly states GSD'}],
             'measurement_guidance': 'For distances, use detected objects and measure_center_distance. An inspect description is not a pixel measurement. Predicted boxes can be inaccurate; report uncertainty rather than inventing geometry.'}

    def tool_result_cost(self, result):
        return CostVector(billed_compute_ms=result.get('latency_ms', 0.)
                          if result.get('frozen_auxiliary_inferences', 0) > 0 else 0.)
