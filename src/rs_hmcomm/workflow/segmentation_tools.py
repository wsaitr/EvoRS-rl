"""Real source-bound segmentation, available to every tool-using baseline."""
from PIL import Image
from .evidence import image_digest
from .expert_measurement_tools import ExpertMeasurementToolbox, ExpertMeasurementRuntime
from .mask_measurement import public_pixel_scale, summarize_masks


class SegmentationMeasurementToolbox(ExpertMeasurementToolbox):
    def __init__(self, *args, segmenter, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmenter = segmenter; self.segmentations = {}

    def execute(self, action):
        if action.get('tool') == 'detect_objects':
            result = super().execute(action)
            with Image.open(self.path(result['asset'])) as image:
                result['source_image_sha256'] = image_digest(image.convert('RGB'))
            return result
        if action.get('tool') not in {'segment_detected_objects', 'summarize_mask_areas'}:
            return super().execute(action)
        if action['tool'] == 'segment_detected_objects':
            if set(action) != {'tool','detection_id'}:
                raise ValueError('segmentation requires an actual detection_id only')
            if action['detection_id'] not in self.detections:
                raise ValueError('segmentation requires earlier actual detection')
            detection = self.detections[action['detection_id']]
            sid = 'segmentation-'+str(len(self.segmentations))
            with Image.open(self.path(detection['asset'])) as image:
                image = image.convert('RGB')
                if image_digest(image) != detection['source_image_sha256']:
                    raise ValueError('source image changed after detection')
                segmented = self.segmenter.segment(image, detection['objects'], self.output/sid)
            if segmented['source_image_sha256'] != detection['source_image_sha256']:
                raise ValueError('segmentation returned different source')
            self.segmentations[sid] = {**segmented, 'source_asset': detection['asset'],
                'detection_id': action['detection_id']}
            result = {k:v for k,v in self.segmentations[sid].items() if k != 'masks'}
            result.update(segmentation_id=sid, ground_truth_claim=False)
        else:
            if set(action) != {'tool','segmentation_id','threshold_pixels'}:
                raise ValueError('area summary requires segmentation_id and threshold_pixels')
            if action['segmentation_id'] not in self.segmentations:
                raise ValueError('area summary requires earlier actual masks')
            record = self.segmentations[action['segmentation_id']]
            result = summarize_masks(record['masks'], action['threshold_pixels'],
                meters_per_pixel=public_pixel_scale(self.question))
            result.update(segmentation_id=action['segmentation_id'], source_asset=record['source_asset'],
                mask_artifact_sha256=record['mask_artifact_sha256'],
                scale_source='initial public question only', threshold_source='model tool request; agreement with question requires verification',
                reference_used=False)
        self.events.append({'kind':'executed_tool','action':action,'output':result,
                            'origin':'actual SAM masks / deterministic original pixel area'})
        return result


class SegmentationMeasurementRuntime(ExpertMeasurementRuntime):
    def __init__(self, *args, segmenter, **kwargs):
        super().__init__(*args, **kwargs); self.segmenter = segmenter

    def create_toolbox(self, task):
        return SegmentationMeasurementToolbox(task['assets'], self.policy, output=self.output,
            detector=self.detector, classifier=self.classifier, segmenter=self.segmenter,
            public_question=task['question'])

    def public_task(self, task):
        public = super().public_task(task)
        return {**public, 'available_tools': [*public['available_tools'],
            {'tool':'segment_detected_objects','required_fields':{'detection_id':'actual earlier detection_id'},
             'output':'original-pixel masks and predicted mask quality; no semantic coverage guarantee'},
            {'tool':'summarize_mask_areas','required_fields':{'segmentation_id':'actual earlier segmentation_id',
             'threshold_pixels':'positive requested threshold'},
             'output':'mask pixel areas, threshold fraction, metric areas only with public GSD'}],
            'segmentation_guidance':'Segmentation is required for object pixel areas; bounding-box area is not object area. '
                'Report predicted-mask counts and missing detection coverage. A percentage over detected masks does not prove all vehicles were found.'}
