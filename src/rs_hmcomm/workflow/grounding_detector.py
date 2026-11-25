"""Frozen CPU detector for real image coordinates, separate from the VLM.

Predicted boxes remain uncertain observations, never ground truth. This is an
adapted Grounding DINO backend, not the ThinkGeo author's GLIP implementation.
"""
import hashlib
import json
import math
from pathlib import Path
import time


class GroundingDetector:
    def __init__(self, model_path, *, box_threshold=.4, text_threshold=.3, device='cpu'):
        import torch
        import inspect
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        path = Path(model_path)
        metadata = json.loads((path/'evors_detector_provenance.json').read_text())
        for name, record in metadata['files'].items():
            digest = hashlib.sha256()
            with (path/name).open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError('detector file changed')
        if not 0 < box_threshold < 1 or not 0 < text_threshold < 1:
            raise ValueError('invalid detector thresholds')
        if device not in ('cpu','cuda:0'):
            raise ValueError('Explicit CPU or the admitted visible CUDA device required')
        self.device=torch.device(device)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        parameters=inspect.signature(self.processor.post_process_grounded_object_detection).parameters
        self.threshold_parameter=('box_threshold'if 'box_threshold'in parameters else 'threshold')
        if self.threshold_parameter not in parameters:
            raise ValueError('Unsupported published detector post-processing API')
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.identity = {'repository': metadata['repository'], 'revision': metadata['revision'],
                         'weights_sha256': metadata['files']['model.safetensors']['sha256'],
                         'device': str(self.device), 'author_GLIP_replication': False,
                         'postprocess_threshold_parameter':self.threshold_parameter}
        self.box_threshold = box_threshold; self.text_threshold = text_threshold

    def detect(self, image, labels, *, clip_to_source=False):
        import torch
        if (not isinstance(labels, list) or not 1 <= len(labels) <= 32
                or any(not isinstance(x, str) or not x.strip() or len(x) > 160 for x in labels)):
            raise ValueError('detector requires one to 32 explicit nonempty object labels')
        labels = [label.strip() for label in labels]
        # Documented punctuation separates classes without reading annotations.
        text = '. '.join(labels)+'.'
        started = time.perf_counter()
        inputs = self.processor(images=image.convert('RGB'), text=text, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        # Preserve one raw neural prediction so post-processing comparisons can
        # reuse it. The caller may save this small carrier beside its source.
        self.last_raw_prediction={
            'logits':output.logits.detach().float().cpu().numpy(),
            'pred_boxes':output.pred_boxes.detach().float().cpu().numpy(),
            'input_ids':inputs.input_ids.detach().cpu().numpy()}
        result = self.processor.post_process_grounded_object_detection(
            output, inputs.input_ids, **{self.threshold_parameter:self.box_threshold},
            text_threshold=self.text_threshold, target_sizes=[image.size[::-1]])[0]
        return {**source_box_predictions(result, image.size, clip_to_source=clip_to_source),
                'original_image_size': list(image.size),
                'detector_identity': self.identity, 'requested_labels': labels,
                'box_threshold': self.box_threshold, 'text_threshold': self.text_threshold,
                'latency_ms': (time.perf_counter()-started)*1000, 'reference_used': False,
                'coordinate_space': 'original image pixels', 'ground_truth_claim': False}


def source_box_predictions(result, size, *, clip_to_source=False):
    """Expose the visible extent of edge-crossing objects when a recipient requests pixels."""
    width, height = size
    raw_boxes = result['boxes']
    if clip_to_source:
        from torchvision.ops import clip_boxes_to_image
        visible_boxes = clip_boxes_to_image(raw_boxes, (height, width))
    else:
        visible_boxes = raw_boxes
    label_key = 'text_labels' if 'text_labels' in result else 'labels'
    objects, rejected = [], []
    for raw, visible, score, label in zip(raw_boxes, visible_boxes, result['scores'], result[label_key]):
        original = raw.tolist()
        coordinates = visible.tolist()
        x1, y1, x2, y2 = coordinates
        if not all(map(math.isfinite, original)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            rejected.append({'label': str(label), 'original_bbox_pixels': original,
                             'reason': 'invalid or outside original image'})
            continue
        obj = {'label': str(label), 'score': float(score), 'original_bbox_pixels': coordinates,
               'bbox': [x1/width, y1/height, x2/width, y2/height]}
        if coordinates != original:
            obj['unclipped_detector_bbox_pixels'] = original
        objects.append(obj)
    return {'objects': objects, 'rejected_predictions': rejected, 'postprocess_label_key': label_key}


def center_distance(first, second, *, meters_per_pixel=None):
    """Compute from two actual predicted boxes; precision follows localization."""
    centers = []
    for row in (first, second):
        box = row['original_bbox_pixels']
        if len(box) != 4 or any(type(x) not in {int, float} or not math.isfinite(x) for x in box):
            raise ValueError('invalid original pixel box')
        x1, y1, x2, y2 = box
        if not x1 < x2 or not y1 < y2:
            raise ValueError('empty predicted box')
        centers.append([(x1+x2)/2, (y1+y2)/2])
    distance = math.hypot(centers[1][0]-centers[0][0], centers[1][1]-centers[0][1])
    result = {'center_coordinates_pixels': centers, 'center_distance_pixels': distance,
              'measurement_definition': 'Euclidean distance between predicted bounding-box centers',
              'localization_uncertainty_preserved': True}
    if meters_per_pixel is not None:
        if type(meters_per_pixel) not in {int, float} or not math.isfinite(meters_per_pixel) or meters_per_pixel <= 0:
            raise ValueError('invalid public GSD')
        result.update(meters_per_pixel=meters_per_pixel, center_distance_meters=distance*meters_per_pixel)
    return result
