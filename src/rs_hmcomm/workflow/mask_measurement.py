"""Original-pixel mask measurements; segmentation predictions are not truth."""
import hashlib
import math
from pathlib import Path
from .public_scale import public_pixel_scale
import time
import numpy as np
from .evidence import image_digest


def summarize_masks(masks, threshold_pixels, *, meters_per_pixel=None):
    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError('original binary instance masks required')
    if type(threshold_pixels) not in {int, float} or not math.isfinite(threshold_pixels) or threshold_pixels <= 0:
        raise ValueError('positive pixel threshold required')
    if meters_per_pixel is not None and (type(meters_per_pixel) not in {int, float} or
            not math.isfinite(meters_per_pixel) or meters_per_pixel <= 0):
        raise ValueError('positive public metric scale required')
    areas = masks.sum(axis=(1, 2), dtype=np.int64).tolist()
    hashes = [hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() for mask in masks]
    valid = [i for i, area in enumerate(areas) if area > 0]
    unique = list(dict.fromkeys(hashes[i] for i in valid))
    unique_areas = [areas[next(i for i in valid if hashes[i] == h)] for h in unique]
    count = sum(area < threshold_pixels for area in unique_areas)
    total = len(unique_areas)
    result = {'mask_pixel_areas': areas, 'mask_sha256': hashes,
        'empty_mask_indices': [i for i, area in enumerate(areas) if not area],
        'raw_nonempty_masks': len(valid), 'exact_duplicate_masks': len(valid)-total,
        'distinct_nonempty_predicted_masks': total, 'below_threshold_masks': count,
        'threshold_pixels': threshold_pixels,
        'below_threshold_percentage': 100*count/total if total else None,
        'union_pixels': int(np.any(masks, axis=0).sum()) if len(masks) else 0,
        'all_objects_detected_or_correctly_classified': False,
        'percentage_denominator': 'distinct nonempty predicted masks; exhaustive object coverage unverified',
        'localization_and_segmentation_uncertainty_preserved': True, 'ground_truth_claim': False}
    if meters_per_pixel is not None:
        result.update(meters_per_pixel=meters_per_pixel,
            threshold_square_meters=threshold_pixels*meters_per_pixel**2,
            mask_square_meter_areas=[area*meters_per_pixel**2 for area in areas])
    return result


class SamMaskMeasurement:
    def __init__(self, checkpoint, *, model_type='vit_h', expected_sha256=None, max_boxes=256, device='cpu'):
        from segment_anything import SamPredictor, sam_model_registry
        if type(max_boxes) is not int or max_boxes < 1:
            raise ValueError('positive mask-instance bound required')
        path = Path(checkpoint); digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
        if expected_sha256 is None or digest.hexdigest() != expected_sha256:
            raise ValueError('verified segmentation weight hash required')
        if device not in ('cpu','cuda:0'):
            raise ValueError('Explicit CPU or the admitted visible CUDA device required')
        self.model = sam_model_registry[model_type](checkpoint=str(path)).to(device).eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.predictor = SamPredictor(self.model); self.max_boxes = max_boxes
        self.identity = {'model_type': model_type, 'weights_sha256': digest.hexdigest(), 'device': device}

    def segment(self, image, objects, output_prefix, *, image_feature_cache=None):
        import torch
        if not isinstance(objects, list) or len(objects) > self.max_boxes:
            raise ValueError('bounded actual detection list required; do not truncate')
        image = image.convert('RGB'); width, height = image.size
        boxes = []
        for obj in objects:
            box = obj['original_bbox_pixels']
            if len(box) != 4 or any(type(v) not in {float, int} or not math.isfinite(v) for v in box):
                raise ValueError('finite original pixel coordinates required')
            x1,y1,x2,y2 = box
            if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                raise ValueError('segmentation box outside actual image')
            boxes.append(np.array(box, dtype=np.float32))
        prefix = Path(output_prefix)
        if prefix.with_suffix('.npz').exists(): raise ValueError('mask artifact already exists')
        prefix.parent.mkdir(parents=True,exist_ok=True)
        masks = []; scores = []; started = time.perf_counter()
        reused=False;feature_receipt=None
        with torch.inference_mode():
            if boxes and image_feature_cache is not None:
                entry=image_feature_cache;path=Path(entry['path'])
                if (hashlib.sha256(path.read_bytes()).hexdigest()!=entry['SHA256']
                        or entry['source_image_sha256']!=image_digest(image)
                        or entry['model_weights_SHA256']!=self.identity['weights_sha256']):
                    raise ValueError('SAM image feature cache source or weights differ')
                with np.load(path,allow_pickle=False)as archive:
                    features=archive['features'].copy();input_size=tuple(archive['input_size'].tolist())
                expected=self.predictor.transform.get_preprocess_shape(height,width,self.model.image_encoder.img_size)
                if (features.shape!=(1,256,64,64)or features.dtype!=np.float32
                        or not np.isfinite(features).all()or input_size!=expected):
                    raise ValueError('Actual original SAM image-feature shape or display differs')
                self.predictor.features=torch.from_numpy(features).to(self.predictor.device)
                self.predictor.original_size=(height,width);self.predictor.input_size=input_size
                self.predictor.is_image_set=True;reused=True
            elif boxes:self.predictor.set_image(np.asarray(image))
            if boxes:
                feature_file=prefix.with_suffix('.features.npz')
                if feature_file.exists():raise ValueError('SAM source feature artifact already exists')
                np.savez_compressed(feature_file,features=self.predictor.get_image_embedding().detach().float().cpu().numpy(),input_size=np.asarray(self.predictor.input_size,dtype=np.int64))
                feature_receipt=dict(path=str(feature_file),SHA256=hashlib.sha256(feature_file.read_bytes()).hexdigest(),
                    source_image_sha256=image_digest(image),model_weights_SHA256=self.identity['weights_sha256'])
            for box in boxes:
                predicted, quality, _ = self.predictor.predict(box=box, multimask_output=False)
                if predicted.shape != (1, height, width) or predicted.dtype != np.bool_:
                    raise ValueError('segmentation did not return original binary mask')
                if quality.shape != (1,) or not np.isfinite(quality).all():
                    raise ValueError('invalid predicted mask quality')
                masks.append(predicted[0]); scores.append(float(quality[0]))
        masks = np.stack(masks) if masks else np.zeros((0,height,width), dtype=np.bool_)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prefix.with_suffix('.npz'), masks=masks)
        artifact = prefix.with_suffix('.npz')
        return {'masks': masks, 'source_image_sha256': image_digest(image),
            'original_image_size': [width,height], 'source_boxes_pixels': [b.tolist() for b in boxes],
            'mask_artifact': str(artifact), 'mask_artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'predicted_mask_quality': scores, 'segmentation_identity': self.identity,
            'latency_ms': (time.perf_counter()-started)*1000,
            'frozen_auxiliary_inferences': int(bool(boxes)and not reused)+len(boxes),
            'image_encoder_features_reused':reused,'image_feature_cache':feature_receipt,
            'mask_quality_is_semantic_correctness': False, 'reference_used': False}
