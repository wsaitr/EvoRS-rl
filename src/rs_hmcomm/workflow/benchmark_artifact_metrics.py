"""External image-output validation; textual 'drawn' claims receive no credit."""
import ast
import hashlib
from pathlib import Path
from PIL import Image
from .semantic import bbox_iou


def annotation_score(reference, events, assets):
    """Declared geometry proxy, not the official multimodal benchmark judge.

    This function runs AFTER model inference. Reference boxes are never tools
    or observations. Unsupported annotation formats remain explicitly unscored.
    """
    target = []
    for call in reference.get('annotated_tools', []):
        if call.get('name') != 'DrawBox': continue
        arguments = call.get('arguments', {})
        if not isinstance(arguments, dict): return {'supported': False, 'reason': 'unparsed reference arguments'}
        name = arguments.get('image'); box = arguments.get('bbox')
        if name not in assets: return {'supported': False, 'reason': 'reference uses derived or missing image'}
        try:
            if isinstance(box, str): box = ast.literal_eval(box)
            if not isinstance(box, (tuple, list)) or len(box) != 4: raise ValueError('single box required')
            with Image.open(assets[name]) as image: width, height = image.size
            x1, y1, x2, y2 = map(float, box)
            if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height: raise ValueError('invalid reference extent')
        except (ValueError, TypeError, SyntaxError, OSError):
            return {'supported': False, 'reason': 'invalid reference annotation coordinates'}
        target.append((name, [x1/width, y1/height, x2/width, y2/height]))
    if not target: return {'supported': False, 'reason': 'no supported DrawBox reference'}
    # Every saved overlay is from the original asset. Only the last output for
    # an asset is its submitted artifact, not a union of abandoned candidates.
    outputs = {}
    for event in events:
        result = event.get('output', {})
        if event.get('kind') != 'executed_tool' or result.get('artifact_type') != 'predicted_box_overlay': continue
        artifact = Path(result['artifact']); name = result['source_asset']
        if name not in assets: raise ValueError('predicted output uses unauthorized asset')
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != result['sha256']:
            raise ValueError('annotation artifact absent or changed')
        if hashlib.sha256(Path(assets[name]).read_bytes()).hexdigest() != result['source_file_sha256']:
            raise ValueError('annotation source changed')
        outputs[name] = result['boxes']
    predicted = [(name, row['bbox']) for name, boxes in outputs.items() for row in boxes]
    edges = sorted([(bbox_iou(a, b), i, j) for i, (source_a, a) in enumerate(predicted)
                    for j, (source_b, b) in enumerate(target) if source_a == source_b], reverse=True)
    used_p = set(); used_t = set(); total = 0.; hits = 0
    for overlap, i, j in edges:
        if i in used_p or j in used_t: continue
        used_p.add(i); used_t.add(j); total += overlap; hits += overlap >= .5
    return {'supported': True, 'reference_boxes': len(target), 'predicted_boxes': len(predicted),
        'real_artifact_submitted': bool(outputs),
        'greedy_bbox_iou_f1_dev': 2*total/(len(target)+len(predicted)),
        'bbox_f1_at_0_5_dev': 2*hits/(len(target)+len(predicted)),
        'official_image_judge_run': False}
