"""A common prediction-to-artifact interface; no labels or expert decisions."""
from PIL import Image
from .benchmark_tools_v5 import EvidenceBenchmarkToolbox
from .evidence import _normalized_bbox
from .runtime import WorkflowEpisode, strict_json
from rs_hmcomm.contracts import CostVector

SUBMISSION_SCHEMA = {
    'answer': 'final answer',
    'annotations': [{'asset': 'authorized asset name',
                     'boxes': [{'bbox': ['x1', 'y1', 'x2', 'y2'], 'label': 'predicted category'}]}]}


def submit_prediction(prediction, assets, *, output):
    """Render only model-predicted coordinates, identically for every method."""
    if not isinstance(prediction, dict) or set(prediction)-{'answer', 'annotations'}:
        raise ValueError('invalid submission fields')
    answer = prediction.get('answer')
    if not isinstance(answer, str) or not answer.strip(): raise ValueError('empty submission answer')
    annotations = prediction.get('annotations', [])
    if not isinstance(annotations, list) or len(annotations) > 16: raise ValueError('invalid annotations')
    box = EvidenceBenchmarkToolbox(assets, None, output=output)
    seen = set()
    # Validate the entire submission before producing any artifact.
    for item in annotations:
        if not isinstance(item, dict) or set(item) != {'asset', 'boxes'}: raise ValueError('invalid annotation fields')
        name = item['asset']
        if name in seen: raise ValueError('duplicate annotation asset')
        seen.add(name)
        with Image.open(box.path(name)) as image:
            if image.mode not in {'RGB', 'RGBA', 'L', 'P'}: raise ValueError('annotation requires display imagery')
        rows = item['boxes']
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32: raise ValueError('invalid predicted boxes')
        for row in rows:
            if not isinstance(row, dict) or set(row)-{'bbox', 'label'} or 'bbox' not in row:
                raise ValueError('invalid predicted box fields')
            _normalized_bbox(row['bbox'])
            if not isinstance(row.get('label', ''), str) or len(row.get('label', '')) > 160:
                raise ValueError('invalid annotation label')
    for item in annotations:
        box.execute({'tool': 'annotate_boxes', **item})
    return answer.strip(), box.events


class PureArtifactRunner:
    """One model call with actual images and the common output format."""
    def __init__(self, policy, *, output, max_visual_tokens=1024):
        self.policy = policy; self.output = output; self.max_visual_tokens = max_visual_tokens

    def run(self, task, training=False):
        import json
        if training: raise ValueError('pure baseline does not train')
        self.policy.begin_episode(training=False); self.policy.payload_intervention = 'none'
        events = []; cost = CostVector(); answer = None; stop = 'invalid_submission'
        try:
            packets = []
            for name, filename in task['assets'].items():
                with Image.open(filename) as image:
                    if image.mode not in {'RGB', 'RGBA', 'L', 'P'}:
                        raise ValueError('native scientific pure input adapter pending')
                    pid = 'benchmark-view-'+str(len(packets))
                    self.policy.encode(image.convert('RGB'), artifact_id=pid,
                        source_image_ref=filename, roi=(0, 0, 1, 1), label=name)
                    packets.append(pid)
            tokens = sum(self.policy.packets[p].artifact.shape[0] for p in packets)
            if tokens > self.max_visual_tokens: raise ValueError('declared pure visual budget exceeded')
            prompt = 'Solve the task from the attached actual images. Return one JSON submission. '
            prompt += 'Annotations use normalized ORIGINAL image coordinates and only predicted visible objects. '
            prompt += json.dumps({'question': task['question'], 'assets': list(task['assets']),
                                 'submission_schema': SUBMISSION_SCHEMA})
            response = self.policy.decide('reasoning', prompt, tuple(packets), output_format='json')
            record = self.policy.records[-1]
            cost += CostVector(model_calls=1, model_input_tokens=record.input_ids.numel(),
                model_output_tokens=record.output_ids.numel(), billed_compute_ms=record.latency_ms)
            events.append({'kind': 'pure_submission_prediction', 'response': response,
                           'packet_ids': packets, 'visual_tokens': tokens})
            answer, rendered = submit_prediction(strict_json(response), task['assets'], output=self.output)
            events.extend(rendered); cost += CostVector(tool_calls=len(rendered))
            stop = 'finished'
        except (ValueError, TypeError, KeyError, OSError) as error:
            cost += CostVector(errors=1, invalid_actions=1)
            events.append({'kind': 'submission_error', 'reason': str(error)})
        return WorkflowEpisode(task['sample_id'], answer, stop, None, events,
                               list(self.policy.records), cost, 0.)
