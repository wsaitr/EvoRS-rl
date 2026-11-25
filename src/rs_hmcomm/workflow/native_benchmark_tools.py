"""Explicit pixel detection on the actual processor canvas; no field guessing."""
import json
import math
from .benchmark_tools import BenchmarkToolbox
from .benchmark_runtime import BenchmarkAgentRuntime
from .runtime import strict_json, schema
from .object_labels import usable_object_label


class NativeDetectionToolbox(BenchmarkToolbox):
    def canvas(self, packets):
        if len(packets) != 1:
            raise ValueError('one actual detection canvas required')
        packet = self.policy.packets[packets[0]]
        if packet.recipe is not None:
            raise ValueError('detection requires original native view')
        temporal, height, width = packet.grid
        patch = self.policy.base.config.vision_config.patch_size
        if temporal != 1 or type(patch) != int or patch <= 0:
            raise ValueError('unsupported detection grid')
        return width*patch, height*patch

    def detection_prompt(self, question, packets, sizes):
        width, height = self.canvas(packets)
        return question+'\n'+json.dumps({
            'action': 'localize question-relevant visible objects',
            'actual_attached_pixel_canvas': {'width': width, 'height': height},
            'response_schema': {'objects': 'list of zero to 32 objects; each has exactly label and bbox_pixels'},
            'label_rule': 'actual observed class name; no placeholder such as category, name or target objects',
            'bbox_pixels_rule': 'four finite numbers x1,y1,x2,y2; 0<=x1<x2<=canvas width and 0<=y1<y2<=canvas height',
            'uncertain_localization': 'return an empty objects list',
            'measurement_rule': 'the tool maps this canvas to original image pixels; do not guess original coordinates or normalized coordinates'})

    def parse_detections(self, raw, packets, sizes):
        width, height = self.canvas(packets)
        if len(sizes) != 1:
            raise ValueError('one original size required')
        original_width, original_height = sizes[0]
        value = strict_json(raw); schema(value, {'objects'})
        rows = value['objects']
        if not isinstance(rows, list) or len(rows) > 32:
            raise ValueError('invalid detections')
        objects = []
        for row in rows:
            schema(row, {'label', 'bbox_pixels'})
            label, bbox = row['label'], row['bbox_pixels']
            if (not usable_object_label(label)):
                raise ValueError('ungrounded detection label')
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or any(type(v) not in {int, float} or not math.isfinite(v) for v in bbox)):
                raise ValueError('invalid detection pixel box')
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('detection outside actual canvas')
            normalized = [x1/width, y1/height, x2/width, y2/height]
            original = [normalized[0]*original_width, normalized[1]*original_height,
                        normalized[2]*original_width, normalized[3]*original_height]
            objects.append({'label': label.strip(), 'bbox': normalized,
                            'original_bbox_pixels': original, 'canvas_bbox_pixels': bbox})
        return {'objects': objects, 'original_image_sizes': sizes,
                'actual_pixel_canvas': [width, height],
                'coordinate_transform': 'full-image resize; separate measured x/y scale',
                'reference_used': False, 'normalization_guessed': False}


class NativeDetectionRuntime(BenchmarkAgentRuntime):
    toolbox_class = NativeDetectionToolbox


class TaskBoundNativeRuntime(NativeDetectionRuntime):
    def public_task(self, task):
        """Use actual authorized assets/questions instead of literal example values."""
        names = list(task['assets']); question = task['question']
        public = super().public_task(task)
        tools = []
        for old in public['available_tools']:
            action = dict(old)
            if 'asset' in action and names:
                action['asset'] = names[0]
            if 'assets' in action:
                action['assets'] = names[:4] if action['tool'] == 'inspect' else names[:2]
            if 'question' in action:
                action['question'] = question
            if action['tool'] == 'calculate':
                # Schema text, not arbitrary numerical values copied as evidence.
                action = {'tool': 'calculate', 'required_fields': {'expression':
                    'arithmetic string over numerical values from actual earlier tool results'}}
            tools.append(action)
        return {**public, 'available_tools': tools}


class ObservingBenchmarkSingle(TaskBoundNativeRuntime):
    """The same tool-using Agent retains its own actual observed image context."""
    def __init__(self, *args, max_visual_tokens=1024, **kwargs):
        super().__init__(*args, **kwargs)
        if self.multi:
            raise ValueError('observing single-agent protocol cannot masquerade as MAS')
        if type(max_visual_tokens) != int or max_visual_tokens <= 0:
            raise ValueError('positive visual token cap required')
        self.max_visual_tokens = max_visual_tokens

    def internal_visual_packets(self):
        selected = []; tokens = 0
        for pid, packet in reversed(tuple(self.policy.packets.items())):
            if (not pid.startswith('benchmark-view-') or packet.recipe is not None
                    or 'reasoning' not in packet.artifact.compatible_consumers):
                continue
            count = packet.artifact.shape[0]
            if tokens+count <= self.max_visual_tokens:
                selected.append(pid); tokens += count
            if len(selected) == 4:
                break
        return tuple(reversed(selected))

    def answer_precondition(self, history):
        if self.internal_visual_packets():
            return True
        return any(item.get('action', {}).get('tool') in {
            'raster_info', 'raster_statistics', 'raster_expression'} and 'result' in item
            for item in history)

    def public_task(self, task):
        return {**super().public_task(task),
            'observation_rule': 'Before answering, execute an actual visual or native numerical observation. Previously inspected actual images are retained for your own later decisions under the declared token cap.',
            'max_retained_visual_tokens': self.max_visual_tokens}
