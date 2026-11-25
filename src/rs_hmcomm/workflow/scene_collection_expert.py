"""Frozen remote-sensing scene observations; aggregation stays with the Agent."""
import re
import time
from PIL import Image
from rs_hmcomm.contracts import CostVector
from .collection_observation import CollectionToolbox, CollectionRuntime
from .remoteclip_expert import validate_labels


def natural_scene_name(name):
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return name.replace('_', ' ').strip().casefold()


def public_scene_classes(question):
    match = re.search(r'belongs to\s*\{([^{}]+)\}', question)
    if not match: return None
    names = [item.strip() for item in match.group(1).split(',')]
    validate_labels(names)
    return names


class SceneCollectionToolbox(CollectionToolbox):
    def __init__(self, *args, classifier, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = classifier

    def execute(self, action):
        if action.get('tool') != 'classify_scene_collection':
            return super().execute(action)
        if set(action) != {'tool', 'assets', 'labels'}:
            raise ValueError('scene classification requires exactly tool, assets and labels')
        names = action['assets']; labels = action['labels']
        if (not isinstance(names, list) or not 1 <= len(names) <= 32
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('one to 32 distinct actual asset names required')
        validate_labels(labels)
        descriptions = [natural_scene_name(label) for label in labels]
        validate_labels(descriptions)
        for name in names: self.path(name)
        rows = []; elapsed = 0.
        for name in names:
            started = time.perf_counter()
            packet_id = None
            with Image.open(self.path(name)) as image:
                if image.mode not in {'RGB', 'RGBA', 'L', 'P'}:
                    raise ValueError('scene classification requires ordinary images, not scientific raster values')
                image = image.convert('RGB'); size = list(image.size)
                result = self.classifier.classify(image, descriptions)
                if self.policy is not None:
                    # Use the receiver's OWN encoder, never CLIP vectors in a Qwen latent space.
                    packet_id = f'benchmark-view-{len(self.policy.packets)}'
                    self.policy.encode(image, artifact_id=packet_id, source_image_ref=str(self.path(name)),
                        roi=(0, 0, 1, 1), label=name)
            elapsed += (time.perf_counter()-started)*1000
            predicted = labels[descriptions.index(result['highest_similarity_label'])]
            row = {'asset': name, 'original_image_size': size, 'requested_labels': labels,
                   'predicted_scene_class': predicted, 'raw_classification': result,
                   'native_packet_id': packet_id,
                   'scores_are_calibrated_probabilities': False}
            rows.append(row)
            self.events.append({'kind': 'scene_collection_observation', **row,
                                'reference_used': False, 'ground_truth_claim': False})
        result = {'observations': rows, 'observed_assets': names,
                  'frozen_auxiliary_inferences': len(rows), 'latency_ms': elapsed,
                  'reference_used': False, 'ground_truth_claim': False,
                  'final_answer_generated_by_tool': False}
        self.events.append({'kind': 'executed_tool', 'action': action, 'output': result,
            'origin': 'actual frozen scene classifier predictions; no reference observations'})
        return result


class SceneCollectionRuntime(CollectionRuntime):
    def __init__(self, *args, classifier, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = classifier

    def create_toolbox(self, task):
        return SceneCollectionToolbox(task['assets'], self.policy, output=self.output,
                                      classifier=self.classifier)

    def public_task(self, task):
        public = super().public_task(task)
        tool = {'tool': 'classify_scene_collection', 'assets': list(task['assets']),
            'required_fields': {'labels': 'two to 64 competing scene classes from the public question or domain'},
            'output': 'separate predicted class and uncalibrated similarity per image; Agent must aggregate and check'}
        labels = public_scene_classes(task['question'])
        if labels: tool['labels'] = labels
        return {**public, 'available_tools': [*public['available_tools'], tool],
                'auxiliary_scene_classifier': self.classifier.identity}

    def action_context(self, public, history, cost):
        context = super().action_context(public, history, cost)
        observed = set(context['collection_observation_state']['actually_observed_assets'])
        for item in history:
            if item.get('action', {}).get('tool') == 'classify_scene_collection' and 'result' in item:
                observed.update(item['result']['observed_assets'])
        state = dict(context['collection_observation_state'])
        state.update(actually_observed_assets=sorted(observed),
                     not_yet_observed_assets=[name for name in public['assets'] if name not in observed])
        state['inspect_all_remaining_model_calls_including_action'] = 1+len(state['not_yet_observed_assets'])
        return {**context, 'collection_observation_state': state}

    def tool_result_cost(self, result):
        return CostVector(billed_compute_ms=result.get('latency_ms', 0.)
                          if result.get('frozen_auxiliary_inferences', 0) > 0 else 0.)
