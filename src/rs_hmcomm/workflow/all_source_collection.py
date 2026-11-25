"""Preserve every observed authorized collection source in its visual route."""
from pathlib import Path
from .scene_collection_expert import SceneCollectionRuntime


def latest_authorized_views(packets, assets, recipient):
    """One latest full native view per actual source; no label-based selection."""
    sources = list(dict.fromkeys(str(Path(p).resolve()) for p in assets.values()))
    latest = {}
    for pid, packet in packets.items():
        if not pid.startswith('benchmark-view-') or packet.recipe is not None:
            continue
        source = str(Path(packet.artifact.source_image_ref).resolve())
        if source not in sources or recipient not in packet.artifact.compatible_consumers:
            continue
        if packet.artifact.roi is not None and tuple(packet.artifact.roi) != (0, 0, 1, 1):
            continue
        latest[source] = pid
    return tuple(latest[s] for s in sources if s in latest), tuple(s for s in sources if s not in latest)


class AllSourceSceneCollectionRuntime(SceneCollectionRuntime):
    def __init__(self, *args, slot_allocation='total', **kwargs):
        if slot_allocation not in {'total', 'per_source'}:
            raise ValueError('unknown collection slot allocation')
        super().__init__(*args, **kwargs)
        self.slot_allocation = slot_allocation

    def public_task(self, task):
        self.collection_assets = dict(task['assets'])
        return super().public_task(task)

    def internal_visual_packets(self):
        # A single Agent also needs its own observed pixels at later decisions.
        # Reuse the original native features: no new observation, compression,
        # hidden classifier call, or source selected using an answer label.
        selected, missing = latest_authorized_views(
            self.policy.packets, self.collection_assets, 'reasoning')
        self.route_events.append({'kind': 'single_all_source_collection_context',
            'selected_native_packets': list(selected),
            'not_encoded_or_unauthorized_sources': list(missing),
            'complete_source_set': not missing, 'reference_used': False,
            'new_observation_created': False})
        return selected

    def routed_visual_packets(self, recipient):
        selected, missing = latest_authorized_views(self.policy.packets, self.collection_assets, recipient)
        if (self.communication == 'compressed' and getattr(self, 'slot_allocation', 'total') == 'total'
                and len(selected) > self.policy.compressed_tokens):
            raise ValueError('all-source route cannot fit one compressed slot per source; no sources silently dropped')
        self.route_events.append({'kind': 'all_source_collection_route', 'recipient': recipient,
            'selected_native_packets': list(selected), 'not_encoded_or_unauthorized_sources': list(missing),
            'complete_source_set': not missing, 'same_source_repeats_removed': True,
            'reference_used': False, 'ground_truth_claim': False})
        return selected

    def disclose_routed_packets(self, original, *, recipient, question):
        if self.slot_allocation == 'per_source' and self.communication == 'compressed':
            # Reuse the same learned single-source query table. Actual transmitted
            # bytes scale with source count and remain billed by the caller.
            return tuple(pid for source in original for pid in self.policy.disclose(
                (source,), recipient=recipient, question=question, mode='compressed'))
        return super().disclose_routed_packets(original, recipient=recipient, question=question)

    def run(self, task, training=False):
        self.route_events = []
        episode = super().run(task, training=training)
        episode.events.extend(self.route_events)
        return episode
