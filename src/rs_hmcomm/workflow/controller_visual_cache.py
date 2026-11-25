"""Reuse a controller's already delivered planning view for candidate decisions.

This does not send an image to a new recipient, encode another view, grant
semantic support, add a model call or force a terminal action. Actual decoder
input tokens and the original disclosure remain in the ordinary cost ledger.
"""
import hashlib
import json

from .public_search_feedback import PublicSearchFeedbackRuntime


class ControllerVisualCacheRuntime(PublicSearchFeedbackRuntime):
    def __init__(self, *args, controller_visual_cache=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller_visual_cache = controller_visual_cache

    def _cached_candidate_view(self):
        packets = tuple(getattr(self, 'planning_packets', ()))
        if not packets:
            return None
        if len(set(packets)) != len(packets):
            raise ValueError('duplicate controller cached packets')
        disclosure = next((e for e in reversed(self.events)
                           if e['kind'] == 'requirement_planning_disclosure'), None)
        if (disclosure is None or tuple(disclosure['packet_ids']) != packets
                or disclosure['mode'] not in {'tokens', 'compressed'}
                or disclosure['mode'] not in self.mechanisms.modes
                or disclosure['source_evidence_ids'] != ['overview']
                or disclosure['support_granted']):
            raise PermissionError('controller cache lacks an actual planning disclosure')
        delivery = next((e for e in reversed(self.events)
                         if e['kind'] == 'route' and e['role'] == 'controller'
                         and tuple(e['packet_ids']) == packets
                         and e['mode'] == disclosure['mode']), None)
        if delivery is None or delivery['total_wire_bytes'] <= 0:
            raise PermissionError('controller cache lacks its billed recipient delivery')
        if any(p not in self.policy.packets or
               'controller' not in self.policy.packets[p].artifact.compatible_consumers
               for p in packets):
            raise PermissionError('cached packet not authorized for controller')
        if sum(self.policy.packets[p].artifact.shape[0] for p in packets) > self.max_visual_tokens:
            raise ValueError('cached controller view exceeds token cap')
        return packets, disclosure, delivery

    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        legal = (context or {}).get('legal_action_examples', ())
        # Pointer, tool, private-search, fitness and requirement actions retain
        # their original inputs. This hook applies only to choosing a role or
        # source-linked candidate when a finish action is actually offered.
        if (not self.controller_visual_cache or role != 'controller' or packet_ids
                or not any(a.get('role') == 'finish' for a in legal)):
            return super()._call(role, objective, packet_ids, context, **kwargs)
        cached = self._cached_candidate_view()
        if cached is None:
            return super()._call(role, objective, packet_ids, context, **kwargs)
        packets, disclosure, delivery = cached
        context = dict(context or {})
        context['cached_controller_visual_context'] = {
            'packet_ids': list(packets), 'mode': disclosure['mode'],
            'source_evidence_ids': list(disclosure['source_evidence_ids']),
            'previously_delivered_to_this_recipient': True,
            'new_observation_or_support_claimed': False,
            'interpretation_verified': False,
        }
        objective += (' Compare source-linked candidates against the attached visual context '
                      'already delivered to you during requirement planning. Read visible labels, '
                      'regions and option meanings rather than treating earlier descriptions as '
                      'certified facts. The cached view is the same observation, not another vote. '
                      'Select exactly one currently offered action; attached pixels do not force finish.')
        event = {'kind': 'controller_visual_cache_read', 'mode': disclosure['mode'],
            'packet_ids': list(packets), 'source_evidence_ids': ['overview'],
            'original_delivery_sha256': hashlib.sha256(json.dumps(delivery, sort_keys=True).encode()).hexdigest(),
            'extra_model_calls': 0, 'new_image_encodings': 0, 'additional_wire_bytes': 0,
            'semantic_support_granted': False, 'reference_used': False,
            'action_space_changed': False, 'completed_policy_decision': False,
            'decision_index': None}
        self.events.append(event)
        before = len(self.policy.records)
        try:
            return super()._call(role, objective, packets, context, **kwargs)
        finally:
            if len(self.policy.records) > before:
                event['completed_policy_decision'] = True
                event['decision_index'] = len(self.policy.records)-1

    def _initialize_task(self):
        self.events.append({'kind': 'controller_visual_cache_protocol',
            'enabled': self.controller_visual_cache, 'new_image_encodings': 0,
            'extra_model_calls': 0, 'semantic_support_granted': False,
            'action_space_changed': False, 'reference_used': False})
        return super()._initialize_task()
