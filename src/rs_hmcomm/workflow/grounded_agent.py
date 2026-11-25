"""Versioned role prompts and controller operation memory, with unchanged actions.

The memory contains execution metadata only. It grants no visual authorization,
semantic support, reward or new tool. Every model call still enters PPO replay.
"""
import json
from pathlib import Path

from .agent_prompts import VERSION, protocol_descriptor, system_instruction
from .independent_fitness import IndependentFitnessRuntime
from .quality import QualityWorkflowPolicy
from .qwen_policy import QwenWorkflowPolicy


class GroundedWorkflowPolicy(QualityWorkflowPolicy):
    prompt_profile = 'legacy'

    def set_prompt_profile(self, profile):
        if profile not in {'legacy', VERSION}:
            raise ValueError('unknown prompt profile')
        self.prompt_profile = profile

    def _system_instruction(self, role, *, output_format='json'):
        if self.prompt_profile == VERSION:
            return system_instruction(role, output_format)
        return super()._system_instruction(role, output_format=output_format)

    def prepare(self, role, prompt, packet_ids, *, output_format='json'):
        if self.prompt_profile == VERSION:
            # Responsibilities are now in the system message. Retain the original
            # task JSON envelope and native-image serialization without duplication.
            return QwenWorkflowPolicy.prepare(self, role, prompt, packet_ids, output_format=output_format)
        return super().prepare(role, prompt, packet_ids, output_format=output_format)

    def save(self, path):
        super().save(path)
        (Path(path)/'prompt_protocol.json').write_text(json.dumps(self.prompt_protocol(), indent=2)+'\n')

    def prompt_protocol(self):
        return {'profile': self.prompt_profile,
                'descriptor': protocol_descriptor() if self.prompt_profile == VERSION else None}

    def load(self, path, *, resume_training=False):
        meta = Path(path)/'prompt_protocol.json'
        saved = json.loads(meta.read_text()) if meta.exists() else {'profile': 'legacy', 'descriptor': None}
        if saved != self.prompt_protocol():
            raise ValueError('checkpoint prompt protocol mismatch; initialize from legacy before explicit transfer')
        return super().load(path, resume_training=resume_training)


def operation_memory(events, feedback, limit=6):
    """Bounded factual receipts; never copy another expert's private drafts."""
    rows = []
    for event in events:
        if event['kind'] == 'tool' and isinstance(event.get('operation'), dict):
            rows.append({'kind': 'observation_executed', 'operation': event['operation'],
                         'output_packet_ids': event.get('outputs', []),
                         'semantic_correctness_verified': False})
        elif event['kind'] == 'invalid':
            rows.append({'kind': 'operation_rejected', 'reason': event.get('reason', '')[:240],
                         'decision_index': event.get('decision_index'),
                         'new_evidence_from_rejection': False})
    return {'recent_operations': rows[-limit:], 'last_error': feedback[:240],
            'interpretations_are_verified_facts': False,
            'recovery_instruction': 'Use the current legal actions to repair the concrete error or change the observation/hypothesis; do not repeat an unchanged failed request.'}


class GroundedAgentRuntime(IndependentFitnessRuntime):
    def __init__(self, *args, operation_ledger=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation_ledger = operation_ledger

    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        # Only the controller gets public workflow receipts. Experts receive
        # their original routed context; blind fitness gets no candidate leak.
        if self.operation_ledger and role == 'controller':
            context = dict(context or {})
            context['operation_ledger'] = operation_memory(self.events, self.feedback)
            context['progress_state'] = {
                'public_evidence_records': len(self.state.evidence),
                'public_candidate_records': len(self.state.answer_candidates),
                'model_calls_used': self._model_calls_used(),
                'tool_calls_used': self.cost.tool_calls,
                'missing_requirements': self.ledger.missing() if getattr(self, 'ledger', None) else [],
                'semantic_support_is_model_estimate': True,
            }
        if (self.policy.prompt_profile == VERSION and role == 'visual' and (context or {}).get('legal_action_examples')
                and any('op' in a and 'bbox' in a for a in context['legal_action_examples'])):
            context = dict(context)
            context['observation_tool_contract'] = {
                'coordinates': 'normalized [x1,y1,x2,y2], top-left origin, x right, y down; not [x,y,width,height]',
                'crop': 'one region of one frame, one tool call; output is an actual new visual packet',
                'grid': 'bbox split into 2x2, four actual crops and four tool calls',
                'paired_crop': 'same reference ROI mapped to dates using supplied registration; one crop per frame',
                'preconditions': 'only currently listed legal actions are executable; actual frame and registration metadata govern availability',
                'measurement_limits': 'no implicit ground scale, north direction, spectral bands or elevation',
            }
        return super()._call(role, objective, packet_ids, context, **kwargs)

    def _initialize_task(self):
        self.events.append({'kind': 'grounded_agent_protocol',
            'prompt_protocol': self.policy.prompt_protocol(), 'operation_ledger': self.operation_ledger,
            'semantic_support_granted': False, 'reference_used': False,
            'action_space_changed': False, 'extra_model_calls': 0})
        return super()._initialize_task()
