"""Executable adapted-tool baseline/MAS for ThinkGeo and Earth-Bench.

The original benchmark's annotated observations are scoring data only. This
environment does not claim an exact replication of its expert-model tool pool.
"""
import json
from .runtime import strict_json, WorkflowEpisode
from .tool_action_protocol import parse_tool_action
from .benchmark_tools import BenchmarkToolbox
from .learned import state_vector
from rs_hmcomm.contracts import CostVector

CATALOG = [
    {'tool': 'list_assets'},
    {'tool': 'inspect', 'assets': ['task asset name'], 'question': 'what to inspect'},
    {'tool': 'detect', 'asset': 'one task asset name', 'question': 'target objects'},
    {'tool': 'calculate', 'expression': '(162*52+79*44)*0.5*0.5'},
    {'tool': 'raster_info', 'asset': 'task asset name'},
    {'tool': 'raster_statistics', 'asset': 'task asset name', 'bands': [1]},
    {'tool': 'raster_expression', 'assets': ['first.tif', 'second.tif'], 'bands': [1, 1], 'expression': '(a-b)/(a+b)'},
    {'tool': 'answer', 'answer': 'final task answer'}]


class BenchmarkAgentRuntime:
    toolbox_class = BenchmarkToolbox

    def create_toolbox(self, task):
        return self.toolbox_class(task['assets'], self.policy, output=self.output)

    def tool_result_cost(self, result):
        return CostVector()

    def tool_model_calls(self, action):
        return int(action.get('tool') in {'inspect', 'detect'})

    def public_task(self, task):
        return {'question': task['question'], 'assets': list(task['assets']), 'available_tools': CATALOG}

    def internal_visual_packets(self):
        return ()

    def routed_visual_packets(self, recipient):
        return tuple(p for p in self.policy.packets if p.startswith('benchmark-view-'))[-4:]

    def disclose_routed_packets(self, original, *, recipient, question):
        return self.policy.disclose(original, recipient=recipient, question=question, mode=self.communication)

    def answer_precondition(self, history):
        return True

    def action_context(self, public, history, cost):
        return {**public, 'history': history}

    def tool_action_choices(self, public, history, cost):
        return None

    def __init__(self, policy, *, multi=False, communication='text', max_calls=32, max_tools=16, output):
        self.policy = policy; self.multi = multi
        self.max_calls = max_calls; self.max_tools = max_tools; self.output = output
        if communication not in {'text', 'tokens', 'compressed'}: raise ValueError('unknown communication')
        self.communication = communication

    def run(self, task, training=False):
        self.policy.begin_episode(training=training)
        box = self.create_toolbox(task)
        history = []; cost = CostVector(); answer = None; stop = 'model_budget_exhausted'
        def call(role, prompt, *, choices=None, format='json', visual=False):
            nonlocal cost
            if len(self.policy.records) >= self.max_calls: raise ValueError('model call budget')
            before = cost.to_dict()
            packets = ()
            if not self.multi:
                packets = tuple(self.internal_visual_packets())
                if packets:
                    wire = sum(self.policy.packets[p].artifact.byte_size for p in packets)
                    cost += CostVector(message_latent_bytes=wire)
                    box.events.append({'kind': 'single_internal_visual_context',
                        'packet_ids': list(packets), 'bytes': wire,
                        'new_observation_created': False, 'reference_used': False})
            if visual and self.communication != 'text':
                original = self.routed_visual_packets(role)
                if original:
                    packets = self.disclose_routed_packets(original, recipient=role, question=task['question'])
                    wire = sum(self.policy.packets[p].artifact.byte_size for p in packets)
                    cost += CostVector(message_latent_bytes=wire)
                    box.events.append({'kind': 'benchmark_visual_route', 'recipient': role,
                                       'mode': self.communication, 'packets': list(packets), 'bytes': wire})
            result = self.policy.decide(role, prompt, packets, output_format=format, allowed_actions=choices)
            record = self.policy.records[-1]
            record.cost_before = before; record.potential_before = 0.
            record.state_features = state_vector(coverage=0., model_calls=cost.model_calls,
                tool_calls=cost.tool_calls, evidence_count=len(history), candidate_count=0,
                missing_count=0, role=role, invalid_actions=cost.invalid_actions, communication_bytes=0)
            record.old_value = float(self.policy.value(record.state_features)) if self.policy.auxiliary is not None else 0.
            cost += CostVector(model_calls=1, agent_calls=1, model_input_tokens=record.input_ids.numel(),
                               model_output_tokens=record.output_ids.numel(), billed_compute_ms=record.latency_ms)
            return result
        public = self.public_task(task)
        if self.multi:
            plan = call('controller', 'Plan the required evidence and calculations, without answering. '+json.dumps(public), format='text')
            rules = call('knowledge', 'State relevant units, formulas or domain rules. Do not invent observations. '+json.dumps(public), format='text')
            history.extend([{'plan': plan}, {'domain_rules': rules}])
        while len(self.policy.records) < self.max_calls:
            prompt = 'Solve this task using ACTUAL tools. Return one tool JSON action with the documented fields. Never use reference observations. '+json.dumps(self.action_context(public, history, cost))
            try:
                choices = self.tool_action_choices(public, history, cost)
                if choices == []:
                    stop = 'action_budget_exhausted'; break
                action = parse_tool_action(call('controller' if self.multi else 'reasoning', prompt, choices=choices))
                if action['tool'] == 'answer':
                    if not self.answer_precondition(history):
                        raise ValueError('Answer requires actual visual or native numerical observation first. Use inspect, detect or a native raster tool; listing assets or a bare calculation is not an observation.')
                    candidate = action['answer']
                    if not isinstance(candidate, str) or not candidate.strip(): raise ValueError('empty answer')
                    if self.multi and len(self.policy.records)+2 <= self.max_calls:
                        candidate = call('reasoning', 'Produce the final answer to the original task from the actual executed evidence and calculations. Check units and dates; the proposed answer may be wrong. '+json.dumps({'task': public, 'history': history, 'proposal': candidate}), format='text', visual=True).strip()
                        review = strict_json(call('verification', 'Verify the answer against only the executed tool results. Accept a negative answer if supported. '+json.dumps({'task': public, 'history': history, 'candidate': candidate}), choices=[{'decision': 'accept'}, {'decision': 'reject'}], visual=True))
                        if review['decision'] == 'reject':
                            history.append({'rejected_candidate': candidate}); continue
                    elif self.multi:
                        stop = 'verification_budget_exhausted'; break
                    answer = candidate; stop = 'finished'; break
                if cost.tool_calls >= self.max_tools: stop = 'tool_budget_exhausted'; break
                if len(self.policy.records)+self.tool_model_calls(action) > self.max_calls:
                    stop = 'model_budget_exhausted'; break
                # Inspect/detect add genuine visual-expert records to the same trajectory.
                start = len(self.policy.records); before = cost.to_dict()
                cost += CostVector(tool_calls=1)
                try:
                    result = box.execute(action)
                    cost += self.tool_result_cost(result)
                finally:
                    for i, record in enumerate(self.policy.records[start:]):
                        record.cost_before = before if i == 0 else cost.to_dict(); record.potential_before = 0.
                        record.state_features = state_vector(coverage=0., model_calls=cost.model_calls,
                            tool_calls=cost.tool_calls, evidence_count=len(history), candidate_count=0,
                            missing_count=0, role='visual', invalid_actions=cost.invalid_actions, communication_bytes=0)
                        record.old_value = float(self.policy.value(record.state_features)) if self.policy.auxiliary is not None else 0.
                        cost += CostVector(model_calls=1, agent_calls=1, model_input_tokens=record.input_ids.numel(),
                                           model_output_tokens=record.output_ids.numel(), billed_compute_ms=record.latency_ms)
                history.append({'action': action, 'result': result})
            except (ValueError, TypeError, KeyError, ZeroDivisionError, OSError) as error:
                cost += CostVector(invalid_actions=1, errors=1)
                history.append({'tool_error': str(error)})
                if len(self.policy.records) >= self.max_calls: break
        events = [*box.events, {'kind': 'adapted_benchmark_context', 'public_task': public,
                  'history': history, 'reference_observations_replayed': False}]
        return WorkflowEpisode(task['sample_id'], answer, stop, None, events,
                               list(self.policy.records), cost, 0.)
