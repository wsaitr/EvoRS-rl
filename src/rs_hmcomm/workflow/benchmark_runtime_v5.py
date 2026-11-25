"""Executable adapted-tool baseline/MAS for ThinkGeo and Earth-Bench.

The original benchmark's annotated observations are scoring data only. This
environment does not claim an exact replication of its expert-model tool pool.
"""
import json
from .runtime import strict_json, WorkflowEpisode
from .tool_action_protocol import parse_tool_action
from .benchmark_tools_v5 import EvidenceBenchmarkToolbox as BenchmarkToolbox
import re
from .benchmark_submission import SUBMISSION_SCHEMA, submit_prediction
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


CATALOG = [*CATALOG,
    {'tool': 'annotate_boxes', 'asset': 'one authorized image', 'boxes': [{'bbox': ['x1', 'y1', 'x2', 'y2'], 'label': 'actual class'}]},
    {'tool': 'calibrated_raster_info', 'asset': 'one authorized raster'},
    {'tool': 'calibrated_raster_statistics', 'asset': 'one authorized raster', 'band': 1, 'calibration': 'metadata'},
    {'tool': 'raster_series_statistics', 'assets': ['authorized raster names'], 'band': 1, 'calibration': 'metadata'}]


class GroundedBenchmarkRuntime:
    def __init__(self, policy, *, multi=False, communication='text', max_calls=32, max_tools=16, output):
        self.policy = policy; self.multi = multi
        self.max_calls = max_calls; self.max_tools = max_tools; self.output = output
        if communication not in {'text', 'tokens', 'compressed'}: raise ValueError('unknown communication')
        self.communication = communication

    def run(self, task, training=False):
        self.policy.begin_episode(training=training)
        box = BenchmarkToolbox(task['assets'], self.policy, output=self.output)
        history = []; cost = CostVector(); answer = None; stop = 'model_budget_exhausted'
        observed = False
        requires_annotation = bool(re.search(r'\b(draw|mark|highlight|annotate)\b.{0,45}\bbox(?:es)?\b', task['question'], re.I))
        def call(role, prompt, *, choices=None, format='json', visual=False):
            nonlocal cost
            if len(self.policy.records) >= self.max_calls: raise ValueError('model call budget')
            before = cost.to_dict()
            packets = ()
            if visual and self.communication != 'text':
                original = tuple(p for p in self.policy.packets if p.startswith('benchmark-view-'))[-4:]
                if original:
                    packets = self.policy.disclose(original, recipient=role, question=task['question'], mode=self.communication)
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
        public = {'question': task['question'], 'assets': list(task['assets']), 'available_tools': CATALOG, 'requires_actual_asset_observation': bool(task['assets']),
                  'requires_real_box_overlay': requires_annotation, 'final_submission_schema': SUBMISSION_SCHEMA}
        if self.multi:
            plan = call('controller', 'Plan the required evidence and calculations, without answering. '+json.dumps(public), format='text')
            rules = call('knowledge', 'State relevant units, formulas or domain rules. Do not invent observations. '+json.dumps(public), format='text')
            history.extend([{'plan': plan}, {'domain_rules': rules}])
        while len(self.policy.records) < self.max_calls:
            prompt = 'Solve this task using ACTUAL tools. Return one tool JSON action with the documented fields. Never use reference observations. '+json.dumps({**public, 'history': history})
            try:
                action = parse_tool_action(call('controller' if self.multi else 'reasoning', prompt))
                if action['tool'] == 'answer':
                    if task['assets'] and not observed:
                        raise ValueError('final answer requires an actual image inspection or raster measurement; filenames and calculations alone are insufficient')
                    if action.get('annotations'):
                        if not isinstance(action['annotations'], list) or cost.tool_calls+len(action['annotations']) > self.max_tools:
                            raise ValueError('submission rendering exceeds tool budget')
                        _, outputs = submit_prediction({'answer': action['answer'], 'annotations': action['annotations']}, task['assets'], output=self.output)
                        box.events.extend(outputs); cost += CostVector(tool_calls=len(outputs))
                    if requires_annotation and not any(e.get('output', {}).get('artifact_type') == 'predicted_box_overlay' for e in box.events):
                        raise ValueError('requested box annotation requires an actual annotate_boxes output; a text claim is insufficient')
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
                if action['tool'] in {'inspect', 'detect'} and len(self.policy.records) >= self.max_calls:
                    stop = 'model_budget_exhausted'; break
                # Inspect/detect add genuine visual-expert records to the same trajectory.
                start = len(self.policy.records); before = cost.to_dict()
                cost += CostVector(tool_calls=1)
                try:
                    result = box.execute(action)
                    if action['tool'] in {'inspect', 'detect', 'raster_statistics', 'raster_expression', 'calibrated_raster_statistics', 'raster_series_statistics'}:
                        observed = True
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
                  'history': history, 'reference_observations_replayed': False, 'actual_asset_observed': observed,
                  'annotation_required': requires_annotation}]
        return WorkflowEpisode(task['sample_id'], answer, stop, None, events,
                               list(self.policy.records), cost, 0.)
