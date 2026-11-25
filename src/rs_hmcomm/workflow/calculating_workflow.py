"""Sampled knowledge actions execute bounded arithmetic in the full workflow."""
import json
import math
import time
from rs_hmcomm.contracts import CostVector, Evidence
from .adaptive_closing_budget import AdaptiveClosingBudgetRuntime
from .benchmark_tools import calculate
from .runtime import schema, refs


class CalculatingWorkflowRuntime(AdaptiveClosingBudgetRuntime):
    def _expert(self, role, context, packets, selected):
        if role != 'knowledge':
            return super()._expert(role, context, packets, selected)
        if not selected:
            raise ValueError('knowledge requires selected evidence')
        remaining = max(0, self.max_tool_calls-self.cost.tool_calls)
        # One actual sampled role turn replaces the previous guidance turn.
        # No extra inference is hidden in this deterministic arithmetic tool.
        action = self._call('knowledge',
            'Give relevant interpretation guidance and, when needed, actual arithmetic. '
            'Return JSON with exactly guidance (string) and calculations (list). '
            'Each calculation has exactly expression (arithmetic string), '
            'source_evidence_ids (selected supporting IDs), and output_unit (string). '
            'Use numbers from the question or selected evidence; do not invent measurements. '
            'Supported operations: + - * / ** %, sqrt, abs, min, max, round. '
            'Calculations require actual numeric operands; put symbolic derivations in guidance. '
            'GIS operations such as BUFFER are not arithmetic requests. Do not invent numbers to fit this interface. '
            'Use no more than four calculations and no more than the remaining tool calls. '
            'Use an empty list when arithmetic is unnecessary or inputs are missing.',
            packets, {**context, 'arithmetic_tool_remaining_calls': remaining},
            json_field_schema=[['guidance','string'],['calculations',{'array':{'object':[
                ['expression','numeric_expression'],['source_evidence_ids',{'array':{'enum':list(selected)},'minimum':1,'maximum':len(selected)}],
                ['output_unit','string']]},'maximum':min(4,remaining)}]])
        schema(action, {'guidance', 'calculations'})
        guidance, rows = action['guidance'], action['calculations']
        if not isinstance(guidance, str) or not guidance.strip() or len(guidance) > 2048:
            raise ValueError('nonempty bounded guidance required')
        if not isinstance(rows, list) or len(rows) > min(4, remaining):
            raise ValueError('arithmetic tool budget exceeded')
        validated = []
        for row in rows:
            schema(row, {'expression', 'source_evidence_ids', 'output_unit'})
            expression, unit = row['expression'], row['output_unit']
            if not isinstance(expression, str) or not 1 <= len(expression) <= 256:
                raise ValueError('bounded arithmetic string required')
            if not isinstance(unit, str) or len(unit) > 80:
                raise ValueError('bounded output unit required')
            sources = refs(row['source_evidence_ids'], selected)
            if not sources:
                raise ValueError('arithmetic needs selected support evidence')
            validated.append((expression, unit, sources))
        results = []
        for expression, unit, sources in validated:
            self.cost += CostVector(tool_calls=1)
            started = time.perf_counter()
            event = {'kind': 'workflow_arithmetic', 'expression': expression,
                     'source_evidence_ids': list(sources), 'output_unit': unit,
                     'input_measurements_verified': False, 'output_unit_verified': False,
                     'reference_used': False, 'decision_index': len(self.policy.records)-1}
            try:
                value = calculate(expression, rounding=True)
                if type(value) not in {int, float} or not math.isfinite(value):
                    raise ValueError('arithmetic result must be a finite real scalar')
                event.update(success=True, result=value)
                results.append({**event})
            except Exception as error:
                event.update(success=False, error=str(error))
                raise
            finally:
                elapsed = (time.perf_counter()-started)*1000
                self.cost += CostVector(billed_compute_ms=elapsed)
                event['latency_ms'] = elapsed
                self.events.append(event)
        summary = guidance if not results else json.dumps({'guidance': guidance, 'arithmetic': results})
        self._commit('knowledge', evidence=(Evidence(f'guidance-{self.step}', 'workflow_guidance',
            summary, 'knowledge', self.step,
            payload={'source_evidence_ids': list(selected), 'arithmetic_results': results,
                     'input_measurements_verified': False, 'output_unit_verified': False}),))
