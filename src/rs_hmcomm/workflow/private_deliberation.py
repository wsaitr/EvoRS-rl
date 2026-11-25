"""Opt-in private reasoning before a bounded public candidate label.

Adapt the observation/reasoning/action separation used by pinned Qwen-Agent and
VisualSketchpad references; do not copy their tools or claim their performance.
The whole JSON completion remains in the original joint PPO and resource log.
"""
import hashlib
import json

from .feedback_revision_runtime import FeedbackRevisionRuntime
from .runtime import answer_instruction, answer_valid, question_choices, schema

VERSION = 'private-deliberation-v1'
INSTRUCTION = (
    'Resolve the assigned candidate step from the observations actually supplied. '
    'In analysis, briefly identify the decisive visible labels, locations or relationships; '
    'separate an observed fact from a physical or geographic inference. Compare the public '
    'alternatives against those facts, including negation in the question. For diagrams or '
    'plots use their visible legends, axes and symbols; displayed RGB color alone is not '
    'spectral calibration. For spatial or time comparisons keep the stated regions, dates, '
    'orientation and units. If revising a parent, address its supplied source review rather '
    'than repeat its conclusion. Missing detail remains uncertainty. Then put the selected '
    'public label in answer. Keep analysis concise to leave room for answer within the '
    'existing token budget. Return only the declared JSON object.'
)


def deliberation_schema(question, answer_format):
    if answer_format == 'choice':
        labels = [v.split('.',1)[0].strip() for v in question_choices(question)]
        if (not 2 <= len(labels) <= 26 or len(labels) != len(set(labels))
                or any(len(v) != 1 or not 'A' <= v <= 'Z' for v in labels)):
            raise ValueError('distinct public option labels required for private deliberation')
    elif answer_format == 'yes_no': labels = ['Yes','No']
    else: raise ValueError('private deliberation requires a bounded public answer space')
    return [['analysis','string'],['answer',{'enum':labels}]]


def protocol_descriptor():
    return {'version':VERSION,'instruction_sha256':hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        'upstream_commits':{'Qwen-Agent':'31a4d36d123688581a9e9744427272b33ce940e0',
                            'VisualSketchpad':'b03859858c63df4cea874e85e417e8b498a2f63b'},
        'all_generated_tokens_in_joint_PPO':True,'additional_model_calls':0,
        'reference_in_inputs':False,'private_analysis_not_workspace_evidence':True}


class PrivateDeliberationRuntime(FeedbackRevisionRuntime):
    def __init__(self,*args,private_deliberation=False,**kwargs):
        super().__init__(*args,**kwargs);self.private_deliberation=private_deliberation

    def _call(self,role,objective,packet_ids=(),context=None,**kwargs):
        if not (self.private_deliberation and role=='reasoning'
                and kwargs.get('output_format')=='text'
                and self.task.answer_format in {'choice','yes_no'}):
            return super()._call(role,objective,packet_ids,context,**kwargs)
        if (context or {}).get('legal_action_examples') or kwargs.get('json_field_schema') is not None:
            raise ValueError('deliberation cannot replace a different action contract')
        instruction=answer_instruction(self.task.answer_format)
        if instruction not in objective:
            raise ValueError('deliberation requires the original semantic candidate instruction')
        fields=deliberation_schema(self.task.question,self.task.answer_format)
        action=super()._call(role,objective.replace(instruction,'')+' '+INSTRUCTION,
            packet_ids,context,**{**kwargs,'output_format':'json','json_field_schema':fields})
        schema(action,{'analysis','answer'})
        if not isinstance(action['analysis'],str):raise ValueError('private analysis must be text')
        answer=answer_valid(action['answer'],self.task.answer_format)
        if action['answer'] not in fields[1][1]['enum']:
            raise ValueError('deliberation answer outside public alternatives')
        # Raw JSON, sampled masks, likelihoods, state, input/output costs and
        # auxiliary gradients remain in the same actual TokenDecision. Only the
        # semantic label returns to the existing candidate engine.
        index=len(self.policy.records)-1
        self.events.append({'kind':'private_deliberation_completion','protocol':VERSION,
            'decision_index':index,'role':role,'packet_ids':list(packet_ids),
            'analysis_sha256':hashlib.sha256(action['analysis'].encode()).hexdigest(),
            'analysis_UTF8_bytes':len(action['analysis'].encode()),'analysis_nonempty':bool(action['analysis'].strip()),
            'candidate_answer':answer,'all_completion_tokens_retained_for_joint_PPO':True,
            'analysis_exported_to_workspace':False,'new_visual_observation_claimed':False,
            'semantic_support_granted':False,'reference_used':False,'additional_model_calls':0})
        return answer

    def _initialize_task(self):
        self.events.append({'kind':'private_deliberation_protocol','enabled':self.private_deliberation,
            'descriptor':protocol_descriptor(),'reference_used':False,'additional_model_calls':0})
        return super()._initialize_task()
