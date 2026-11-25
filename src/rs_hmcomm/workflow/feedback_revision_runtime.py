"""Optional, billed private parent reviews for the existing variation turns."""
from rs_hmcomm.contracts import CostVector

from .controller_visual_cache import ControllerVisualCacheRuntime
from .private_revision_feedback import parent_source_feedback, feedback_wire_bytes
from .runtime import question_choices


class FeedbackRevisionRuntime(ControllerVisualCacheRuntime):
    # Like the routing overhead bound, this bounds an actual serialized receipt.
    # Both variation turns are charged; logical bytes are not network capture.
    feedback_stage_byte_cap = 65536

    def __init__(self,*args,private_revision_feedback=False,**kwargs):
        super().__init__(*args,**kwargs)
        if private_revision_feedback and not self.independent_fitness:
            raise ValueError('private revision feedback requires candidate-hidden fitness')
        self.private_revision_feedback=private_revision_feedback

    def _feedback_enabled(self):
        return (self.private_revision_feedback and self.mechanisms.search_type=='revision'
                and self.task.answer_format in {'choice','yes_no'})

    def _fitness_transaction_options(self,selected):
        options=super()._fitness_transaction_options(selected)
        if not self._feedback_enabled():
            return options
        variations=self.mechanisms.generations*(self.mechanisms.population_size-1)
        reserve=2*variations*self.feedback_stage_byte_cap
        return [{**row,'private_revision_feedback_wire_reserve':reserve} for row in options
                if row['fitness_wire_bound']+row['closing_wire_reserve']+reserve<=row['remaining_wire_bytes']]

    def _private_variation_context(self,parent,context,stage):
        context=super()._private_variation_context(parent,context,stage)
        if not self._feedback_enabled():
            return context
        if stage not in {'variation_action','candidate_answer'}:
            raise ValueError('unknown private revision stage')
        alternatives=([s.split('.',1)[0].strip() for s in question_choices(self.task.question)]
                      if self.task.answer_format=='choice' else ['Yes','No'])
        receipt=parent_source_feedback(self.events,self.policy.records,parent.candidate_id,
                                      parent.payload['answer'],alternatives)
        wire=feedback_wire_bytes(receipt)
        if wire>self.feedback_stage_byte_cap:
            raise ValueError('private feedback exceeds reserved bounded receipt')
        if self.cost.message_latent_bytes+self.cost.message_structure_bytes+wire>self.max_packet_bytes:
            raise ValueError('private feedback exceeds actual wire cap')
        self.cost+=CostVector(message_structure_bytes=wire)
        self.events.append({'kind':'private_revision_feedback_delivery','stage':stage,
            'parent_candidate_id':parent.candidate_id,'receipt':receipt,
            'actual_receipt_UTF8_bytes':wire,'logical_message_accounting':True,
            'physical_network_measurement':False,'receiver':'reasoning',
            'decision_index_before_call':len(self.policy.records),'extra_model_calls':0,
            'new_image_encodings':0,'semantic_support_granted':False,'public_draft_exported':False})
        return {**context,'private_parent_source_review':receipt,
            'private_revision_review_rule':
                'Use the parent-bound source review to identify the assumption that needs testing. '
                'Compare differing public options against the attached evidence and exact question. '
                'A review is fallible: neither agreeing with it nor changing the parent is mandatory. '
                'Preserve a supported answer, revise a contradicted interpretation, or request useful '
                'missing detail through the currently offered operation. Repeated same-source readings '
                'are not extra votes. Return only this action output; do not invent a new observation.'}

    def _initialize_task(self):
        self.events.append({'kind':'private_revision_feedback_protocol',
            'enabled':self.private_revision_feedback,'active_for_task':self._feedback_enabled(),
            'max_source_groups':4,'receipt_stage_byte_cap':self.feedback_stage_byte_cap,
            'extra_model_calls':0,'new_support_or_answer_override':False,'reference_used':False})
        return super()._initialize_task()
