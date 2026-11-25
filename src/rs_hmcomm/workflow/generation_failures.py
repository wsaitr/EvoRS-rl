"""Generation-attempt telemetry; failures are not fabricated policy completions."""
import time


def model_attempts(policy):
    return len(policy.records)+len(getattr(policy,'failed_generations',()))


def tracked_generate(policy, model_inputs, *, role, prompt, packet_ids, started, **arguments):
    native = getattr(policy, 'generate_native', None)
    if callable(native):
        return native(model_inputs, role=role, prompt=prompt,
                      packet_ids=packet_ids, started=started, **arguments)
    from transformers import LogitsProcessor,LogitsProcessorList
    prompt_length=model_inputs['input_ids'].shape[1]
    observed=[]
    class ObservePrefix(LogitsProcessor):
        def __call__(self,current_ids,scores):
            nonlocal observed
            observed=current_ids[0,prompt_length:].detach().cpu().tolist()
            return scores
    processors=list(arguments.pop('logits_processor',None) or ())
    arguments['logits_processor']=LogitsProcessorList([ObservePrefix(),*processors])
    try:
        return policy.model.generate(**model_inputs,**arguments)
    except Exception as error:
        failure={'attempt_index':model_attempts(policy),'role':role,'prompt':prompt,
            'packet_ids':list(packet_ids),'model_input_tokens':int(model_inputs['input_ids'].numel()),
            'observed_output_ids':observed,'observed_output_text':policy.processor.tokenizer.decode(observed,skip_special_tokens=True),
            'model_output_tokens_lower_bound':len(observed),'output_token_count_exact':False,
            'latency_ms':(time.perf_counter()-started)*1000,'error_type':type(error).__name__,'error':str(error),
            'complete_policy_decision':False,'PPO_likelihood_fabricated':False}
        if not hasattr(policy,'failed_generations'):policy.failed_generations=[]
        policy.failed_generations.append(failure)
        error.evors_generation_failure=failure
        raise


def failure_cost(failure, agent):
    from rs_hmcomm.contracts import CostVector
    return CostVector(agent_calls=int(agent),model_calls=1,
        model_input_tokens=failure['model_input_tokens'],
        model_output_tokens=failure['model_output_tokens_lower_bound'],billed_compute_ms=failure['latency_ms'])
