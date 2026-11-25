"""Pure-model and a genuine single tool-using Agent, on identical task inputs."""
import hashlib
import json
import time
from .runtime import WorkflowEpisode,answer_instruction,answer_valid,strict_json
from .evidence import crop_frame,observation_encodable
from .generation_failures import model_attempts,failure_cost
from rs_hmcomm.contracts import CostVector


class IndependentRunner:
    def __init__(self,policy,*,agent=False,max_model_calls=72,max_tool_calls=8,measurement_factory=None,arithmetic_tools=False,numeric_expression_actions=False,private_deliberation=False,react_planning=False,preserve_direct_proposal=False):
        self.policy=policy;self.agent=agent;self.max_model_calls=max_model_calls;self.max_tool_calls=max_tool_calls
        if measurement_factory is not None and not agent:
            raise ValueError('pure model cannot use measurement tools')
        if arithmetic_tools and not agent:raise ValueError('pure model cannot use arithmetic tools')
        if numeric_expression_actions and not (agent and arithmetic_tools):
            raise ValueError('numeric expression actions require a single Agent with arithmetic tools')
        self.arithmetic_tools=arithmetic_tools
        self.numeric_expression_actions=numeric_expression_actions
        self.private_deliberation=private_deliberation
        if react_planning and not agent:
            raise ValueError('ReAct planning requires a tool-using single Agent')
        self.react_planning=react_planning
        if preserve_direct_proposal and not (agent and react_planning and max_model_calls>=2):
            raise ValueError('direct proposal requires a ReAct Agent and at least two model calls')
        self.preserve_direct_proposal=preserve_direct_proposal
        self.measurement_factory = measurement_factory

    def run(self,task,training=False):
        if training:raise ValueError('baseline does not update parameters')
        deliberation_fields=None
        if self.private_deliberation and task.answer_format in {'choice','yes_no'}:
            from .private_deliberation import deliberation_schema
            deliberation_fields=deliberation_schema(task.question,task.answer_format)
        self.policy.begin_episode(training=False);self.policy.payload_intervention='none'
        packets=[];events=[];cost=CostVector();answer=None;stop='model_budget_exhausted'
        for frame in task.frames:
            pid='global-'+frame.frame_id
            self.policy.encode(frame.image,artifact_id=pid,source_image_ref=task.scene_id,roi=(0,0,1,1),label=frame.frame_id+' / '+frame.timestamp)
            packets.append(pid)
        history=[];observed=set();measurements=set();toolbox=None;measurement_no_progress=0;arithmetic_requests=set();arithmetic_no_progress=0
        def call(instruction,output_format='text',choices=None,json_field_schema=None):
            nonlocal cost
            if model_attempts(self.policy)>=self.max_model_calls:raise RuntimeError('model_budget_exhausted')
            try:
                options={} if json_field_schema is None else {'json_field_schema':json_field_schema}
                text=self.policy.decide('reasoning',instruction,tuple(packets),output_format=output_format,allowed_actions=choices,**options)
            except Exception as error:
                failure=getattr(error,'evors_generation_failure',None)
                if failure is not None:
                    cost+=failure_cost(failure,self.agent)
                    events.append({'kind':'policy_generation_failure',**failure})
                raise
            r=self.policy.records[-1];cost+=CostVector(agent_calls=int(self.agent),model_calls=1,
                model_input_tokens=int(r.input_ids.numel()),model_output_tokens=int(r.output_ids.numel()),billed_compute_ms=r.latency_ms)
            events.append({'kind':'policy_decision','role':'single_agent' if self.agent else 'pure_model','packet_ids':list(packets),'response':text})
            return text
        proposal=None
        if self.preserve_direct_proposal:
            # Exactly the direct pure-model prompt, before any private analysis.
            try:
                proposal=answer_valid(call('Task: '+task.question+'\n'+answer_instruction(task.answer_format)),task.answer_format)
                events.append({'kind':'single_direct_proposal','answer':proposal,'reference_used':False})
            except (ValueError,TypeError) as error:
                cost+=CostVector(invalid_actions=1,errors=1)
                events.append({'kind':'single_direct_proposal_failure','error':str(error)})
        if self.agent:
            preparation_failures=0
            while model_attempts(self.policy)<self.max_model_calls-1:
                choices=[{'operation':'answer'}]
                if cost.tool_calls<self.max_tool_calls:
                    boxes=((0,0,1,1),(0,0,.5,.5),(.5,0,1,.5),(0,.5,.5,1),(.5,.5,1,1))
                    choices += [{'operation':'crop','frame_id':f.frame_id,'bbox':list(b)} for f in task.frames for b in boxes
                                if (f.frame_id,b) not in observed and observation_encodable(task.frames,{'op':'crop','frame_id':f.frame_id,'bbox':b})]
                    if self.arithmetic_tools and arithmetic_no_progress<2 and model_attempts(self.policy)<=self.max_model_calls-3:
                        choices.append({'operation':'calculate'})
                    # Reserve a tool-request response and the final answer response.
                    if self.measurement_factory is not None and measurement_no_progress<2 and model_attempts(self.policy)<=self.max_model_calls-3:
                        choices.append({'operation':'measurement'})
                instruction = ('\nChoose a useful crop to inspect or answer now. Previous actions: '
                    if self.measurement_factory is None else
                    '\nChoose a useful crop, optional measurement if available, or answer now. '
                    'Previous actions and real tool results: ')
                if {'operation':'measurement'} in choices:
                    from .measurement_workflow import MEASUREMENT_CAPABILITIES
                    instruction += '\nAvailable optional measurement capabilities: '+json.dumps(MEASUREMENT_CAPABILITIES)+'\nPrevious actions and real tool results: '
                if self.arithmetic_tools:
                    instruction += '\nOptional arithmetic: use calculate for + - * / ** %, sqrt, abs, min, max, round with actual observed or public-task numbers. Never invent measurements, scale or units. Previous calculations: '+str(history)
                    if arithmetic_no_progress>=2:
                        instruction += '\nArithmetic has failed to add a usable result twice. Inspect a new detail crop or obtain a successful measurement before requesting arithmetic again; answering remains available.'
                try:
                    attempts_before=model_attempts(self.policy)
                    if self.react_planning:
                        from .single_react import planning_prompt, planning_schema, resolve_action
                        response=strict_json(call(planning_prompt(task.question,history,choices),
                            'json',json_field_schema=planning_schema(choices)))
                        action=resolve_action(response,choices)
                        history.append({'kind':'agent_analysis','analysis':response['analysis'],
                                        'observed_fact_verified':False})
                        events.append({'kind':'single_react_planning','analysis':response['analysis'],
                                       'operation':action,'reference_used':False})
                    else:
                        action=strict_json(call('You are ONE tool-using Agent. Task: '+task.question+
                            instruction+str(history), 'json',choices))
                except (ValueError,TypeError) as error:
                    cost+=CostVector(invalid_actions=1,errors=1)
                    history.append({'operation_error':str(error),'tool_executed':False})
                    # A failure before generation consumes no model-call slot.
                    # Bound recovery separately so its counter cannot loop forever.
                    preparation_failures=preparation_failures+1 if model_attempts(self.policy)==attempts_before else 0
                    if preparation_failures>=2:
                        events.append({'kind':'operation_preparation_exhausted','attempts':preparation_failures,
                                       'model_attempts':model_attempts(self.policy)})
                        break
                    continue
                preparation_failures=0
                if action['operation']=='answer':break
                if action['operation']=='calculate':
                    from .benchmark_tools import calculate
                    import math
                    request=None;started=None;result=None
                    try:
                        expression_type='numeric_expression' if self.numeric_expression_actions else 'string'
                        request=strict_json(call('Task: '+task.question+'\nYou remain ONE Agent. Request arithmetic as exactly expression (string) and output_unit (string). Use actual public or observed numbers, never inferred GSD or invented measurements. '+str(history),'json',json_field_schema=[['expression',expression_type],['output_unit','string']]))
                        if set(request)!={'expression','output_unit'} or not isinstance(request['expression'],str) or not 1<=len(request['expression'])<=256 or not isinstance(request['output_unit'],str) or len(request['output_unit'])>80:
                            raise ValueError('bounded expression and output_unit required')
                        signature=json.dumps(request,sort_keys=True)
                        if signature in arithmetic_requests:raise ValueError('identical arithmetic already computed')
                        cost+=CostVector(tool_calls=1)
                        started=time.perf_counter()
                        value=calculate(request['expression'],rounding=True)
                        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite numeric result required')
                        arithmetic_requests.add(signature)
                        arithmetic_no_progress=0
                        result={'request':request,'value':value,'source_packet_ids':list(packets),'input_measurements_verified':False,'output_unit_verified':False}
                        history.append(result)
                        events.append({'kind':'workflow_arithmetic','success':True,**result})
                    except (ValueError,TypeError,KeyError,ArithmeticError) as error:
                        cost+=CostVector(invalid_actions=1,errors=1)
                        arithmetic_no_progress+=1
                        history.append({'request':request,'arithmetic_error':str(error)})
                        events.append({'kind':'workflow_arithmetic','success':False,'request':request,'error':str(error)})
                    if started is not None:
                        elapsed=(time.perf_counter()-started)*1000
                        cost+=CostVector(billed_compute_ms=elapsed)
                        events[-1]['actual_execution_ms']=elapsed
                    continue
                if action['operation']=='measurement':
                    from .measurement_workflow import MEASUREMENT_SCHEMAS, measurement_context
                    if toolbox is None:
                        toolbox=self.measurement_factory(task,self.policy)
                    try:
                        request=strict_json(call('Task: '+task.question+'\nYou remain ONE Agent. '
                            'Request ONE optional measurement tool using exactly its listed fields. '
                            'Return the tool request JSON itself. Do not invent measurements or IDs.\n'+
                            json.dumps(measurement_context(toolbox,[f.frame_id for f in task.frames],measurements)), 'json'))
                    except (ValueError,TypeError) as error:
                        cost+=CostVector(invalid_actions=1,errors=1)
                        measurement_no_progress+=1
                        history.append({'error':str(error),'tool_executed':False})
                        continue
                    signature=json.dumps(request,sort_keys=True)
                    if request.get('tool') not in MEASUREMENT_SCHEMAS or signature in measurements:
                        cost+=CostVector(invalid_actions=1)
                        measurement_no_progress+=1
                        history.append({'request':request,'error':'unknown or identical already executed measurement'})
                        continue
                    cost+=CostVector(tool_calls=1)
                    started=time.perf_counter()
                    event={'kind':'workflow_measurement','request':request,'reference_used':False}
                    try:
                        result=toolbox.execute(request)
                        measurements.add(signature)
                        measurement_no_progress=0
                        arithmetic_no_progress=0
                        event.update(success=True,result=result)
                    except (ValueError,KeyError,TypeError) as error:
                        cost+=CostVector(invalid_actions=1,errors=1)
                        measurement_no_progress+=1
                        event.update(success=False,error=str(error))
                    finally:
                        elapsed=(time.perf_counter()-started)*1000
                        cost+=CostVector(billed_compute_ms=elapsed)
                        event['actual_execution_ms']=elapsed;events.append(event)
                    from .measurement_workflow import measurement_observation
                    history.append(measurement_observation(event))
                    continue
                bbox=tuple(action['bbox']);frame=next(f for f in task.frames if f.frame_id==action['frame_id'])
                # Retain global views plus at most four latest detail views; bounded context.
                pid=f'crop-{cost.tool_calls}';self.policy.encode(crop_frame(frame,bbox),artifact_id=pid,
                    source_image_ref=task.scene_id,roi=bbox,label=frame.frame_id+' / ROI '+str(bbox))
                packets.append(pid);packets=packets[:len(task.frames)]+packets[len(task.frames):][-4:]
                cost+=CostVector(tool_calls=1);observed.add((frame.frame_id,bbox));history.append(action)
                if bbox != (0,0,1,1):
                    measurement_no_progress=0
                    arithmetic_no_progress=0
                events.append({'kind':'tool','operation':action,'output':pid})
        successful_new_tool=any((e['kind']=='tool' and tuple(e['operation']['bbox'])!=(0,0,1,1)) or
            (e['kind'] in ('workflow_arithmetic','workflow_measurement') and e.get('success')) for e in events)
        if self.preserve_direct_proposal and proposal is not None and not successful_new_tool:
            events.append({'kind':'single_direct_proposal_retained','new_tool_observation':False,
                           'analysis_alone_allowed_to_overwrite':False,'reference_used':False})
            return WorkflowEpisode(task.sample_id,proposal,'finished',None,events,list(self.policy.records),cost,0.)
        try:
            evidence = ''
            if self.measurement_factory is not None:
                from .measurement_workflow import measurement_observation
                evidence='\nActual measurement results: '+json.dumps(measurement_observation(
                    [e for e in events if e['kind']=='workflow_measurement']))
            if self.arithmetic_tools:
                evidence += '\nActual arithmetic results (inputs and units remain model-supplied): '+json.dumps([event for event in events if event['kind']=='workflow_arithmetic'])
            if self.react_planning:
                evidence += '\nYour execution trace (analysis is a model interpretation, not a verified observation): '+json.dumps(history)
            if deliberation_fields is None:
                answer=answer_valid(call('Task: '+task.question+evidence+'\n'+answer_instruction(task.answer_format)),task.answer_format)
            else:
                from .private_deliberation import INSTRUCTION,VERSION
                from .runtime import schema
                action=strict_json(call('Task: '+task.question+evidence+'\n'+INSTRUCTION,
                    'json',json_field_schema=deliberation_fields))
                schema(action,{'analysis','answer'})
                if not isinstance(action['analysis'],str):raise ValueError('private analysis must be text')
                validated_answer=answer_valid(action['answer'],task.answer_format)
                if action['answer'] not in deliberation_fields[1][1]['enum']:
                    raise ValueError('deliberation answer outside public alternatives')
                answer=validated_answer
                events.append({'kind':'private_deliberation_completion','protocol':VERSION,
                    'decision_index':len(self.policy.records)-1,'role':'single_agent' if self.agent else 'pure_model',
                    'packet_ids':list(packets),'scope':'final_answer',
                    'analysis_sha256':hashlib.sha256(action['analysis'].encode()).hexdigest(),
                    'analysis_UTF8_bytes':len(action['analysis'].encode()),
                    'analysis_nonempty':bool(action['analysis'].strip()),'candidate_answer':answer,
                    'all_completion_tokens_retained_in_record_and_cost':True,
                    'parameters_updated':False,'PPO_updates':0,'analysis_exported_to_workspace':False,
                    'new_visual_observation_claimed':False,'semantic_support_granted':False,
                    'reference_used':False,'additional_model_calls':0})
            stop='finished'
        except (ValueError,TypeError) as error:
            failed=getattr(error,'evors_generation_failure',None) is not None
            cost+=CostVector(invalid_actions=1,errors=int(failed));stop='generation_failed' if failed else 'invalid_answer'
            if self.preserve_direct_proposal and proposal is not None:
                events.append({'kind':'single_failed_revision_retains_proposal','error':str(error),
                               'revision_failure_preserved':True,'reference_used':False})
                answer=proposal;stop='finished'
        return WorkflowEpisode(task.sample_id,answer,stop,None,events,list(self.policy.records),cost,0.)
