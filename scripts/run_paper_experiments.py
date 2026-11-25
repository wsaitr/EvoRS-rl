"""Execute immutable paired experiment cases, with no outcome-based selection."""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import torch
from train_workflow import write_json,snapshot_source
from rs_hmcomm.workflow.evidence import load_workflow_manifest,check_disjoint,ImageFrame,WorkflowTask
from rs_hmcomm.workflow.qwen_policy import QwenWorkflowPolicy
from rs_hmcomm.workflow.paper_runtime import PaperWorkflowRuntime,PaperMechanisms
from rs_hmcomm.workflow.runtime import WorkflowRuntime
from rs_hmcomm.workflow.baselines import IndependentRunner
from rs_hmcomm.workflow.training import PaperWorkflowTrainer,score_episode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',plan);source=snapshot_source(out)
    model=Path(plan['model']);provenance=json.loads((model/'evors_provenance.json').read_text())
    random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    policy=QwenWorkflowPolicy(str(model),revision=provenance['revision'],paper_mechanisms=True,max_new_tokens=256,max_pixels=plan['max_pixels'])
    initial={n:p.detach().cpu().clone() for n,p in policy.model.named_parameters() if p.requires_grad}
    auxiliary=copy.deepcopy(policy.auxiliary.state_dict());initial_optimizer=copy.deepcopy(policy.optimizer.state_dict())
    initial_hash=policy.parameter_hash();results=[]
    write_json(out/'run.json',{'source_sha256':source,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'model':provenance,'gpu':torch.cuda.get_device_name(0),'initial_parameter_hash':initial_hash,'worker':args.worker,
        'metric_note':'subset exact/MCQ; caption token F1 development reward; separately saved official corpus scores',
        'no_test_reward_updates':True})
    for case in [c for c in plan['cases'] if c['worker']==args.worker]:
        with torch.no_grad():
            for n,p in policy.model.named_parameters():
                if p.requires_grad:p.copy_(initial[n].to(p.device))
            policy.auxiliary.load_state_dict(auxiliary)
        policy.optimizer.load_state_dict(initial_optimizer)
        seed=case.get('seed',plan['seed']);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        folder=out/case['id'];folder.mkdir();write_json(folder/'case.json',case)
        modes=tuple(case.get('modes',('text','tokens','compressed','temporal')))
        flags={k:False for k in case.get('disable',[])}
        mechanism=PaperMechanisms(**flags,modes=modes,fixed_schedule=case.get('fixed',False),
            expert_roles=tuple(case.get('experts',('knowledge','reasoning','verification'))),
            search_type=case.get('search_type','revision'),payload_intervention=case.get('payload','none'))
        kwargs={'max_model_calls':case.get('calls',72),'max_tool_calls':case.get('tools',8),'max_steps':case.get('steps',8),
                'max_packet_bytes':case.get('bytes',16777216)}
        runtime=PaperWorkflowRuntime(policy,mechanisms=mechanism,**kwargs)
        if case.get('method') in {'pure_model','single_agent'}:
            runtime=IndependentRunner(policy,agent=case['method']=='single_agent',max_model_calls=kwargs['max_model_calls'],max_tool_calls=kwargs['max_tool_calls'])
        elif case.get('method')=='legacy':runtime=WorkflowRuntime(policy,**kwargs)
        train=list(load_workflow_manifest(case['train'])) if case.get('train') else []
        evaluation=list(load_workflow_manifest(case['manifest']))[:case['limit']]
        for path in ([case['train']] if train else [])+[case['manifest']]:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=plan['manifest_sha256'][path]:raise ValueError('manifest changed after protocol freeze')
        if train:check_disjoint(train,evaluation)
        trainer=PaperWorkflowTrainer(policy,cost_targets=(kwargs['max_model_calls']/2,kwargs['max_tool_calls']/2,kwargs['max_packet_bytes']/2097152))
        case_results=[];updates=[]
        for index in range(case.get('episodes',0)):
            runtime.role_coverage_curriculum=index<2;runtime.communication_dual=trainer.duals[2]
            task,reference=train[index%len(train)];episode=runtime.run(task,training=True)
            target=folder/f'train-{index:04d}';target.mkdir();write_json(target/'trajectory.json',episode.to_dict())
            update=trainer.update(episode,reference);write_json(target/'update.json',update);updates.append(update)
            for packet in policy.packets.values():
                (target/(packet.artifact.artifact_id+'.bin')).write_bytes(policy.store.get(packet.artifact.artifact_id).data)
            print(json.dumps({'case':case['id'],'train':index,'stop':episode.stop_reason,'reward':update['reward']['total']}),flush=True)
        if train:
            checkpoint=folder/'checkpoint';checkpoint.mkdir();policy.save(checkpoint);write_json(checkpoint/'trainer.json',trainer.state_dict())
            torch.save({'torch':torch.get_rng_state(),'python':random.getstate(),'cuda':torch.cuda.get_rng_state_all()},checkpoint/'rng.pt')
            saved=policy.parameter_hash();policy.load(checkpoint,resume_training=True)
            if policy.parameter_hash()!=saved:raise ValueError('checkpoint reload mismatch')
        runtime.role_coverage_curriculum=False;frozen=policy.parameter_hash()
        for index,(task,reference) in enumerate(evaluation):
            if case.get('perturb')=='blur':
                from PIL import ImageFilter
                task=WorkflowTask(task.sample_id,task.question,tuple(ImageFrame(f.frame_id,f.timestamp,f.image.filter(ImageFilter.GaussianBlur(2)),
                    f.reference_to_frame,f.alignment_confidence,f.source) for f in task.frames),task.scene_id,task.answer_format)
            started=time.monotonic();episode=runtime.run(task,training=False);reward=score_episode(episode,reference)
            write_json(folder/f'eval-{index:04d}.json',episode.to_dict())
            case_results.append({'sample_id':task.sample_id,'answer':episode.answer,'reference':reference,'stop':episode.stop_reason,
                'cost':episode.cost.to_dict(),'coverage':episode.coverage,'score':reward.correctness,'elapsed_seconds':time.monotonic()-started,
                'image_sizes':[list(f.image.size) for f in task.frames]})
            # Persist interrupted-run progress without promoting it to a completed case.
            write_json(folder/'progress.json',{'evaluation':case_results,'completed':False})
            print(json.dumps({'case':case['id'],'eval':index,'stop':episode.stop_reason,'score':reward.correctness}),flush=True)
        if policy.parameter_hash()!=frozen:raise RuntimeError('evaluation modified parameters')
        result={'case':case,'evaluation':case_results,'training':updates,'initial_hash':initial_hash,
            'eval_parameter_hash':frozen,'mean_score':sum(r['score'] for r in case_results)/len(case_results),
            'completion_rate':sum(r['stop']=='finished' for r in case_results)/len(case_results),'completed':True}
        write_json(folder/'summary.json',result);results.append(result);write_json(out/'summary.json',{'cases':results,'completed':False})
    write_json(out/'summary.json',{'cases':results,'completed':True});print('PAPER_WORKER_COMPLETE',flush=True)


if __name__=='__main__':main()
