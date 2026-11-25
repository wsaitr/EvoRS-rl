"""Execute the precommitted final protocol; test labels never enter policy inputs."""
import argparse,copy,hashlib,json,random,time,os,shutil
from pathlib import Path
import torch
from train_workflow import write_json,snapshot_source,save_checkpoint
from rs_hmcomm.workflow.evidence import load_workflow_manifest,check_disjoint,ImageFrame,WorkflowTask
from rs_hmcomm.workflow.qwen_policy import QwenWorkflowPolicy
from rs_hmcomm.workflow.paper_runtime import PaperWorkflowRuntime,PaperMechanisms
from rs_hmcomm.workflow.baselines import IndependentRunner
from rs_hmcomm.workflow.training import PaperWorkflowTrainer,score_episode

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--worker',required=True,type=int);p.add_argument('--output',required=True);args=p.parse_args()
    path=Path(args.plan);plan=json.loads(path.read_text());out=Path(args.output);out.mkdir(exist_ok=False)
    for name,digest in plan['manifests'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('frozen manifest changed')
    source=snapshot_source(out);write_json(out/'plan.json',plan)
    random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    provenance=json.loads((Path(plan['model'])/'evors_provenance.json').read_text())
    policy=QwenWorkflowPolicy(plan['model'],revision=provenance['revision'],paper_mechanisms=True,spatial_codec=True,max_pixels=65536,max_new_tokens=256,compressed_tokens=16)
    initial={n:p.detach().cpu().clone() for n,p in policy.model.named_parameters() if p.requires_grad};auxiliary=copy.deepcopy(policy.auxiliary.state_dict());optimizer=copy.deepcopy(policy.optimizer.state_dict());initial_hash=policy.parameter_hash()
    write_json(out/'run.json',{'source_sha256':source,'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_identity':policy.identity,'initial_parameter_hash':initial_hash,'worker':args.worker,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0),'test_labels_in_policy':False})
    results=[]
    def evaluate(runtime,manifest,folder):
        folder.mkdir(exist_ok=False);frozen=policy.parameter_hash();records=[]
        for i,(task,reference) in enumerate(load_workflow_manifest(manifest)):
            if current.get('blur'):
                from PIL import ImageFilter
                task=WorkflowTask(task.sample_id,task.question,tuple(ImageFrame(f.frame_id,f.timestamp,f.image.filter(ImageFilter.GaussianBlur(2)),f.reference_to_frame,f.alignment_confidence,f.source) for f in task.frames),task.scene_id,task.answer_format)
            start=time.monotonic();episode=runtime.run(task,training=False);reward=score_episode(episode,reference);write_json(folder/f'eval-{i:04d}.json',episode.to_dict())
            payloads=folder/f'eval-{i:04d}-payloads';payloads.mkdir()
            for packet in policy.packets.values():(payloads/(packet.artifact.artifact_id+'.bin')).write_bytes(policy.store.get(packet.artifact.artifact_id).data)
            write_json(payloads/'packets.json',{pid:{'artifact':p.artifact.descriptor(),'grid':p.grid,'recipe':p.recipe} for pid,p in policy.packets.items()})
            records.append({'sample_id':task.sample_id,'scene_id':task.scene_id,'answer':episode.answer,'reference':reference,'stop':episode.stop_reason,'score':reward.correctness,'cost':episode.cost.to_dict(),'elapsed_seconds':time.monotonic()-start,'image_sizes':[list(f.image.size) for f in task.frames]})
            write_json(folder/'progress.json',{'evaluation':records,'completed':False});print(json.dumps({'case':current['id'],'evaluation':folder.name,'item':i,'score':reward.correctness,'stop':episode.stop_reason}),flush=True)
        if policy.parameter_hash()!=frozen:raise ValueError('evaluation updated parameters')
        result={'evaluation':records,'completed':True,'eval_parameter_hash':frozen,'evaluation_parameters_frozen':True};write_json(folder/'summary.json',result);return result
    for current in [c for c in plan['cases'] if c['worker']==args.worker]:
        folder=out/current['id'];folder.mkdir();write_json(folder/'case.json',current)
        policy.spatial_codec=True;policy.spatial_variant='both';policy.processor.image_processor.max_pixels=current.get('max_pixels',65536)
        with torch.no_grad():
            for n,p in policy.model.named_parameters():
                if p.requires_grad:p.copy_(initial[n].to(p.device))
            policy.auxiliary.load_state_dict(auxiliary)
        policy.optimizer.load_state_dict(optimizer)
        seed=current.get('seed',17);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        checkpoint=current.get('checkpoint')
        if checkpoint:
            deadline=time.monotonic()+plan['dependency_timeout_seconds']
            # Completion marker comes AFTER final snapshot; do not race incomplete saves.
            ready=Path(checkpoint).parent/'TRAINING_COMPLETE.json'
            while not ready.exists():
                failure=Path(checkpoint).parent/'FAILED.json'
                if failure.exists():raise RuntimeError('training dependency failed: '+str(failure))
                if time.monotonic()>deadline:raise TimeoutError('training dependency did not finish')
                time.sleep(10)
            policy.load(checkpoint,resume_training=False)
        # Inference interventions are explicit; checkpoint is loaded with original variant first.
        policy.spatial_codec=current.get('spatial',True);policy.spatial_variant=current.get('variant','both');policy.payload_intervention=current.get('payload','none')
        flags={x:False for x in current.get('disable',[])}
        mechanism=PaperMechanisms(**flags,modes=tuple(current.get('modes',['compressed','temporal'])),initial_global_observation=True,structured_requirement_actions=True,progress_guard=True,fixed_schedule=current.get('fixed',False),search_type=current.get('search_type','revision'),expert_roles=tuple(current.get('experts',['knowledge','reasoning','verification'])),payload_intervention=current.get('payload','none'))
        calls=current.get('calls',96);runtime=PaperWorkflowRuntime(policy,mechanisms=mechanism,max_model_calls=calls,max_steps=12,max_tool_calls=8,max_packet_bytes=33554432)
        if current.get('method') in {'pure_model','single_agent'}:runtime=IndependentRunner(policy,agent=current['method']=='single_agent',max_model_calls=calls,max_tool_calls=8)
        runtime.communication_dual=json.loads((Path(checkpoint)/'trainer.json').read_text())['duals'][2] if checkpoint else .005
        updates=[]
        try:
            if current.get('updates'):
                train=list(load_workflow_manifest(plan['train_manifest']));check_disjoint(train,list(load_workflow_manifest(plan['dev_manifest'])));check_disjoint(train,list(load_workflow_manifest(plan['test_manifest'])))
                trainer=PaperWorkflowTrainer(policy,gamma=current.get('gamma',.99),quality_scale=current.get('quality_scale',1.0),cost_targets=(48.,4.,16.),initial_duals=(.01,.005,.005))
                evaluate(runtime,plan['dev_manifest'],folder/'dev-step-000000')
                for i in range(current['updates']):
                    runtime.role_coverage_curriculum=i<4;runtime.communication_dual=trainer.duals[2];task,reference=train[i%len(train)];episode=runtime.run(task,training=True)
                    target=folder/f'train-{i:04d}';target.mkdir();write_json(target/'trajectory.json',episode.to_dict())
                    for packet in policy.packets.values():(target/(packet.artifact.artifact_id+'.bin')).write_bytes(policy.store.get(packet.artifact.artifact_id).data)
                    write_json(target/'packets.json',{pid:{'artifact':p.artifact.descriptor(),'grid':p.grid,'recipe':p.recipe} for pid,p in policy.packets.items()})
                    update=trainer.update(episode,reference);write_json(target/'update.json',update);updates.append(update)
                    print(json.dumps({'case':current['id'],'train':i,'stop':episode.stop_reason,'reward':update['reward']['total'],'changed':update['parameters_changed']}),flush=True)
                    if (i+1)%16==0:
                        save_checkpoint(policy,folder/f'checkpoint-step-{i+1:06d}',trainer.state_dict());runtime.role_coverage_curriculum=False;runtime.communication_dual=trainer.duals[2]
                        evaluate(runtime,plan['dev_manifest'],folder/f'dev-step-{i+1:06d}')
                save_checkpoint(policy,folder/'checkpoint',trainer.state_dict());runtime.role_coverage_curriculum=False;runtime.communication_dual=trainer.duals[2]
                write_json(folder/'TRAINING_COMPLETE.json',{'updates':len(updates),'checkpoint_hash':policy.parameter_hash()})
            if current.get('updates') and current['manifest']==plan['dev_manifest']:
                origin=folder/'dev-step-000048';shutil.copytree(origin,folder/'evaluation')
                evaluation=json.loads((origin/'summary.json').read_text());evaluation['evaluation_reused_from']='dev-step-000048; no additional model inference'
                if evaluation['eval_parameter_hash']!=policy.parameter_hash():raise ValueError('reused dev parameters changed')
            else:evaluation=evaluate(runtime,current['manifest'],folder/'evaluation')
            result={'case':current,'initial_hash':initial_hash,'training':updates,**evaluation};write_json(folder/'summary.json',result);results.append({'case':current['id'],'completed':True})
        except Exception as e:
            write_json(folder/'FAILED.json',{'type':type(e).__name__,'message':str(e),'completed_updates':len(updates)});raise
        write_json(out/'summary.json',{'cases':results,'completed':False})
    write_json(out/'summary.json',{'cases':results,'completed':True});print('FINAL_WORKER_COMPLETE',flush=True)
if __name__=='__main__':main()
