"""Train/evaluate the full multi-agent workflow on local real-image manifests."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import os
import importlib.metadata


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def save_checkpoint(policy,path,metadata):
    import torch
    path.mkdir(exist_ok=False)
    policy.save(path)
    write_json(path/'trainer.json',metadata)
    torch.save({'torch':torch.get_rng_state(),'python':random.getstate(),
                'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},path/'rng.pt')


def snapshot_source(out):
    """Archive the executing source, including dependencies, before any rollout."""
    import rs_hmcomm
    root=Path(__file__).resolve().parents[1]
    imported=Path(rs_hmcomm.__file__).resolve()
    if imported.parent!=root/"src/rs_hmcomm":
        raise RuntimeError("wrong editable source imported: expected this mainline checkout")
    files=list((root/"src/rs_hmcomm").rglob("*.py"))
    files += [Path(__file__).resolve(),root/"pyproject.toml"]
    files += list((root/'scripts').glob('*paper*.py'))
    files += list((root/'ops/kml').glob('*paper*'))
    files += list((root/"ops/kml").glob("*workflow*"))
    hashes={}
    for source in sorted(files):
        if not source.is_file():continue
        rel=source.relative_to(root);dest=out/"source"/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        hashes[str(rel)]=hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(out/"source_hashes.json",hashes)
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--model-revision",help="defaults to the model's local evors_provenance.json")
    p.add_argument("--train-manifest")
    p.add_argument("--eval-manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--checkpoint")
    p.add_argument('--checkpoint-every',type=int,default=4,help='save a resumable checkpoint every N updates; zero disables periodic saves')
    p.add_argument("--episodes",type=int,default=8)
    p.add_argument("--eval-limit",type=int)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-pixels",type=int,default=65536)
    p.add_argument("--max-new-tokens",type=int,default=192)
    p.add_argument("--max-steps",type=int,default=10)
    p.add_argument("--max-model-calls",type=int,default=24)
    p.add_argument("--max-tool-calls",type=int,default=8)
    p.add_argument("--learning-rate",type=float,default=2e-5)
    p.add_argument("--ppo-epochs",type=int,default=1)
    p.add_argument('--paper-mechanisms',action='store_true')
    p.add_argument('--disable',default='',help='comma separated semantic,compression,temporal,evolution,pointer,objects')
    p.add_argument('--communication-modes',default='text,tokens,compressed,temporal')
    p.add_argument('--compressed-tokens',type=int,default=16,choices=(4,16,64))
    p.add_argument('--initial-global-observation',action='store_true',help='development control: commit and assess the cached global overviews before adaptive routing')
    p.add_argument('--structured-requirement-actions',action='store_true',help='development control: free requirement text plus grammar-constrained kind/scope/frame attributes')
    p.add_argument('--progress-guard',action='store_true',help='verify a public candidate before replacement; stop after accepted complete support; distinguish negative answers from conflicting observations')
    p.add_argument('--spatial-codec',action='store_true',help='allocate the same total visual slots per source with a spatial attention prior; preserve source/date labels')
    p.add_argument('--max-packet-bytes',type=int,default=16777216)
    p.add_argument('--initial-duals',type=float,nargs=3,default=(.01,.005,.005),metavar=('MODEL','TOOL','WIRE'),
                   help='paper PPO initial budget prices; projected dual updates remain enabled')
    p.add_argument("--curriculum-episodes",type=int,default=0,
                   help="initial full-observation, all-role course; disabled after these updates and during evaluation")
    args=p.parse_args()
    import torch
    from rs_hmcomm.workflow.evidence import load_workflow_manifest,check_disjoint
    from rs_hmcomm.workflow.qwen_policy import QwenWorkflowPolicy
    from rs_hmcomm.workflow.runtime import WorkflowRuntime
    from rs_hmcomm.workflow.training import WorkflowTrainer,PaperWorkflowTrainer,score_episode
    random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    train=list(load_workflow_manifest(args.train_manifest)) if args.train_manifest else []
    evaluation=list(load_workflow_manifest(args.eval_manifest))
    check_disjoint(train,evaluation)
    if args.episodes<=0 or args.checkpoint_every<0 or args.eval_limit is not None and args.eval_limit<=0:
        raise ValueError("episodes/eval limit must be positive")
    if not evaluation or args.train_manifest and not train:raise ValueError("empty manifest")
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    provenance_path=Path(args.model)/"evors_provenance.json"
    provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if args.model_revision and provenance.get("revision") and args.model_revision!=provenance["revision"]:
        raise ValueError("requested model revision conflicts with saved model provenance")
    revision=args.model_revision or provenance.get("revision")
    source_hash=snapshot_source(out)
    versions={name:importlib.metadata.version(name) for name in ("torch","transformers","peft","Pillow")}
    run_info={**vars(args),"model_revision":revision,"model_provenance":provenance,
        "source_sha256":source_hash,"library_versions":versions,
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu":torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else None,
        "train_samples":len(train),
        "eval_samples":len(evaluation),"train_sha256":hashlib.sha256(Path(args.train_manifest).read_bytes()).hexdigest() if train else None,
        "eval_sha256":hashlib.sha256(Path(args.eval_manifest).read_bytes()).hexdigest(),
        "metric":"explicit task-typed development utility (exact/caption F1/bbox IoU)","official_benchmark_metric":False,
        "train_eval_disjoint":True,"training_scope":"all sampled controller and cognitive expert completions"}
    write_json(out/"run.json",run_info)
    policy=QwenWorkflowPolicy(args.model,device=args.device,max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,learning_rate=args.learning_rate,
        revision=revision,paper_mechanisms=args.paper_mechanisms,compressed_tokens=args.compressed_tokens,spatial_codec=args.spatial_codec)
    if args.paper_mechanisms:
        from rs_hmcomm.workflow.paper_runtime import PaperWorkflowRuntime,PaperMechanisms
        disabled=set(filter(None,args.disable.split(',')))
        valid={'semantic','compression','temporal','evolution','pointer','objects'}
        if disabled-valid:raise ValueError('unknown disabled mechanisms')
        mechanism=PaperMechanisms(**{k:k not in disabled for k in valid},modes=tuple(args.communication_modes.split(',')),
            initial_global_observation=args.initial_global_observation,
            structured_requirement_actions=args.structured_requirement_actions,progress_guard=args.progress_guard)
        runtime=PaperWorkflowRuntime(policy,mechanisms=mechanism,max_steps=args.max_steps,
            max_model_calls=args.max_model_calls,max_tool_calls=args.max_tool_calls,max_packet_bytes=args.max_packet_bytes)
        trainer=PaperWorkflowTrainer(policy,epochs=args.ppo_epochs,initial_duals=args.initial_duals,
            cost_targets=(args.max_model_calls/2,args.max_tool_calls/2,args.max_packet_bytes/2097152))
    else:
        runtime=WorkflowRuntime(policy,max_steps=args.max_steps,max_model_calls=args.max_model_calls,max_tool_calls=args.max_tool_calls,max_packet_bytes=args.max_packet_bytes)
        trainer=WorkflowTrainer(policy,epochs=args.ppo_epochs)
    if args.checkpoint:
        run_info["loaded_checkpoint"]=policy.load(args.checkpoint,resume_training=bool(train))
        if train:
            metadata=json.loads((Path(args.checkpoint)/"trainer.json").read_text())
            if args.paper_mechanisms:trainer.load_state_dict(metadata)
            else:trainer.baseline=metadata["baseline"];trainer.updates=metadata["updates"]
            run_info["resumed_trainer"]={**metadata,"optimizer_states":len(policy.optimizer.state)}
            rng=torch.load(Path(args.checkpoint)/"rng.pt",map_location="cpu",weights_only=True)
            torch.set_rng_state(rng["torch"]);random.setstate(rng["python"])
            if torch.cuda.is_available():torch.cuda.set_rng_state_all(rng["cuda"])
        write_json(out/"run.json",run_info)
    role_counts=Counter();gradient_totals=Counter();summaries=[]
    for index in range(args.episodes if train else 0):
        runtime.role_coverage_curriculum = trainer.updates<args.curriculum_episodes
        if args.paper_mechanisms:runtime.communication_dual=trainer.duals[2]
        task,reference=train[trainer.updates%len(train)]
        episode=runtime.run(task,training=True)
        folder=out/f"train-{index:04d}";folder.mkdir()
        write_json(folder/"trajectory.json",episode.to_dict())
        for packet in policy.packets.values():
            (folder/(packet.artifact.artifact_id+".bin")).write_bytes(policy.store.get(packet.artifact.artifact_id).data)
        audit=trainer.update(episode,reference)
        write_json(folder/"update.json",audit)
        summaries.append({"sample_id":task.sample_id,"answer":episode.answer,"stop":episode.stop_reason,**audit})
        role_counts.update(audit["roles_trained"])
        gradient_totals.update(audit["gradient_norm_by_role"])
        if args.checkpoint_every and trainer.updates%args.checkpoint_every==0:
            save_checkpoint(policy,out/f'checkpoint-step-{trainer.updates:06d}',
                trainer.state_dict() if args.paper_mechanisms else {'baseline':trainer.baseline,'updates':trainer.updates})
        print(json.dumps({"episode":index,"reward":audit["reward"]["total"],"stop":episode.stop_reason,
                          "roles":audit["roles_trained"],"changed":audit["parameters_changed"]}),flush=True)
    if train:
        save_checkpoint(policy,out/'checkpoint',trainer.state_dict() if args.paper_mechanisms else {'baseline':trainer.baseline,'updates':trainer.updates})
    frozen_hash=policy.parameter_hash();evaluated=[]
    runtime.role_coverage_curriculum = False
    for index,(task,reference) in enumerate(evaluation[:args.eval_limit]):
        episode=runtime.run(task,training=False);reward=score_episode(episode,reference)
        write_json(out/f"eval-{index:04d}.json",episode.to_dict())
        folder=out/f"eval-{index:04d}-payloads";folder.mkdir()
        for packet in policy.packets.values():
            (folder/(packet.artifact.artifact_id+".bin")).write_bytes(policy.store.get(packet.artifact.artifact_id).data)
        evaluated.append({"sample_id":task.sample_id,"answer":episode.answer,"reference":reference,
                          "stop":episode.stop_reason,"cost":episode.cost.to_dict(),"reward":reward.to_dict()})
    if frozen_hash!=policy.parameter_hash():raise RuntimeError("evaluation modified model parameters")
    write_json(out/"summary.json",{"training":summaries,"evaluation":evaluated,
        "source_sha256":source_hash,"model_identity":policy.identity,"eval_parameter_hash":frozen_hash,
        "roles_trained":dict(role_counts),"gradient_norm_by_role":dict(gradient_totals),
        "all_roles_received_gradients":all(gradient_totals.get(r,0)>0 for r in policy.roles),
        "parameters_changed":any(a["parameters_changed"] for a in summaries),
        "evaluation_parameters_frozen":frozen_hash==policy.parameter_hash(),
        "eval_correct":sum(e["reward"]["correctness"] for e in evaluated),"eval_total":len(evaluated),
        "vision_encodes":policy.vision_encodes,"receiver_visual_forwards":policy.receiver_visual_forwards,
        "scope":"real-image development smoke, not a performance claim"})
    print("WORKFLOW_COMPLETE",flush=True)


if __name__=="__main__":main()
