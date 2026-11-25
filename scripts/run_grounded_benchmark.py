"""Frozen DEV predictions with shared artifact submission and external scoring."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from train_workflow import write_json, snapshot_source
from rs_hmcomm.workflow.quality import QualityWorkflowPolicy
from rs_hmcomm.workflow.benchmark_runtime_v5 import GroundedBenchmarkRuntime
from rs_hmcomm.workflow.benchmark_submission import PureArtifactRunner
from rs_hmcomm.workflow.benchmark_artifact_metrics import annotation_score
from rs_hmcomm.workflow.metrics import token_f1, task_score


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'data', 'output', 'cohort'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--method', choices=['pure_model','single_agent','text_multi','innovation_multi'], required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(exist_ok=False)
    cohort = json.loads(Path(args.cohort).read_text())
    if cohort['phase'] != 'DEV artifact preflight': raise ValueError('not an independent test runner')
    data = Path(args.data)/'ThinkGeo'
    public = [json.loads(line) for line in (data/'public.jsonl').read_text().splitlines()]
    tasks = {row['sample_id']: row for row in public}
    rows = [tasks[sid] for sid in cohort['sample_ids']]
    if len(set(cohort['sample_ids'])) != len(rows): raise ValueError('duplicate cohort')
    if any(row['split'] != 'dev' or not row['assets_downloaded'] or row['modality'] != 'optical' for row in rows):
        raise ValueError('cohort requires available optical DEV assets')
    random.seed(17); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    provenance = json.loads((Path(args.model)/'evors_provenance.json').read_text())
    policy = QualityWorkflowPolicy(args.model, revision=provenance['revision'], max_pixels=262144,
        max_new_tokens=256, paper_mechanisms=True, spatial_codec=True, compressed_tokens=64, stable_codec=True)
    # Role prompts are identical in both multi-Agent arms; communication alone differs.
    policy.role_priors = args.method in {'text_multi', 'innovation_multi'}
    source_sha = snapshot_source(output); frozen = policy.parameter_hash()
    import shutil
    runner = output/'source/scripts'/Path(__file__).name
    shutil.copyfile(__file__,runner)
    write_json(output/'run.json', {'arguments':vars(args), 'source_sha256':source_sha,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cohort_sha256':hashlib.sha256(Path(args.cohort).read_bytes()).hexdigest(),
        'model_identity':policy.identity, 'initial_parameters':frozen,
        'shared_submission_renderer':True, 'official_judge':False, 'phase':'DEV artifact preflight'})
    results = []
    # The inference task contains only public question/assets. Private reference
    # files are first opened AFTER all cohort predictions have been saved.
    for index, task in enumerate(rows):
        inference = {key:task[key] for key in ('sample_id','question','assets')}
        runtime = (PureArtifactRunner(policy, output=output/f'artifacts-{index:04d}') if args.method=='pure_model'
            else GroundedBenchmarkRuntime(policy, multi=args.method!='single_agent',
                communication='compressed' if args.method=='innovation_multi' else 'text',
                max_calls=32,max_tools=16,output=output/f'artifacts-{index:04d}'))
        episode = runtime.run(inference, training=False)
        write_json(output/f'prediction-{index:04d}.json', episode.to_dict())
        results.append({'sample_id':task['sample_id'],'stop':episode.stop_reason,
            'answer':episode.answer,'cost':episode.cost.to_dict()})
        write_json(output/'progress.json', {'evaluation':results,'completed':False})
        print(json.dumps({'method':args.method,'item':index,'stop':episode.stop_reason}),flush=True)
    if policy.parameter_hash()!=frozen: raise ValueError('evaluation modified weights')
    references = {r['sample_id']:r for r in [json.loads(line) for line in (data/'references.jsonl').read_text().splitlines()]}
    for index, (task,row) in enumerate(zip(rows,results)):
        reference = references[task['sample_id']]; valid = row['stop']=='finished'
        prediction = json.loads((output/f'prediction-{index:04d}.json').read_text())
        row['annotation_proxy'] = annotation_score(reference,prediction['events'],task['assets'])
        row['submission_finished'] = valid
        if row['annotation_proxy'].get('supported') and not valid:
            row['annotation_proxy']['bbox_f1_at_0_5_dev'] = 0.
            row['annotation_proxy']['greedy_bbox_iou_f1_dev'] = 0.
        row['exact_dev'] = max((task_score(row['answer'],a) for a in reference['answers']),default=0.) if valid else 0.
        row['token_f1_dev'] = max((token_f1(row['answer'],a) for a in reference['answers']),default=0.) if valid else 0.
    write_json(output/'summary.json', {'evaluation':results,'completed':True,'parameters_frozen':True,
        'official_judge':False,'phase':'DEV artifact preflight','performance_superiority_established':False})


if __name__=='__main__': main()
