"""Save real benchmark predictions and transparent proxy scores separately.

Original ThinkGeo/Earth-Bench semantic judges are NOT replaced by these proxies.
GeoPlan generates plans only. Raw imagery is never replaced by annotated output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from train_workflow import write_json, snapshot_source
from rs_hmcomm.workflow.quality import QualityWorkflowPolicy, QualityWorkflowRuntime, StableAddressedCodec
from rs_hmcomm.workflow.paper_runtime import PaperMechanisms
from rs_hmcomm.workflow.baselines import IndependentRunner
from rs_hmcomm.workflow.benchmark_runtime import BenchmarkAgentRuntime
from rs_hmcomm.workflow.evidence import load_workflow_manifest, WorkflowTask, ImageFrame
from rs_hmcomm.workflow.metrics import task_score, token_f1
from rs_hmcomm.workflow.runtime import strict_json


def plan_scores(prediction, reference):
    """Whole-tool-set and unweighted edit metrics, not official key-tool metrics."""
    if not reference:
        raise ValueError('empty reference plan')
    predicted, expected = set(prediction), set(reference)
    overlap = len(predicted & expected)
    f1 = 2*overlap/max(1, len(predicted)+len(expected))
    previous = list(range(len(reference)+1))
    for i, value in enumerate(prediction, 1):
        row = [i]
        for j, target in enumerate(reference, 1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+int(value != target)))
        previous = row
    return {'tool_set_f1_dev': f1,
            'unweighted_path_similarity_dev': 1-previous[-1]/max(1, len(prediction), len(reference)),
            'exact_tool_flow': float(prediction == reference)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--data', required=True)
    p.add_argument('--benchmark', choices=['GeoMMBench', 'ThinkGeo', 'Earth-Bench', 'GeoPlan'], required=True)
    p.add_argument('--method', choices=['pure_model', 'single_agent', 'text_multi', 'innovation_multi', 'trained_multi'], required=True)
    p.add_argument('--split', choices=['dev', 'train', 'test'], default='dev'); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output', required=True); p.add_argument('--checkpoint'); p.add_argument('--test-protocol')
    p.add_argument('--modality', choices=['all', 'optical', 'sar', 'rgb', 'spectrum', 'products'], default='all')
    p.add_argument('--tool-protocol', choices=['original', 'native', 'native_taskbound', 'native_taskbound_observed', 'native_taskbound_budget', 'native_taskbound_observed_budget', 'expert_measurement', 'collection_observation', 'collection_remoteclip'], default='original')
    p.add_argument('--detector-model', help='pinned local auxiliary detector; expert_measurement DEV only')
    p.add_argument('--crop-classifier-model', help='optional pinned RemoteCLIP; expert_measurement DEV only')
    p.add_argument('--scene-classifier-model', help='pinned RemoteCLIP; collection_remoteclip DEV only')
    p.add_argument('--workflow-protocol', choices=['original', 'transaction', 'adaptive_closing', 'early_adaptive_closing', 'early_candidate_closing', 'policy_terminal'], default='original')
    p.add_argument('--collection-route', choices=['last_four', 'all_sources'], default='last_four')
    p.add_argument('--collection-slots', choices=['total', 'per_source'], default='total')
    p.add_argument('--planning-protocol', choices=['original', 'transaction'], default='original')
    p.add_argument('--collection-actions', choices=['original', 'typed'], default='original')
    args = p.parse_args()
    runtime_class = BenchmarkAgentRuntime
    if args.collection_actions == 'typed' and (args.collection_route != 'all_sources'
                                               or args.tool_protocol != 'collection_remoteclip'):
        raise ValueError('typed collection actions require explicitly reported all-source scene-tool protocol')
    if args.planning_protocol != 'original' and (args.benchmark != 'GeoPlan' or args.split != 'dev'):
        raise ValueError('plan transactions require explicitly reported GeoPlan development protocol')
    workflow_class = QualityWorkflowRuntime
    if args.collection_route != 'last_four' and args.tool_protocol != 'collection_remoteclip':
        raise ValueError('all-source routing requires explicitly reported collection_remoteclip DEV protocol')
    if args.collection_slots != 'total' and args.collection_route != 'all_sources':
        raise ValueError('per-source slots require explicitly reported all-source collection route')
    if args.workflow_protocol != 'original':
        if args.benchmark != 'GeoMMBench' or args.split != 'dev' or args.method in {'pure_model', 'single_agent'}:
            raise ValueError('new closing protocol requires full-workflow GeoMM MAS DEV')
        if args.tool_protocol != 'original':
            raise ValueError('closing protocol must not be mixed with adapted tool protocols')
        if args.workflow_protocol == 'transaction':
            from rs_hmcomm.workflow.transaction_budget import TransactionBudgetRuntime
            workflow_class = TransactionBudgetRuntime
        elif args.workflow_protocol == 'adaptive_closing':
            from rs_hmcomm.workflow.adaptive_closing_budget import AdaptiveClosingBudgetRuntime
            workflow_class = AdaptiveClosingBudgetRuntime
        else:
            if args.workflow_protocol == 'policy_terminal':
                from rs_hmcomm.workflow.policy_terminal import PolicyTerminalRuntime
                workflow_class = PolicyTerminalRuntime
            elif args.workflow_protocol == 'early_candidate_closing':
                from rs_hmcomm.workflow.early_candidate_closing import EarlyCandidateClosingRuntime
                workflow_class = EarlyCandidateClosingRuntime
            else:
                from rs_hmcomm.workflow.early_adaptive_closing import EarlyAdaptiveClosingRuntime
                workflow_class = EarlyAdaptiveClosingRuntime
    if args.tool_protocol != 'original':
        if args.benchmark not in {'ThinkGeo', 'Earth-Bench'} or args.split != 'dev':
            raise ValueError('new tool protocol requires adapted benchmark DEV; not independent test')
        from rs_hmcomm.workflow.native_benchmark_tools import NativeDetectionRuntime, TaskBoundNativeRuntime
        runtime_class = NativeDetectionRuntime if args.tool_protocol == 'native' else TaskBoundNativeRuntime
        if args.tool_protocol in {'native_taskbound_observed', 'native_taskbound_observed_budget'}:
            if args.method != 'single_agent':
                raise ValueError('observed single protocol requires single_agent method')
            from rs_hmcomm.workflow.native_benchmark_tools import ObservingBenchmarkSingle
            runtime_class = ObservingBenchmarkSingle
        if args.tool_protocol in {'native_taskbound_budget', 'native_taskbound_observed_budget'}:
            from rs_hmcomm.workflow.benchmark_budget import BudgetTaskBoundRuntime, BudgetObservingSingle
            runtime_class = BudgetTaskBoundRuntime if args.tool_protocol == 'native_taskbound_budget' else BudgetObservingSingle
        if args.tool_protocol == 'expert_measurement':
            if not args.detector_model or args.method == 'pure_model':
                raise ValueError('expert_measurement requires an Agent and a pinned local detector')
            from rs_hmcomm.workflow.expert_measurement_tools import ExpertMeasurementRuntime
            runtime_class = ExpertMeasurementRuntime
        if args.tool_protocol == 'collection_observation':
            if args.method == 'pure_model':
                raise ValueError('collection tool protocol requires an Agent')
            from rs_hmcomm.workflow.collection_observation import CollectionRuntime
            runtime_class = CollectionRuntime
        if args.tool_protocol == 'collection_remoteclip':
            if args.method == 'pure_model' or not args.scene_classifier_model:
                raise ValueError('collection_remoteclip requires an Agent and a pinned scene classifier')
            from rs_hmcomm.workflow.scene_collection_expert import SceneCollectionRuntime
            runtime_class = SceneCollectionRuntime
            if args.collection_route == 'all_sources':
                from rs_hmcomm.workflow.all_source_collection import AllSourceSceneCollectionRuntime
                runtime_class = AllSourceSceneCollectionRuntime
                if args.collection_actions == 'typed':
                    from rs_hmcomm.workflow.typed_scene_collection import TypedSceneCollectionRuntime
                    runtime_class = TypedSceneCollectionRuntime
    if args.detector_model and args.tool_protocol != 'expert_measurement':
        raise ValueError('detector requires explicitly reported expert_measurement protocol')
    if args.crop_classifier_model and args.tool_protocol != 'expert_measurement':
        raise ValueError('crop classifier requires explicit expert_measurement DEV protocol')
    if args.scene_classifier_model and args.tool_protocol != 'collection_remoteclip':
        raise ValueError('scene classifier requires explicit collection_remoteclip DEV protocol')
    if args.split == 'test' and not args.test_protocol:
        raise ValueError('independent test requires a previously frozen benchmark protocol')
    if args.method == 'trained_multi' and not args.checkpoint: raise ValueError('trained method needs real checkpoint')
    if args.checkpoint and args.method != 'trained_multi': raise ValueError('checkpoint must be reported as trained method')
    if args.test_protocol:
        protocol = json.loads(Path(args.test_protocol).read_text())
        if protocol['method'] != args.method or protocol['benchmark'] != args.benchmark or protocol['limit'] != args.limit:
            raise ValueError('test protocol mismatch')
    out = Path(args.output); out.mkdir(exist_ok=False)
    seed = 17; random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model_provenance = json.loads((Path(args.model)/'evors_provenance.json').read_text())
    policy = QualityWorkflowPolicy(args.model, revision=model_provenance['revision'], max_pixels=262144,
        max_new_tokens=256, paper_mechanisms=True, spatial_codec=True, compressed_tokens=64,
        stable_codec=False)
    if args.method in {'innovation_multi', 'trained_multi'}:
        # Match the joint DEV arm's pre-training codec initialization exactly.
        # Constructing a second random codec after the factory would otherwise
        # confound an eventual trained-vs-untrained comparison with initialization.
        import copy
        initial_auxiliary = copy.deepcopy(policy.auxiliary.state_dict())
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        policy.auxiliary['codec'] = StableAddressedCodec(policy.base.config.hidden_size, max_tokens=64).to(policy.device)
        policy.auxiliary.load_state_dict(initial_auxiliary, strict=False)
    policy.role_priors = args.method in {'innovation_multi', 'trained_multi'}
    if args.checkpoint: policy.load(args.checkpoint, resume_training=False)
    runtime_options = {}
    if args.collection_route == 'all_sources':
        runtime_options['slot_allocation'] = args.collection_slots
    if args.tool_protocol in {'expert_measurement', 'collection_remoteclip'}:
        from rs_hmcomm.workflow.grounding_detector import GroundingDetector
        # Loading the frozen auxiliary tool must not alter the Agent's RNG.
        python_rng = random.getstate(); cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        if args.tool_protocol == 'expert_measurement':
            runtime_options['detector'] = GroundingDetector(args.detector_model)
        if args.crop_classifier_model:
            from rs_hmcomm.workflow.remoteclip_expert import RemoteClipExpert
            runtime_options['classifier'] = RemoteClipExpert(args.crop_classifier_model)
        if args.scene_classifier_model:
            from rs_hmcomm.workflow.remoteclip_expert import RemoteClipExpert
            runtime_options['classifier'] = RemoteClipExpert(args.scene_classifier_model)
        random.setstate(python_rng); torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    snapshot_source(out)
    import shutil
    saved_runner = out/'source/scripts'/Path(__file__).name
    shutil.copyfile(__file__, saved_runner)
    source_hashes = json.loads((out/'source_hashes.json').read_text())
    source_hashes[str(saved_runner.relative_to(out/'source'))] = hashlib.sha256(saved_runner.read_bytes()).hexdigest()
    write_json(out/'source_hashes.json', source_hashes)
    write_json(out/'run.json', {'arguments': vars(args), 'model_identity': policy.identity,
        'source_sha256': hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        'runner_sha256': hashlib.sha256(saved_runner.read_bytes()).hexdigest(),
        'data_deployment_sha256': hashlib.sha256((Path(args.data)/'DATA_DEPLOYMENT.json').read_bytes()).hexdigest(),
        'weights_before': policy.parameter_hash(), 'adapted_environment': args.benchmark in {'ThinkGeo', 'Earth-Bench'},
        'official_semantic_judge_run': False, 'references_in_model': False,
        'evaluation_scope': 'text planning transfer; no visual communication or tool execution'
            if args.benchmark == 'GeoPlan' else 'image task workflow',
        'visual_communication_supported_by_protocol': args.benchmark != 'GeoPlan'})
    data = Path(args.data); results = []
    if args.benchmark == 'GeoMMBench':
        manifest = data/args.benchmark/(args.split+'.jsonl')
        rows = list(load_workflow_manifest(manifest))
        if args.limit: rows = rows[:args.limit]
        for i, (task, reference) in enumerate(rows):
            if args.method in {'pure_model', 'single_agent'}:
                runtime = IndependentRunner(policy, agent=args.method == 'single_agent', max_model_calls=96)
            else:
                runtime = workflow_class(policy, mechanisms=PaperMechanisms(
                    modes=('text',) if args.method == 'text_multi' else ('tokens', 'compressed', 'temporal'),
                    initial_global_observation=True, structured_requirement_actions=True, progress_guard=True),
                    blind_verification=args.method != 'text_multi', max_model_calls=96,
                    max_steps=12, max_tool_calls=8, max_packet_bytes=33554432)
            episode = runtime.run(task, training=False); write_json(out/f'prediction-{i:04d}.json', episode.to_dict())
            results.append({'sample_id': task.sample_id, 'stop': episode.stop_reason, 'answer': episode.answer,
                            'accuracy': task_score(episode.answer, reference) if episode.stop_reason == 'finished' else 0.,
                            'cost': episode.cost.to_dict()})
            write_json(out/'progress.json', {'evaluation': results, 'completed': False})
            print(json.dumps({'benchmark': args.benchmark, 'method': args.method, 'item': i, 'stop': episode.stop_reason}), flush=True)
    elif args.benchmark == 'GeoPlan':
        split = 'train' if args.split in {'train', 'dev'} else 'test'
        rows = [json.loads(l) for l in (data/'GeoPlan'/f'{split}-public.jsonl').read_text().splitlines()]
        refs = {r['sample_id']: r for r in [json.loads(l) for l in (data/'GeoPlan'/f'{split}-references.jsonl').read_text().splitlines()]}
        catalog = [json.loads(l) for l in (data/'GeoPlan/tool_catalog.jsonl').read_text().splitlines()]
        catalog = [{'name': t['name'], 'description': t['description'][:300]} for t in catalog]
        allowed = {t['name'] for t in catalog}
        if args.limit: rows = rows[:args.limit]
        for i, row in enumerate(rows):
            policy.begin_episode(training=False)
            context = json.dumps({'question': row['question'], 'tools': catalog})
            if args.method not in {'pure_model', 'single_agent'}:
                decomposition = policy.decide('controller', 'Decompose the task into ordered processing goals. Do not execute or invent data. '+context, (), output_format='text')
                analysis = policy.decide('knowledge', 'Identify required processing constraints; do not invent data. '+context, (), output_format='text')
                context += '\nProcessing goals: '+decomposition+'\nDomain guidance: '+analysis
            text = policy.decide('reasoning', 'Produce an ordered tool plan. Return JSON {"tool_flow":["tool_name",...]}. Use only listed tool names. '+context, (), output_format='json')
            initial_plan_text = text
            if args.method != 'pure_model':
                reviewer = 'reasoning' if args.method == 'single_agent' else 'verification'
                review = policy.decide(reviewer, 'Check this proposed plan for dependencies, missing steps and irrelevant tools. Give concrete corrections; no reference plan is available. '+context+'\nProposed plan: '+text, (), output_format='text')
                text = policy.decide('reasoning', 'Return the checked final plan as JSON {"tool_flow":["tool_name",...]}. '+context+'\nPrevious plan: '+text+'\nReview: '+review, (), output_format='json')
            transaction_events = []
            try:
                if args.planning_protocol == 'transaction':
                    from rs_hmcomm.workflow.planning_transaction import commit_plan
                    flow, transaction_events = commit_plan(initial_plan_text,
                        text if args.method != 'pure_model' else None, allowed)
                    if flow is None: raise ValueError('no valid sampled plan')
                else:
                    flow = strict_json(text)['tool_flow']
                if not isinstance(flow, list) or any(t not in allowed for t in flow): raise ValueError('unknown plan tool')
                stop = 'planned'; scores = plan_scores(flow, refs[row['sample_id']]['tool_flow'])
            except (ValueError, KeyError, TypeError): flow = []; stop = 'invalid_plan'; scores = plan_scores([], refs[row['sample_id']]['tool_flow'])
            write_json(out/f'prediction-{i:04d}.json', {'task': row, 'decisions': [r.descriptor() for r in policy.records],
                                                       'plan': flow, 'execution_performed': False,
                                                       'planning_transaction': transaction_events})
            results.append({'sample_id': row['sample_id'], 'stop': stop, **scores,
                'cost': {'model_calls': len(policy.records), 'tool_calls': 0,
                         'model_latency_ms': sum(r.latency_ms for r in policy.records),
                         'generated_tokens': sum(r.output_ids.numel() for r in policy.records)}})
            write_json(out/'progress.json', {'evaluation': results, 'completed': False})
    else:
        rows = [json.loads(l) for l in (data/args.benchmark/'public.jsonl').read_text().splitlines()]
        refs = {r['sample_id']: r for r in [json.loads(l) for l in (data/args.benchmark/'references.jsonl').read_text().splitlines()]}
        if args.benchmark == 'ThinkGeo': rows = [r for r in rows if r['split'] == args.split]
        elif args.split != 'dev': raise ValueError('Earth-Bench development inventory needs a scene-disjoint split before test')
        if args.modality != 'all': rows = [r for r in rows if r['modality'] == args.modality]
        if args.limit: rows = rows[:args.limit]
        for i, row in enumerate(rows):
            if not row.get('assets_downloaded'):
                results.append({'sample_id': row['sample_id'], 'stop': 'assets_unavailable', 'completed': False,
                                'exact_dev': 0., 'token_f1_dev': 0.})
            elif args.method == 'pure_model':
                from PIL import Image
                frames = []
                try:
                    if len(row['assets']) > 16: raise ValueError('pure baseline maximum 16 views')
                    for j, (name, filename) in enumerate(row['assets'].items()):
                        with Image.open(filename) as im:
                            if im.mode not in {'RGB', 'RGBA', 'L', 'P'}: raise ValueError('native scientific raster requires declared pure-model input adaptation')
                            frames.append(ImageFrame(f'view{j}', 'unspecified', im.convert('RGB').copy(), source=name))
                    task = WorkflowTask(row['sample_id'], row['question'], tuple(frames), row['sample_id'])
                    episode = IndependentRunner(policy).run(task, training=False)
                    write_json(out/f'prediction-{i:04d}.json', episode.to_dict())
                    answers = refs[row['sample_id']]['answers']; valid = episode.stop_reason == 'finished'
                    results.append({'sample_id': row['sample_id'], 'stop': episode.stop_reason, 'answer': episode.answer,
                        'exact_dev': max((task_score(episode.answer, a) for a in answers), default=0.) if valid else 0.,
                        'token_f1_dev': max((token_f1(episode.answer, a) for a in answers), default=0.) if valid else 0.,
                        'cost': episode.cost.to_dict(), 'official_judge_pending': True})
                except (ValueError, OSError) as error:
                    results.append({'sample_id': row['sample_id'], 'stop': 'pure_input_adapter_pending',
                                    'reason': str(error), 'completed': False, 'exact_dev': 0., 'token_f1_dev': 0.})
            else:
                runtime = runtime_class(policy, multi=args.method != 'single_agent',
                    communication='text' if args.method in {'single_agent', 'text_multi'} else 'compressed',
                    max_calls=32, max_tools=16, output=out/f'tools-{i:04d}', **runtime_options)
                episode = runtime.run(row, training=False)
                write_json(out/f'prediction-{i:04d}.json', episode.to_dict())
                answers = refs[row['sample_id']]['answers']; valid = episode.stop_reason == 'finished'
                results.append({'sample_id': row['sample_id'], 'stop': episode.stop_reason, 'answer': episode.answer,
                    'exact_dev': max((task_score(episode.answer, a) for a in answers), default=0.) if valid else 0.,
                    'token_f1_dev': max((token_f1(episode.answer, a) for a in answers), default=0.) if valid else 0.,
                    'cost': episode.cost.to_dict(), 'official_judge_pending': True})
            write_json(out/'progress.json', {'evaluation': results, 'completed': False})
            print(json.dumps({'benchmark': args.benchmark, 'method': args.method, 'item': i, 'stop': results[-1]['stop']}), flush=True)
    frozen = json.loads((out/'run.json').read_text())['weights_before']
    if frozen != policy.parameter_hash(): raise ValueError('benchmark inference modified weights')
    write_json(out/'summary.json', {'evaluation': results, 'completed': True,
        'official_semantic_judge_run': False, 'parameters_frozen': True, 'test_selects_checkpoint': False})

if __name__ == '__main__': main()
