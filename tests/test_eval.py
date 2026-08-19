from rs_hmcomm.eval.baseline import SingleVLMBaseline, TextOnlyMASBaseline, StaticTreeMASBaseline, BaselineResult
from rs_hmcomm.eval.metrics import EvaluationMetrics, compute_metrics
from rs_hmcomm.backends.mock import MockBackend


def test_single_vlm_baseline():
    backend = MockBackend()
    baseline = SingleVLMBaseline(backend)
    result = baseline.run({"mock": True}, "How many aircraft?")
    assert result.method == "single_vlm"
    assert result.agent_calls == 1
    assert len(result.answer) > 0


def test_text_only_mas_baseline():
    backend = MockBackend()
    baseline = TextOnlyMASBaseline(backend)
    result = baseline.run({"mock": True}, "How many aircraft?")
    assert result.method == "text_only_mas"
    assert result.agent_calls > 1


def test_static_tree_mas_baseline():
    backend = MockBackend()
    baseline = StaticTreeMASBaseline(backend)
    result = baseline.run({"mock": True}, "How many aircraft?")
    assert result.method == "static_tree_mas"


def test_compute_metrics():
    results = [
        {"task_score": 1.0, "agent_calls": 3, "text_tokens": 50, "struct_nodes": 2,
         "latent_bytes": 0, "communication_cost": 5.0, "task_type": "counting", "correct": True},
        {"task_score": 0.5, "agent_calls": 2, "text_tokens": 30, "struct_nodes": 1,
         "latent_bytes": 0, "communication_cost": 3.0, "task_type": "counting", "correct": False},
    ]
    metrics = compute_metrics(results)
    assert metrics.accuracy == 0.5
    assert metrics.avg_task_score == 0.75
    assert metrics.avg_agent_calls == 2.5
    assert metrics.score_per_token > 0


def test_compute_metrics_empty():
    metrics = compute_metrics([])
    assert metrics.accuracy == 0.0


def test_evaluation_metrics_summary():
    m = EvaluationMetrics(accuracy=0.85, avg_task_score=0.9)
    s = m.summary()
    assert "0.85" in s
