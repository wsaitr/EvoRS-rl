#!/usr/bin/env python3
"""Stage 2 HM-MAGRPO training script."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from rs_hmcomm.agents import GlobalAgent, LocalAgent, HierarchyAgent, VerifierAgent, ResidualAgent
from rs_hmcomm.backends.mock import MockBackend
from rs_hmcomm.controllers import RuleController
from rs_hmcomm.orchestrator import MultiAgentOrchestrator
from rs_hmcomm.rl.hmagrpo import HMMAGRPOTrainer, HMMAGRPOConfig
from rs_hmcomm.rl.trainer_config import TrainerConfig


def make_orchestrator():
    """Factory for creating orchestrators."""
    backend = MockBackend()
    agents = {
        "global": GlobalAgent(backend),
        "local": LocalAgent(backend),
        "hierarchy": HierarchyAgent(backend),
        "verifier": VerifierAgent(backend),
        "residual": ResidualAgent(backend),
    }
    return MultiAgentOrchestrator(agents, RuleController(max_steps=4))


def main():
    parser = argparse.ArgumentParser(description="Stage 2 HM-MAGRPO Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="outputs/stage2", help="Output directory")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = TrainerConfig.from_yaml(args.config)
    else:
        config = TrainerConfig(
            train_samples=args.samples,
            max_epochs=args.epochs,
            output_dir=args.output,
        )

    print(f"=== HM-MAGRPO Training ===")
    print(f"Algorithm: {config.algorithm}")
    print(f"Samples: {config.train_samples}")
    print(f"Epochs: {config.max_epochs}")
    print(f"Seeds: {config.seeds}")
    print(f"Output: {config.output_dir}")
    print()

    # Create trainer
    hm_config = HMMAGRPOConfig(
        group_size=config.group_size,
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
    )
    trainer = HMMAGRPOTrainer(hm_config)
    trainer.setup_rollout_generator(make_orchestrator)

    # Generate mock dataset for pilot
    dataset = []
    questions = [
        "How many aircraft are visible?",
        "What type of vehicles are in the parking area?",
        "Describe the buildings in the central region.",
        "Count the ships in the harbor.",
        "What is the land use pattern in this image?",
    ]
    for i in range(min(config.train_samples, 20)):
        dataset.append({
            "image": {"mock": True, "id": i},
            "question": questions[i % len(questions)],
            "answer": "mock_answer",
        })

    # Training loop
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.max_epochs):
        print(f"--- Epoch {epoch + 1}/{config.max_epochs} ---")
        result = trainer.train_epoch(dataset)
        print(f"  Loss: {result['epoch_loss']:.6f}")
        print(f"  Samples: {result['n_samples']}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(trainer.training_history, f, indent=2, ensure_ascii=False)
    print(f"\nTraining history saved to {history_path}")
    print("Done!")


if __name__ == "__main__":
    main()
