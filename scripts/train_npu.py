#!/usr/bin/env python3
"""
Ascend 910B NPU training script for HM-MAGRPO.

Usage:
  # Single NPU
  python scripts/train_npu.py --config configs/experiments/stage2_pilot.yaml

  # Multi-NPU distributed (via torchrun)
  torchrun --nproc_per_node=8 scripts/train_npu.py --config configs/experiments/stage2_pilot.yaml --distributed
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_device(distributed: bool = False) -> tuple[str, int, int]:
    """Setup NPU device, return (device_str, rank, world_size)."""
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as e:
        logger.error(f"Failed to import torch/torch_npu: {e}")
        logger.info("Falling back to CPU for testing...")
        return "cpu", 0, 1

    if not torch.npu.is_available():
        logger.warning("NPU not available, falling back to CPU")
        return "cpu", 0, 1

    if distributed:
        # Initialize distributed process group (HCCL for Ascend)
        torch.distributed.init_process_group(backend="hccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"npu:{local_rank}"
        torch.npu.set_device(device)
        logger.info(f"Rank {rank}/{world_size}, device={device}")
    else:
        rank = 0
        world_size = 1
        device = "npu:0"
        torch.npu.set_device(device)
        logger.info(f"Single NPU mode, device={device}")

    return device, rank, world_size


def load_obs_data(obs_prefix: str, cache_dir: str = "./data/cache", limit: int = 100):
    """Load training data from OBS."""
    from rs_hmcomm.data.obs_loader import OBSDataLoader, OBSConfig
    from rs_hmcomm.data.cache import DataCache  # noqa: F401

    config = OBSConfig.from_env()
    if not config.is_configured:
        logger.warning("OBS not configured. Using mock data for testing.")
        return None

    loader = OBSDataLoader(config, cache_dir=cache_dir)
    logger.info(f"Downloading data from obs://{config.bucket}/{obs_prefix}")
    paths = loader.download_prefix(obs_prefix, max_keys=limit)
    logger.info(f"Downloaded {len(paths)} files")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="NPU Training for HM-MAGRPO")
    parser.add_argument("--config", type=str, default="configs/experiments/stage2_pilot.yaml")
    parser.add_argument("--distributed", action="store_true", help="Enable multi-NPU distributed training")
    parser.add_argument("--obs-prefix", type=str, default="vrsbench/train/", help="OBS data prefix")
    parser.add_argument("--cache-dir", type=str, default="./data/cache")
    parser.add_argument("--output", type=str, default="outputs/npu_training")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    device, rank, world_size = setup_device(args.distributed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== EvoRS-Comm NPU Training ===")
    logger.info(f"Device: {device}")
    logger.info(f"Rank: {rank}/{world_size}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")

    # Load config
    from rs_hmcomm.rl.trainer_config import TrainerConfig

    config = TrainerConfig.from_yaml(args.config)
    config.train_samples = args.samples
    config.max_epochs = args.epochs

    # Try loading OBS data
    data_paths = load_obs_data(args.obs_prefix, args.cache_dir, limit=args.samples)

    # Create training dataset
    if data_paths is not None:
        logger.info(f"Training with {len(data_paths)} OBS samples")
        dataset = [{"image": p, "question": "mock", "answer": "mock"} for p in data_paths]
    else:
        logger.info("Using mock data (OBS not configured)")
        questions = [
            "How many aircraft are visible?",
            "What type of vehicles are in the parking area?",
            "Describe the buildings in the central region.",
            "Count the ships in the harbor.",
            "What is the land use pattern?",
        ]
        dataset = [
            {"image": {"mock": True, "id": i}, "question": questions[i % len(questions)], "answer": "mock"}
            for i in range(min(args.samples, 100))
        ]

    # Create trainer
    from rs_hmcomm.rl.hmagrpo import HMMAGRPOTrainer, HMMAGRPOConfig
    from rs_hmcomm.agents import GlobalAgent, LocalAgent, HierarchyAgent, VerifierAgent, ResidualAgent
    from rs_hmcomm.backends.mock import MockBackend
    from rs_hmcomm.controllers import RuleController
    from rs_hmcomm.orchestrator import MultiAgentOrchestrator

    def make_orchestrator():
        backend = MockBackend()
        agents = {
            "global": GlobalAgent(backend),
            "local": LocalAgent(backend),
            "hierarchy": HierarchyAgent(backend),
            "verifier": VerifierAgent(backend),
            "residual": ResidualAgent(backend),
        }
        return MultiAgentOrchestrator(agents, RuleController(max_steps=4))

    hm_config = HMMAGRPOConfig(
        group_size=config.group_size,
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
    )
    trainer = HMMAGRPOTrainer(hm_config)
    trainer.setup_rollout_generator(make_orchestrator)

    # Training loop
    start_time = time.time()
    for epoch in range(config.max_epochs):
        epoch_start = time.time()
        result = trainer.train_epoch(dataset)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{config.max_epochs} | "
            f"Loss: {result['epoch_loss']:.6f} | "
            f"Samples: {result['n_samples']} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.json"
        ckpt_data = {
            "epoch": epoch + 1,
            "loss": result["epoch_loss"],
            "device": device,
            "rank": rank,
            "world_size": world_size,
            "timestamp": time.time(),
        }
        ckpt_path.write_text(json.dumps(ckpt_data, indent=2))

    total_time = time.time() - start_time

    # Save final results
    final_result = {
        "total_time_s": round(total_time, 1),
        "epochs": config.max_epochs,
        "samples_per_epoch": len(dataset),
        "device": device,
        "world_size": world_size,
        "final_loss": trainer.training_history[-1]["total_loss"] if trainer.training_history else 0.0,
        "training_history": trainer.training_history,
    }

    result_path = output_dir / "training_result.json"
    result_path.write_text(json.dumps(final_result, indent=2, ensure_ascii=False))

    logger.info(f"Training complete! Total time: {total_time:.1f}s")
    logger.info(f"Results saved to {result_path}")

    # Cleanup distributed
    if args.distributed and world_size > 1:
        import torch

        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
