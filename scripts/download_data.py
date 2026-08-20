#!/usr/bin/env python3
"""
Download training/eval data from Huawei Cloud OBS.

Usage:
  # Set OBS credentials
  export OBS_ENDPOINT="obs.cn-north-4.myhuaweicloud.com"
  export OBS_BUCKET="evors-data"
  export OBS_ACCESS_KEY_ID="your-ak"
  export OBS_SECRET_ACCESS_KEY="your-sk"

  # Download all datasets
  python scripts/download_data.py --all

  # Download specific dataset
  python scripts/download_data.py --dataset vrsbench --split train --limit 500
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS = {
    "vrsbench": {"prefix": "vrsbench/", "description": "VRSBench: 29,614 RS images with VQA"},
    "choice": {"prefix": "choice/", "description": "CHOICE: 10,507 perception/reasoning questions"},
    "xlrs": {"prefix": "xlrs-bench/", "description": "XLRS-Bench: ultra-high-res RS imagery"},
    "geobench": {"prefix": "geobench-vlm/", "description": "GEO-Bench-VLM: 31 fine-grained tasks"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download data from OBS")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Dataset to download")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Max files to download")
    parser.add_argument("--cache-dir", default="./data/cache")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name, info in DATASETS.items():
            print(f"  {name:12s} - {info['description']}")
        return

    from rs_hmcomm.data.obs_loader import OBSDataLoader, OBSConfig

    config = OBSConfig.from_env()
    if not config.is_configured:
        logger.error("OBS not configured. Set environment variables:")
        logger.error("  OBS_ENDPOINT, OBS_BUCKET, OBS_ACCESS_KEY_ID, OBS_SECRET_ACCESS_KEY")
        return

    loader = OBSDataLoader(config, cache_dir=args.cache_dir)

    if args.all:
        for name, info in DATASETS.items():
            prefix = f"{info['prefix']}{args.split}/"
            logger.info(f"Downloading {name} ({info['description']})")
            try:
                paths = loader.download_prefix(prefix, max_keys=args.limit or 10000)
                logger.info(f"  Downloaded {len(paths)} files")
            except Exception as e:
                logger.error(f"  Failed: {e}")
    elif args.dataset:
        info = DATASETS[args.dataset]
        prefix = f"{info['prefix']}{args.split}/"
        logger.info(f"Downloading {args.dataset} from {prefix}")
        paths = loader.download_prefix(prefix, max_keys=args.limit or 10000)
        logger.info(f"Downloaded {len(paths)} files to {args.cache_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
