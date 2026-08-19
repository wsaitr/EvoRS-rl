#!/usr/bin/env python3
import argparse
from rs_hmcomm.backends.device import resolve_device

p = argparse.ArgumentParser()
p.add_argument("--device", default="auto", choices=["auto", "cuda", "npu", "ascend", "cpu"])
args = p.parse_args()
print(resolve_device(args.device))
