# EvoRS-Comm ModelArts Ascend Image

This image is the production packaging path for the four-card Ascend SNT9B
environment.  It is separate from `docker/Dockerfile.ascend`, which remains the
hot-mounted development image.

## Registered release (2026-09-19)

- ModelArts image ID: `312ebef9-5fc6-4225-a0aa-a48ead950d39`
- Status: `ACTIVE`
- SWR image:
  `swr.cn-southwest-2.myhuaweicloud.com/evors-comm/training:phase5-20260919-d52590e8c17b`
- Digest:
  `sha256:b1d866ae780b2685e15baf641a5f7bb25cfaf32dc674aebcb2581b17726d9779`

In ModelArts `cn-southwest-2`, select the registered image above, choose the
four-card `ASCEND_SNT9B` training flavor, and mount `lws2`.  Keep the default
command for the first environment verification.  Set, for example,
`EVORS_OUTPUT_DIR=/mnt/lws2/evors/runs/image-check-20260919` so the report
persists after the job exits.

## Runtime contract

- Architecture: `linux/arm64`
- ModelArts flavor: `ASCEND_SNT9B`
- Default user: UID/GID `1000:1000`
- Source root: `/opt/evors`
- Default output: `/home/ma-user/evors-output`
- Default command: `verify`, which requires a visible NPU and writes
  `ascend-environment.json`
- Other built-in commands: `routing`, `test`, and `shell`
- Any other first argument is executed as an explicit command.

Examples inside ModelArts:

```bash
# Honest NPU environment gate (default)
evors-modelarts verify

# Existing Phase 5 CPU controller mechanism gate
EVORS_OUTPUT_DIR=/mnt/lws2/evors/runs/phase5 evors-modelarts routing

# Override the entry point for a later four-card VLM/RL launcher
evors-modelarts python scripts/my_distributed_training.py --config config.yaml
```

The current Phase 5 routing policy is CPU-only.  Selecting four Ascend cards
does not make that small policy use the NPUs; NPU utilization begins when the
VLM/token-policy training command is added.

## Offline dependencies

Run `scripts/prepare_modelarts_vendor.py` before building.  It downloads
CPython 3.10 ARM64 wheels and expands them into `.modelarts/vendor`.  The script
explicitly excludes torch, torch_npu, torchvision, torchaudio, and triton so it
cannot replace the CANN-coupled framework installed by the base image.

## Security

The build context excludes `.env`, local model caches, datasets, historical
outputs, cloud state, and temporary files.  Huawei credentials are used only by
the local release process and are never copied into an image layer.

Experiments use the mounted `lws2` parallel file system for data and outputs;
the runtime does not require OBS credentials or an OBS SDK inside the image.
