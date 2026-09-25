# EvoRS-Comm ModelArts Image Release

## Result

The EvoRS-Comm Phase 0–5 experiment image was published to Huawei Cloud SWR
and registered in ModelArts successfully.

- Region: `cn-southwest-2`
- SWR image: `swr.cn-southwest-2.myhuaweicloud.com/evors-comm/training:phase5-20260919-d52590e8c17b`
- Manifest digest: `sha256:b1d866ae780b2685e15baf641a5f7bb25cfaf32dc674aebcb2581b17726d9779`
- ModelArts image ID: `312ebef9-5fc6-4225-a0aa-a48ead950d39`
- ModelArts status: `ACTIVE`
- Architecture/flavor: `aarch64` / `ASCEND_SNT9B`
- Service type: `TRAIN`
- Visibility: `PRIVATE`
- Registered compressed size: `6,595,450,594` bytes

## Runtime contract

- Default user: `1000:1000`
- Entry point: `/usr/local/bin/evors-modelarts`
- Default command: `verify`
- Default output: `/home/ma-user/evors-output`
- For persistent results, set `EVORS_OUTPUT_DIR` to a directory on the mounted
  `lws2` parallel file system.

The default `verify` command requires an available Ascend NPU and records the
observed torch/torch_npu/verl environment. The `routing` command runs the
existing Phase 5 routing-policy mechanism gate, which is intentionally CPU-only.
It does not claim four-card NPU utilization.

## Packaging evidence

- Project content SHA-256:
  `d52590e8c17bc53f82173a2a72713eff888ea1b64f7b327c688a75113eab9abf`
- Application layer digest:
  `sha256:6f79efb0dc9f08e101f199ec7003ec5c976528ff963e08ee0096587450332284`
- Application layer diff ID:
  `sha256:4adb3647d1d2963fab1de099c3d7aa9cfff344cbce4bdf96182d8eaf94fdf8e5`
- Application layer: 13,121 files, 150,536,874 compressed bytes
- Base: 57 SWR-resident layers reused without embedding credentials
- Remote OCI verification: all checks passed
- Secret scan: no Huawei access key or secret key in layer inputs
- Accelerator safety: PyPI torch was excluded; the CANN-coupled base torch and
  torch_npu installation remains authoritative

## Tests

- ModelArts image contract: 3 passed
- Deployment-safe repository suite: 321 passed, 3 skipped
- Excluded from this packaging verification: 3 heavyweight local
  `qwen_native_latent` tests that load the workstation model; they are unrelated
  to image structure or cloud registration.

## Evidence files

- `swr_publish.json`: immutable SWR tag, digest, layer sizes, and provenance
- `oci_remote_verification.json`: remote manifest/config contract checks
- `modelarts_registration_initial.json`: registration response
- `modelarts_registration_verified.json`: independent post-registration lookup
