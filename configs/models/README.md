# Local model runtime

`qwen38_27b_q4_local.yaml` is the immutable provenance record for the local
Phase 1 VLM gate. It pins the Hugging Face repository revision, both GGUF
artifacts, the llama.cpp release, the two official Windows CUDA archives, and
their exact sizes and SHA-256 values. It also records the GGUF repository's
`.src_sha`, which pins the upstream `Qwen/Qwen3.8-27B` source revision.

Large artifacts remain under the Git-ignored `.models/` directory. The model is
not copied into a container, virtual environment, or another archive.

## Validate and run

From the repository root in PowerShell:

```powershell
& .\scripts\start_qwen38_local.ps1 -Action Validate
& .\scripts\start_qwen38_local.ps1 -Action Start
& .\scripts\start_qwen38_local.ps1 -Action Status
```

The server is started in a hidden window, listens only on
`127.0.0.1:18080`, and publishes the model alias
`evors/qwen3.8-27b-q4`. Request concurrency is one. Token prefill uses a
512/128 logical/physical batch; these are token counts, not request counts.
GPU layers are fitted automatically while retaining a 1536 MiB VRAM margin.
Browser access is restricted to localhost origins and CORS credentials are
disabled; the endpoint is not intended to be exposed to another host.
Server-side reasoning is disabled; any residual thought tags are extracted away
from `message.content` so the Agent's strict JSON action protocol remains valid.
Override the layer policy only after measuring available VRAM:

```powershell
& .\scripts\start_qwen38_local.ps1 -Action Start -GpuLayers "4"
```

After `/health` reports ready, run the real-model integration gate:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_phase1_generalist.py --config configs/experiments/phase1_qwen38_local.yaml --overwrite
```

Stop only the process recorded and ownership-checked by the script:

```powershell
& .\scripts\start_qwen38_local.ps1 -Action Stop
```

`Validate` hashes the complete 17.67 GiB Q4 file, so it takes noticeably longer
than checking file names. The pinned SHA-256 values come from Hugging Face LFS
metadata for `ggml-org/Qwen3.8-27B-GGUF` and GitHub release-asset digests for
llama.cpp `b10520`; validation recomputes all four files locally and verifies
every extracted executable/DLL against the contents of those pinned archives.
