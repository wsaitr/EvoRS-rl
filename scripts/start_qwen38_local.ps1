[CmdletBinding()]
param(
    [ValidateSet("Validate", "Start", "Status", "Stop")]
    [string]$Action = "Start",

    [ValidatePattern("^(auto|all|[0-9]+)$")]
    [string]$GpuLayers = "auto"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$modelPath = Join-Path $repoRoot ".models\qwen3.8-27b-q4\Qwen3.8-27B-Q4_K_M.gguf"
$mmprojPath = Join-Path $repoRoot ".models\qwen3.8-27b-q4\mmproj-Qwen3.8-27B-Q8_0.gguf"
$runtimeRoot = Join-Path $repoRoot ".models\llama.cpp\b10520"
$serverPath = Join-Path $runtimeRoot "bin\llama-server.exe"
$serverArchive = Join-Path $runtimeRoot "downloads\llama-b10520-bin-win-cuda-12.4-x64.zip"
$cudaArchive = Join-Path $runtimeRoot "downloads\cudart-llama-bin-win-cuda-12.4-x64.zip"
$stateDirectory = Join-Path $runtimeRoot "state"
$pidPath = Join-Path $stateDirectory "qwen38-18080.pid"
$stdoutPath = Join-Path $stateDirectory "qwen38-18080.stdout.log"
$stderrPath = Join-Path $stateDirectory "qwen38-18080.stderr.log"
$serverHost = "127.0.0.1"
$serverPort = 18080
$modelAlias = "evors/qwen3.8-27b-q4"
$runtimeArchives = @($serverArchive, $cudaArchive)

$expectedArtifacts = @(
    [pscustomobject]@{
        Role = "language_model"
        Path = $modelPath
        Bytes = [int64]18973870432
        Sha256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
    },
    [pscustomobject]@{
        Role = "multimodal_projector"
        Path = $mmprojPath
        Bytes = [int64]629247008
        Sha256 = "2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb"
    },
    [pscustomobject]@{
        Role = "llama_cpp_cuda_binary_archive"
        Path = $serverArchive
        Bytes = [int64]250890269
        Sha256 = "ea9c64786333f8052056fb3735d3edb95d4b9a5e4c816390d81c45ec8e59780f"
    },
    [pscustomobject]@{
        Role = "llama_cpp_cuda_runtime_archive"
        Path = $cudaArchive
        Bytes = [int64]391443627
        Sha256 = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
    }
)

function Get-StreamSha256 {
    param([Parameter(Mandatory = $true)][IO.Stream]$Stream)

    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha256.ComputeHash($Stream))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

function Test-ExtractedRuntimeMatchesArchives {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $expectedFiles = @{}
    foreach ($archivePath in $runtimeArchives) {
        $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
        try {
            foreach ($entry in $archive.Entries) {
                if ([string]::IsNullOrEmpty($entry.Name)) {
                    continue
                }
                if ($entry.FullName -ne $entry.Name) {
                    throw "Pinned runtime archive contains an unexpected nested path."
                }
                if ($expectedFiles.ContainsKey($entry.Name)) {
                    throw "Pinned runtime archives contain a duplicate filename."
                }
                $entryStream = $entry.Open()
                try {
                    $expectedFiles[$entry.Name] = [pscustomobject]@{
                        bytes = [int64]$entry.Length
                        sha256 = Get-StreamSha256 -Stream $entryStream
                    }
                }
                finally {
                    $entryStream.Dispose()
                }
            }
        }
        finally {
            $archive.Dispose()
        }
    }

    $binDirectory = Split-Path -Parent $serverPath
    foreach ($filename in $expectedFiles.Keys) {
        $extractedPath = Join-Path $binDirectory $filename
        if (-not (Test-Path -LiteralPath $extractedPath -PathType Leaf)) {
            throw "Extracted pinned runtime file is missing."
        }
        $item = Get-Item -LiteralPath $extractedPath
        $expected = $expectedFiles[$filename]
        if ($item.Length -ne $expected.bytes) {
            throw "Extracted pinned runtime file has an unexpected byte size."
        }
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $extractedPath).Hash.ToLowerInvariant()
        if ($actualHash -ne $expected.sha256) {
            throw "Extracted pinned runtime file has an unexpected SHA-256 digest."
        }
    }

    $unexpectedFiles = @(
        Get-ChildItem -LiteralPath $binDirectory -File |
            Where-Object { -not $expectedFiles.ContainsKey($_.Name) }
    )
    if ($unexpectedFiles.Count -gt 0) {
        throw "Extracted pinned runtime directory contains unexpected files."
    }

    return [pscustomobject]@{
        verified = $true
        file_count = $expectedFiles.Count
    }
}

function Test-PinnedArtifacts {
    $results = foreach ($artifact in $expectedArtifacts) {
        if (-not (Test-Path -LiteralPath $artifact.Path -PathType Leaf)) {
            throw "Missing pinned artifact for role '$($artifact.Role)'."
        }
        $item = Get-Item -LiteralPath $artifact.Path
        if ($item.Length -ne $artifact.Bytes) {
            throw "Byte-size mismatch for pinned artifact role '$($artifact.Role)'."
        }
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact.Path).Hash.ToLowerInvariant()
        if ($actualHash -ne $artifact.Sha256) {
            throw "SHA-256 mismatch for pinned artifact role '$($artifact.Role)'."
        }
        [pscustomobject]@{
            role = $artifact.Role
            bytes = $item.Length
            sha256 = $actualHash
            verified = $true
        }
    }
    if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
        throw "Pinned llama-server executable is missing."
    }
    $null = Test-ExtractedRuntimeMatchesArchives
    return $results
}

function Get-OwnedServerProcess {
    if (-not (Test-Path -LiteralPath $pidPath -PathType Leaf)) {
        return $null
    }
    $rawPid = [IO.File]::ReadAllText($pidPath).Trim()
    $parsedPid = 0
    if (-not [int]::TryParse($rawPid, [ref]$parsedPid)) {
        throw "Server PID file is invalid."
    }
    $process = Get-Process -Id $parsedPid -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return $null
    }
    $actualExecutable = [IO.Path]::GetFullPath($process.Path)
    $expectedExecutable = [IO.Path]::GetFullPath($serverPath)
    if (-not $actualExecutable.Equals($expectedExecutable, [StringComparison]::OrdinalIgnoreCase)) {
        throw "PID file points to a process not owned by this local model runtime."
    }
    return $process
}

function Test-ServerHealth {
    try {
        $health = Invoke-RestMethod -Uri "http://${serverHost}:${serverPort}/health" -TimeoutSec 2
        return [pscustomobject]@{
            reachable = $true
            status = $health.status
        }
    }
    catch {
        return [pscustomobject]@{
            reachable = $false
            status = "unavailable"
        }
    }
}

switch ($Action) {
    "Validate" {
        $validation = Test-PinnedArtifacts
        [pscustomobject]@{
            status = "verified"
            artifact_count = $validation.Count
            artifacts = $validation
        } | ConvertTo-Json -Depth 4
        break
    }
    "Status" {
        $ownedProcess = Get-OwnedServerProcess
        $health = Test-ServerHealth
        [pscustomobject]@{
            running = $null -ne $ownedProcess
            pid = if ($null -ne $ownedProcess) { $ownedProcess.Id } else { $null }
            endpoint = "http://${serverHost}:${serverPort}"
            alias = $modelAlias
            health_reachable = $health.reachable
            health_status = $health.status
        } | ConvertTo-Json -Compress
        break
    }
    "Stop" {
        $ownedProcess = Get-OwnedServerProcess
        if ($null -eq $ownedProcess) {
            [pscustomobject]@{ stopped = $false; reason = "not_running" } | ConvertTo-Json -Compress
            break
        }
        Stop-Process -Id $ownedProcess.Id
        $null = $ownedProcess.WaitForExit(30000)
        if (Test-Path -LiteralPath $pidPath -PathType Leaf) {
            Remove-Item -LiteralPath $pidPath
        }
        [pscustomobject]@{ stopped = $true; pid = $ownedProcess.Id } | ConvertTo-Json -Compress
        break
    }
    "Start" {
        $null = Test-PinnedArtifacts
        $ownedProcess = Get-OwnedServerProcess
        if ($null -ne $ownedProcess) {
            [pscustomobject]@{
                started = $false
                reason = "already_running"
                pid = $ownedProcess.Id
                endpoint = "http://${serverHost}:${serverPort}"
            } | ConvertTo-Json -Compress
            break
        }
        New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
        $arguments = @(
            "--model", $modelPath,
            "--mmproj", $mmprojPath,
            "--host", $serverHost,
            "--port", "$serverPort",
            "--alias", $modelAlias,
            "--cors-origins", "localhost",
            "--no-cors-credentials",
            "--ctx-size", "4096",
            "--batch-size", "512",
            "--ubatch-size", "128",
            "--parallel", "1",
            "--n-gpu-layers", $GpuLayers,
            "--fit", "on",
            "--fit-target", "1536",
            "--mmproj-offload",
            "--reasoning", "off",
            "--reasoning-format", "deepseek",
            "--no-reasoning-preserve",
            "--no-webui",
            "--embeddings",
            "--pooling", "last"
        )
        $process = Start-Process `
            -FilePath $serverPath `
            -ArgumentList $arguments `
            -WorkingDirectory (Split-Path -Parent $serverPath) `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru
        Start-Sleep -Milliseconds 1500
        if ($process.HasExited) {
            throw "llama-server exited during startup; inspect the state logs."
        }
        [IO.File]::WriteAllText($pidPath, "$($process.Id)`n")
        [pscustomobject]@{
            started = $true
            pid = $process.Id
            endpoint = "http://${serverHost}:${serverPort}"
            alias = $modelAlias
            gpu_layers = $GpuLayers
            context_size = 4096
            batch_size = 512
            ubatch_size = 128
            parallel = 1
            stdout_log = $stdoutPath
            stderr_log = $stderrPath
        } | ConvertTo-Json -Compress
        break
    }
}
