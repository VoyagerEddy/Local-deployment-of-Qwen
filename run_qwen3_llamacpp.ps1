$ErrorActionPreference = "Stop"

$env:HF_HOME = "D:\QwenModels\hf_cache"
$env:TEMP = "D:\QwenTemp"
$env:TMP = "D:\QwenTemp"
New-Item -ItemType Directory -Force $env:HF_HOME, $env:TEMP | Out-Null

$apiModel = $env:QWEN3_GGUF_MODEL
if (-not $apiModel) {
    $apiModel = "qwen3-omni-30b-a3b-instruct-q4km"
}

$hfRepo = $env:QWEN3_GGUF_HF_REPO
if (-not $hfRepo) {
    $hfRepo = "ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF:Q4_K_M"
}

$ctx = $env:QWEN3_GGUF_CONTEXT
if (-not $ctx) { $ctx = "768" }

$cacheTypeK = $env:QWEN3_GGUF_CACHE_TYPE_K
if (-not $cacheTypeK) { $cacheTypeK = "q8_0" }

$cacheTypeV = $env:QWEN3_GGUF_CACHE_TYPE_V
if (-not $cacheTypeV) { $cacheTypeV = "q8_0" }

$flashAttn = $env:QWEN3_GGUF_FLASH_ATTN
if (-not $flashAttn) { $flashAttn = "on" }

$mlock = $env:QWEN3_GGUF_MLOCK
if (-not $mlock) { $mlock = "off" }

$parallel = $env:QWEN3_GGUF_PARALLEL
if (-not $parallel) { $parallel = "1" }

$batchSize = $env:QWEN3_GGUF_BATCH_SIZE
if (-not $batchSize) { $batchSize = "256" }

$ubatchSize = $env:QWEN3_GGUF_UBATCH_SIZE
if (-not $ubatchSize) { $ubatchSize = "64" }

$promptCacheRamMb = $env:QWEN3_GGUF_PROMPT_CACHE_RAM_MB
if (-not $promptCacheRamMb) { $promptCacheRamMb = "0" }

$promptCache = $env:QWEN3_GGUF_PROMPT_CACHE
if (-not $promptCache) { $promptCache = "off" }

$ctxCheckpoints = $env:QWEN3_GGUF_CTX_CHECKPOINTS
if (-not $ctxCheckpoints) { $ctxCheckpoints = "0" }

$minFreeRamBeforeLaunchMb = $env:QWEN3_GGUF_MIN_FREE_RAM_BEFORE_LAUNCH_MB
if (-not $minFreeRamBeforeLaunchMb) { $minFreeRamBeforeLaunchMb = "4096" }
$minFreeRamBeforeLaunchMb = [int]$minFreeRamBeforeLaunchMb

$minFreeRamMb = $env:QWEN3_GGUF_MIN_FREE_RAM_MB
if (-not $minFreeRamMb) { $minFreeRamMb = "1024" }
$minFreeRamMb = [int]$minFreeRamMb

$nglSetting = $env:QWEN3_GGUF_GPU_LAYERS
if (-not $nglSetting) { $nglSetting = "auto" }

$port = $env:QWEN3_GGUF_PORT
if (-not $port) { $port = "8080" }

$modelDir = $env:QWEN3_GGUF_MODEL_DIR
if (-not $modelDir) {
    $modelDir = "D:\QwenModels\Qwen3-Omni-30B-A3B-Instruct-GGUF"
}

$localModel = Join-Path $modelDir "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
$localMmproj = Join-Path $modelDir "mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf"
$localModelAria2 = "$localModel.aria2"
$minimumModelBytes = 15GB
$useLocalModel = $false

if (Test-Path -LiteralPath $localModel) {
    if (Test-Path -LiteralPath $localModelAria2) {
        throw "Local Qwen3 GGUF is still downloading: $localModelAria2. Wait for the download to finish, then run this script again."
    }

    $modelFile = Get-Item -LiteralPath $localModel
    if ($modelFile.Length -lt $minimumModelBytes) {
        throw "Local Qwen3 GGUF is incomplete: $localModel. Wait for the download to finish, then run this script again."
    }

    if (-not (Test-Path -LiteralPath $localMmproj)) {
        throw "Qwen3 multimodal projector was not found: $localMmproj"
    }

    $useLocalModel = $true
}

$llama = Get-Command llama -ErrorAction SilentlyContinue
$llamaServer = Get-Command llama-server -ErrorAction SilentlyContinue
$wingetLlamaServer = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"

function Get-NvidiaFreeVramMb {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) { return $null }

    $freeText = & $nvidiaSmi.Source --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if (-not $freeText) { return $null }

    $freeText = "$freeText".Trim()
    if (-not $freeText) { return $null }

    return [int]($freeText -split "\s+")[0]
}

function Get-MemoryPressureSummary {
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
    if ($memory -and $null -ne $memory.AvailableMBytes) {
        return [ordered]@{
            available_mb = [int]$memory.AvailableMBytes
            pages_per_sec = [int]$memory.PagesPersec
            page_reads_per_sec = [int]$memory.PageReadsPersec
            page_writes_per_sec = [int]$memory.PageWritesPersec
        }
    }

    $os = Get-CimInstance Win32_OperatingSystem
    return [ordered]@{
        available_mb = [int][math]::Round($os.FreePhysicalMemory / 1024)
        pages_per_sec = $null
        page_reads_per_sec = $null
        page_writes_per_sec = $null
    }
}

function Get-PageFileUsageSummary {
    $pageFiles = @(Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue)
    if ($pageFiles.Count -eq 0) {
        return [ordered]@{
            allocated_mb = 0
            current_mb = 0
            peak_mb = 0
        }
    }

    return [ordered]@{
        allocated_mb = [int](($pageFiles | Measure-Object -Property AllocatedBaseSize -Sum).Sum)
        current_mb = [int](($pageFiles | Measure-Object -Property CurrentUsage -Sum).Sum)
        peak_mb = [int](($pageFiles | Measure-Object -Property PeakUsage -Sum).Sum)
    }
}

function Test-Enabled {
    param([string] $Value)

    $normalized = "$Value".Trim().ToLowerInvariant()
    return @("1", "true", "yes", "on", "enabled").Contains($normalized)
}

function Get-GpuLayerCandidates {
    param([string] $Setting)

    $normalized = "$Setting".Trim().ToLowerInvariant()
    if ($normalized -and $normalized -ne "auto") {
        return @([int]$normalized)
    }

    $candidateText = $env:QWEN3_GGUF_GPU_LAYER_CANDIDATES
    if ($candidateText) {
        $parsed = @(
            $candidateText -split "[,\s]+" |
                Where-Object { $_ } |
                ForEach-Object { [int]$_ }
        )
        if ($parsed.Count -gt 0) { return $parsed }
    }

    return @(999, 96, 80, 64, 48, 40, 36, 32, 30, 28, 26, 24, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
}

function Test-LlamaServerReady {
    param([string] $Port)

    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Stop-ProcessTree {
    param([int] $ProcessId)

    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

$serverExe = $null
$serverPrefixArgs = @()
if ($llama) {
    $serverExe = $llama.Source
    $serverPrefixArgs = @("serve")
} elseif ($llamaServer) {
    $serverExe = $llamaServer.Source
} elseif (Test-Path -LiteralPath $wingetLlamaServer) {
    $serverExe = $wingetLlamaServer
} else {
    throw "llama.cpp was not found. Install it first, for example: winget install llama.cpp"
}

if ($useLocalModel) {
    $baseArgs = @("-m", $localModel, "-mm", $localMmproj, "-a", $apiModel)
} else {
    $baseArgs = @("-hf", $hfRepo, "-a", $apiModel)
}
$baseArgs += @(
    "--host", "127.0.0.1",
    "--port", "$port",
    "-c", "$ctx",
    "-np", "$parallel",
    "-b", "$batchSize",
    "-ub", "$ubatchSize",
    "-ctk", "$cacheTypeK",
    "-ctv", "$cacheTypeV",
    "-fa", "$flashAttn",
    "--cache-ram", "$promptCacheRamMb",
    "--ctx-checkpoints", "$ctxCheckpoints",
    "--jinja"
)

if (Test-Enabled -Value $mlock) {
    $baseArgs += @("--mlock")
}
if (Test-Enabled -Value $promptCache) {
    $baseArgs += @("--cache-prompt")
} else {
    $baseArgs += @("--no-cache-prompt")
}

$startupTimeoutSeconds = $env:QWEN3_GGUF_STARTUP_TIMEOUT_SECONDS
if (-not $startupTimeoutSeconds) { $startupTimeoutSeconds = "180" }
$startupTimeoutSeconds = [int]$startupTimeoutSeconds

$minFreeVramMb = $env:QWEN3_GGUF_MIN_FREE_VRAM_MB
if (-not $minFreeVramMb) { $minFreeVramMb = "768" }
$minFreeVramMb = [int]$minFreeVramMb

$logDir = $env:QWEN3_GGUF_LOG_DIR
if (-not $logDir) { $logDir = "D:\QwenTemp" }
New-Item -ItemType Directory -Force $logDir | Out-Null

$stateFile = $env:QWEN3_LLAMA_STATE_FILE
if (-not $stateFile) { $stateFile = "D:\QwenTemp\qwen3-llamacpp-state.json" }
Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue

$isAutoGpuLayers = "$nglSetting".Trim().ToLowerInvariant() -eq "auto"
$gpuLayerCandidates = Get-GpuLayerCandidates -Setting $nglSetting

$memoryBeforeLaunch = Get-MemoryPressureSummary
$availableRamBeforeLaunchMb = [int]$memoryBeforeLaunch.available_mb
if ($minFreeRamBeforeLaunchMb -gt 0 -and $availableRamBeforeLaunchMb -lt $minFreeRamBeforeLaunchMb) {
    throw "Only ${availableRamBeforeLaunchMb}MB physical RAM is available before launching Qwen3. Close other apps or lower QWEN3_GGUF_MIN_FREE_RAM_BEFORE_LAUNCH_MB to continue. This guard prevents Windows paging."
}

foreach ($candidate in $gpuLayerCandidates) {
    $stdoutLog = Join-Path $logDir "qwen3-llamacpp-ngl-$candidate.out.log"
    $stderrLog = Join-Path $logDir "qwen3-llamacpp-ngl-$candidate.err.log"
    Remove-Item -LiteralPath $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue

    $argsForCandidate = @($serverPrefixArgs + $baseArgs + @("-ngl", "$candidate"))
    Write-Host "Starting Qwen3 llama.cpp with -ngl $candidate, context $ctx..."
    $process = Start-Process -FilePath $serverExe `
        -ArgumentList $argsForCandidate `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    $ready = $false
    $deadline = (Get-Date).AddSeconds($startupTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if ($process.HasExited) { break }
        if (Test-LlamaServerReady -Port $port) {
            $ready = $true
            break
        }
    }

    if ($ready) {
        $freeVramMb = Get-NvidiaFreeVramMb
        if ($null -ne $freeVramMb -and $freeVramMb -lt $minFreeVramMb -and $candidate -gt 0) {
            Write-Warning "Qwen3 loaded with -ngl $candidate, but free VRAM is only ${freeVramMb}MB. Retrying with fewer GPU layers."
            Stop-ProcessTree -ProcessId $process.Id
            Start-Sleep -Seconds 3
            continue
        }

        $memoryAfterLoad = Get-MemoryPressureSummary
        $availableRamMb = [int]$memoryAfterLoad.available_mb
        if ($minFreeRamMb -gt 0 -and $availableRamMb -lt $minFreeRamMb) {
            Stop-ProcessTree -ProcessId $process.Id
            throw "Qwen3 loaded with -ngl $candidate, but available physical RAM is only ${availableRamMb}MB. Stopping to avoid Windows paging. Close other apps, lower context, or lower QWEN3_GGUF_MIN_FREE_RAM_MB if you want to allow this."
        }

        $pageFileUsage = Get-PageFileUsageSummary
        $state = [ordered]@{
            api_model = $apiModel
            process_id = $process.Id
            gpu_layers = $candidate
            context = [int]$ctx
            parallel = [int]$parallel
            batch_size = [int]$batchSize
            ubatch_size = [int]$ubatchSize
            cache_type_k = $cacheTypeK
            cache_type_v = $cacheTypeV
            flash_attn = $flashAttn
            mlock = $mlock
            prompt_cache = $promptCache
            prompt_cache_ram_mb = [int]$promptCacheRamMb
            ctx_checkpoints = [int]$ctxCheckpoints
            min_free_ram_mb = $minFreeRamMb
            available_ram_mb_before_launch = $availableRamBeforeLaunchMb
            available_ram_mb_after_load = $availableRamMb
            memory_before_launch = $memoryBeforeLaunch
            memory_after_load = $memoryAfterLoad
            min_free_vram_mb = $minFreeVramMb
            free_vram_mb_after_load = $freeVramMb
            pagefile = $pageFileUsage
            server = "http://127.0.0.1:$port/v1"
            started_at = (Get-Date).ToString("o")
            stdout_log = $stdoutLog
            stderr_log = $stderrLog
        }
        $state | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8

        Write-Host "Qwen3 llama.cpp is ready with -ngl $candidate. State: $stateFile"
        while (-not $process.HasExited) {
            Start-Sleep -Seconds 5
        }
        exit $process.ExitCode
    }

    Stop-ProcessTree -ProcessId $process.Id
    $tail = Get-Content -LiteralPath $stderrLog -Tail 20 -ErrorAction SilentlyContinue
    Write-Warning "Qwen3 llama.cpp did not become ready with -ngl $candidate."
    if ($tail) { $tail | ForEach-Object { Write-Warning $_ } }

    if (-not $isAutoGpuLayers) {
        throw "Qwen3 llama.cpp failed with fixed -ngl $candidate. Set QWEN3_GGUF_GPU_LAYERS=auto to allow fallback."
    }
}

throw "Qwen3 llama.cpp failed for all GPU layer candidates: $($gpuLayerCandidates -join ', ')"
