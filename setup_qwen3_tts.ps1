[CmdletBinding()]
param(
    [string]$ModelDir = $env:QWEN3_TTS_MODEL_DIR,
    [switch]$VerifyOnly,
    [switch]$Test,
    [string]$TestText = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ModelDir) {
    $ModelDir = Join-Path $PSScriptRoot "QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF"
}
$ModelDir = [IO.Path]::GetFullPath($ModelDir)
$voiceDir = Join-Path $PSScriptRoot "QwenModels\Qwen3-TTS-12Hz-1.7B-Base-GGUF\voices"
$speakerFile = Join-Path $voiceDir "Chelsie.wav"
$repo = "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
$revision = "ca27d74bc954b73dadab5b71ca265d87fc861a7c"
$files = @(
    @{ Name = "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"; Directory = $ModelDir; Url = "https://huggingface.co/$repo/resolve/$revision/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"; Size = 1035965280L; Hash = "8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129" },
    @{ Name = "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"; Directory = $ModelDir; Url = "https://huggingface.co/$repo/resolve/$revision/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"; Size = 446422912L; Hash = "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2" },
    @{ Name = "Chelsie.wav"; Directory = $voiceDir; Url = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20251126/jnleoh/Chelsie_ZH.wav"; Size = 347564L; Hash = "e2461d0e0fc2bf1e083ea7af40c3b6849e5912b2744162cab6e151c366097b53" }
)

function Find-LlamaTts {
    if ($env:QWEN3_TTS_LLAMA_TTS) {
        if (-not (Test-Path -LiteralPath $env:QWEN3_TTS_LLAMA_TTS -PathType Leaf)) {
            throw "QWEN3_TTS_LLAMA_TTS does not point to an existing executable."
        }
        return $env:QWEN3_TTS_LLAMA_TTS
    }
    $command = Get-Command llama-tts.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $wingetExe = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-tts.exe"
    if (Test-Path -LiteralPath $wingetExe -PathType Leaf) { return $wingetExe }
    throw "llama-tts.exe was not found. Install a llama.cpp build with Qwen3-TTS support, or set QWEN3_TTS_LLAMA_TTS to its full path."
}

function Test-ModelFile([string]$Path, [hashtable]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne $Expected.Size) { return $false }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Expected.Hash
}

$llamaTts = Find-LlamaTts
$helpText = (& $llamaTts --help 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $helpText -notmatch "--tts-lang" -or $helpText -notmatch "--tts-speaker-file") {
    throw "This llama-tts build does not expose the required Qwen3-TTS options."
}
Write-Host "llama-tts: $llamaTts"
Write-Host "Model directory: $ModelDir"
Write-Host "Fixed Chelsie reference: $speakerFile"
Write-Host "Official Hugging Face revision: $revision"

$missing = @()
foreach ($file in $files) {
    $target = Join-Path $file.Directory $file.Name
    if (Test-ModelFile $target $file) {
        Write-Host "Verified: $($file.Name)"
    } elseif (Test-Path -LiteralPath $target) {
        throw "Existing TTS asset failed size/SHA256 verification: $target. Move that file aside before rerunning setup."
    } else {
        $missing += $file
    }
}
if ($VerifyOnly -and $missing.Count -gt 0) {
    throw "Required TTS model files or the fixed Chelsie reference are missing. Run setup_qwen3_tts.ps1 without -VerifyOnly to download them."
}

if ($missing.Count -gt 0) {
    foreach ($group in ($missing | Group-Object -Property Directory)) {
        New-Item -ItemType Directory -Force -Path $group.Name | Out-Null
        $drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($group.Name))
        $needed = ($group.Group | Measure-Object -Property Size -Sum).Sum + 1GB
        if ($drive.AvailableFreeSpace -lt $needed) {
            throw "Not enough disk space at $($group.Name). Need at least $([Math]::Ceiling($needed / 1GB)) GiB free for this download."
        }
    }
    $aria = Get-Command aria2c.exe -ErrorAction SilentlyContinue
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $aria -and -not $curl) { throw "Neither aria2c.exe nor curl.exe was found. Windows curl.exe is sufficient." }

    # Native downloaders do not inherit the Windows system proxy automatically.
    foreach ($file in $missing) {
        $target = Join-Path $file.Directory $file.Name
        $partial = "$target.partial"
        $url = $file.Url
        $downloadUri = [Uri]$url
        $proxy = $null
        if (-not [Net.WebRequest]::DefaultWebProxy.IsBypassed($downloadUri)) {
            $proxy = [Net.WebRequest]::DefaultWebProxy.GetProxy($downloadUri).AbsoluteUri
        }
        Write-Host "Downloading $($file.Name) ($([Math]::Round($file.Size / 1MB, 2)) MiB)..."
        if ($aria) {
            $downloadArgs = @("--continue=true", "--max-connection-per-server=4", "--split=4", "--min-split-size=8M", "--max-tries=5", "--retry-wait=5", "--connect-timeout=30", "--console-log-level=warn", "--summary-interval=30", "--show-console-readout=false", "--download-result=hide", "--dir=$($file.Directory)", "--out=$($file.Name).partial")
            if ($proxy) { $downloadArgs += "--all-proxy=$proxy" }
            $downloadArgs += $url
            & $aria.Source @downloadArgs
        } else {
            $downloadArgs = @("--fail", "--location", "--continue-at", "-", "--retry", "4", "--connect-timeout", "30", "--output", $partial)
            if ($proxy) { $downloadArgs += @("--proxy", $proxy) }
            $downloadArgs += $url
            & $curl.Source @downloadArgs
        }
        if ($LASTEXITCODE -ne 0) { throw "Download failed for $($file.Name). Rerun setup to resume the partial download." }
        if (-not (Test-ModelFile $partial $file)) { throw "Downloaded file failed size/SHA256 verification: $partial" }
        Move-Item -LiteralPath $partial -Destination $target
        Write-Host "Verified: $($file.Name)"
    }
}

Write-Host "TTS model files and the fixed Chelsie reference are ready. App startup does not run this download script."
if ($Test) {
    $tempDir = Join-Path $PSScriptRoot "QwenTemp"
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    if (-not $TestText) { $TestText = '"\u4f60\u597d\uff0c\u8fd9\u662f\u672c\u5730\u8bed\u97f3\u6d4b\u8bd5\u3002"' | ConvertFrom-Json }
    $promptPath = Join-Path $tempDir "qwen3-tts-setup-test.txt"
    $outputPath = Join-Path $tempDir ("qwen3-tts-setup-test-{0}.wav" -f [Guid]::NewGuid().ToString("N"))
    [IO.File]::WriteAllText($promptPath, $TestText, (New-Object Text.UTF8Encoding $false))
    $testArgs = @("-m", (Join-Path $ModelDir $files[0].Name), "-mm", (Join-Path $ModelDir $files[1].Name), "-f", $promptPath, "--tts-lang", "zh", "--tts-speaker-file", $speakerFile, "-o", $outputPath, "--offline", "-ngl", "0", "--no-mmproj-offload", "--no-op-offload", "--no-kv-offload", "-c", "512", "-b", "256", "-ub", "64", "-t", "4", "-n", "120", "--seed", "42")
    $timer = [Diagnostics.Stopwatch]::StartNew()
    & $llamaTts @testArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $outputPath -PathType Leaf)) { throw "TTS synthesis test failed." }
    $timer.Stop()
    $wavBytes = [IO.File]::ReadAllBytes($outputPath)
    if ($wavBytes.Length -lt 44 -or [Text.Encoding]::ASCII.GetString($wavBytes, 0, 4) -ne "RIFF" -or [Text.Encoding]::ASCII.GetString($wavBytes, 8, 4) -ne "WAVE") {
        throw "TTS synthesis did not produce a valid WAV container."
    }
    Write-Host "Test WAV: $outputPath ($([Math]::Round($timer.Elapsed.TotalSeconds, 1)) seconds). Audio was not played."
}
