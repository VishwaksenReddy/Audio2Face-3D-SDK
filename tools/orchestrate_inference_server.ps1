<#
.SYNOPSIS
Builds Audio2Face-3D-SDK inference server and produces a minimal runnable bundle.

.DESCRIPTION
Runs the repo's existing pipeline:
  1) fetch_deps.bat (if needed)
  2) build.bat
  3) download_models.bat
  4) gen_testdata.bat
Then validates the produced binaries and the selected model.json "file closure",
and finally creates a minimal "bundle" folder containing only:
  - audio2face-inference-server.exe (+ manifest)
  - audio2x.dll (+ manifest)
  - model.json + referenced assets
  - optional licenses + startup scripts + manifest

.EXAMPLE
.\orchestrate_inference_server.bat

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools\orchestrate_inference_server.ps1 -Config release

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools\orchestrate_inference_server.ps1 -SkipDownloadModels -SkipGenTestdata
#>

[CmdletBinding()]
param(
    [ValidateSet('release', 'debug')]
    [string]$Config = 'release',

    [string]$BuildTarget = 'audio2face-inference-server',

    [string]$ModelJson = '_data/generated/audio2face-sdk/samples/data/mark/model.json',

    [string]$OutputBundle = '',

    [switch]$SkipDeps,
    [switch]$SkipBuild,
    [switch]$SkipDownloadModels,
    [switch]$SkipGenTestdata,
    [switch]$SkipValidation,
    [switch]$SkipBundle,

    [switch]$SmokeTest,
    [int]$SmokeTestTimeoutSec = 20,

    [string]$BindHost = '127.0.0.1',
    [int]$Port = 0,

    [bool]$IncludeLicenses = $true,
    [switch]$IncludeHashes,

    [switch]$InPlacePrune,
    [switch]$DryRun,
    [switch]$Force,
    [switch]$DeleteInsteadOfMove
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Section([string]$title) {
    Write-Host ""
    Write-Host "==== $title ===="
}

function Fail([string]$message) {
    throw $message
}

function Resolve-FullPath([string]$path, [string]$baseDir) {
    if ([string]::IsNullOrWhiteSpace($path)) {
        Fail "Path is empty."
    }
    if ([System.IO.Path]::IsPathRooted($path)) {
        return [System.IO.Path]::GetFullPath($path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $baseDir $path))
}

function Assert-CommandExists([string]$commandName, [string]$help) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        Fail "Required command '$commandName' not found in PATH. $help"
    }
}

function Assert-EnvVar([string]$name, [string]$help) {
    $value = (Get-Item -Path "Env:$name" -ErrorAction SilentlyContinue).Value
    if ([string]::IsNullOrWhiteSpace($value)) {
        Fail "$name is not defined. $help"
    }
}

function Invoke-CheckedBat([string]$batPath, [string[]]$args) {
    $displayArgs = ($args | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    Write-Host ">> $batPath $displayArgs"
    & $batPath @args
    if ($LASTEXITCODE -ne 0) {
        Fail "Command failed (exit code $LASTEXITCODE): $batPath"
    }
}

function Ensure-Directory([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function New-CleanDirectory([string]$path) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $path | Out-Null
}

function Test-IsSubPath([string]$parentDir, [string]$childPath) {
    $parentFull = [System.IO.Path]::GetFullPath($parentDir).TrimEnd('\', '/')
    $childFull = [System.IO.Path]::GetFullPath($childPath).TrimEnd('\', '/')
    if ($childFull.Length -lt $parentFull.Length) { return $false }
    if ($childFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    return $childFull.StartsWith($parentFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-SafeToOverwriteBundleDir([string]$bundleDir, [string]$repoRoot) {
    if (-not (Test-Path -LiteralPath $bundleDir)) { return }
    $distRoot = Join-Path $repoRoot 'dist'
    if (Test-IsSubPath $distRoot $bundleDir) { return }
    if (-not $Force) {
        Fail "OutputBundle '$bundleDir' exists and is outside '$distRoot'. Re-run with -Force or choose an output under dist/."
    }
}

function Test-HasProperty($obj, [string]$name) {
    if ($null -eq $obj) { return $false }
    return $null -ne $obj.PSObject.Properties[$name]
}

function Get-PropertyValue($obj, [string]$name) {
    return $obj.PSObject.Properties[$name].Value
}

function Add-StringPath([System.Collections.Generic.List[string]]$out, $value, [string]$fieldName) {
    if ($null -eq $value) {
        Fail "Missing required field '$fieldName' in model.json."
    }
    if ($value -isnot [string] -or [string]::IsNullOrWhiteSpace($value)) {
        Fail "Field '$fieldName' in model.json must be a non-empty string."
    }
    $out.Add($value)
}

function Add-OptionalStringPath([System.Collections.Generic.List[string]]$out, $value, [string]$fieldName) {
    if ($null -eq $value) { return }
    if ($value -isnot [string] -or [string]::IsNullOrWhiteSpace($value)) {
        Fail "Field '$fieldName' in model.json must be a string when present."
    }
    $out.Add($value)
}

function Add-StringPathArray([System.Collections.Generic.List[string]]$out, $value, [string]$fieldName) {
    if ($null -eq $value) {
        Fail "Missing required field '$fieldName' in model.json."
    }
    if ($value -isnot [System.Array]) {
        Fail "Field '$fieldName' in model.json must be an array."
    }
    foreach ($item in $value) {
        Add-StringPath $out $item $fieldName
    }
}

function Add-BlendshapeEntry([System.Collections.Generic.List[string]]$out, $entry, [string]$fieldName) {
    if ($null -eq $entry -or $entry -isnot [psobject]) {
        Fail "Invalid '$fieldName' entry in model.json."
    }
    foreach ($part in @('skin', 'tongue')) {
        if (-not (Test-HasProperty $entry $part)) {
            Fail "Missing '$fieldName.$part' in model.json."
        }
        $partObj = Get-PropertyValue $entry $part
        if ($null -eq $partObj -or $partObj -isnot [psobject]) {
            Fail "Invalid '$fieldName.$part' in model.json."
        }
        if (-not (Test-HasProperty $partObj 'config') -or -not (Test-HasProperty $partObj 'data')) {
            Fail "Missing '$fieldName.$part.config' or '$fieldName.$part.data' in model.json."
        }
        Add-StringPath $out (Get-PropertyValue $partObj 'config') "$fieldName.$part.config"
        Add-StringPath $out (Get-PropertyValue $partObj 'data') "$fieldName.$part.data"
    }
}

function Get-ModelReferencedRelativePaths([string]$modelJsonPath) {
    if (-not (Test-Path -LiteralPath $modelJsonPath)) {
        Fail "Model JSON not found: $modelJsonPath"
    }

    $raw = Get-Content -LiteralPath $modelJsonPath -Raw
    $data = $raw | ConvertFrom-Json

    $paths = New-Object 'System.Collections.Generic.List[string]'

    if (-not (Test-HasProperty $data 'networkInfoPath')) { Fail "Missing required field 'networkInfoPath' in model.json." }
    if (-not (Test-HasProperty $data 'networkPath')) { Fail "Missing required field 'networkPath' in model.json." }
    Add-StringPath $paths (Get-PropertyValue $data 'networkInfoPath') 'networkInfoPath'
    Add-StringPath $paths (Get-PropertyValue $data 'networkPath') 'networkPath'

    if (Test-HasProperty $data 'emotionDatabasePath') {
        Add-OptionalStringPath $paths (Get-PropertyValue $data 'emotionDatabasePath') 'emotionDatabasePath'
    }

    if (Test-HasProperty $data 'modelConfigPath') {
        Add-StringPath $paths (Get-PropertyValue $data 'modelConfigPath') 'modelConfigPath'
    } elseif (Test-HasProperty $data 'modelConfigPaths') {
        Add-StringPathArray $paths (Get-PropertyValue $data 'modelConfigPaths') 'modelConfigPaths'
    }

    if (Test-HasProperty $data 'modelDataPath') {
        Add-StringPath $paths (Get-PropertyValue $data 'modelDataPath') 'modelDataPath'
    } elseif (Test-HasProperty $data 'modelDataPaths') {
        Add-StringPathArray $paths (Get-PropertyValue $data 'modelDataPaths') 'modelDataPaths'
    }

    if (-not (Test-HasProperty $data 'blendshapePaths')) {
        Fail "Missing required field 'blendshapePaths' in model.json."
    }
    $bs = Get-PropertyValue $data 'blendshapePaths'
    if ($bs -is [System.Array]) {
        $idx = 0
        foreach ($entry in $bs) {
            Add-BlendshapeEntry $paths $entry ("blendshapePaths[$idx]")
            $idx++
        }
    } elseif ($bs -is [psobject]) {
        Add-BlendshapeEntry $paths $bs 'blendshapePaths'
    } else {
        Fail "Field 'blendshapePaths' in model.json must be an object or array."
    }

    # De-duplicate while preserving order.
    $seen = @{}
    $out = New-Object 'System.Collections.Generic.List[string]'
    foreach ($p in $paths) {
        if (-not $seen.ContainsKey($p)) {
            $seen[$p] = $true
            $out.Add($p)
        }
    }
    return $out
}

function Validate-ModelClosure([string]$modelJsonPath) {
    Write-Section "Validate model.json closure"

    $modelDir = Split-Path -Parent $modelJsonPath
    $relPaths = Get-ModelReferencedRelativePaths $modelJsonPath

    $missing = @()
    foreach ($rel in $relPaths) {
        $full = Resolve-FullPath $rel $modelDir
        if (-not (Test-Path -LiteralPath $full)) {
            $missing += $rel
        }
    }
    if ($missing.Count -gt 0) {
        $list = ($missing | ForEach-Object { "  - $_" }) -join "`n"
        Fail "Model asset(s) missing under '$modelDir':`n$list"
    }

    Write-Host "OK: model.json references $($relPaths.Count) file(s), all present."
    return $relPaths
}

function Get-FreeTcpPort() {
    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        return ([int]$listener.LocalEndpoint.Port)
    } finally {
        $listener.Stop()
    }
}

function Invoke-ServerSmokeTest(
    [string]$serverExe,
    [string]$audio2xBinDir,
    [string]$modelJsonPath,
    [string]$host,
    [int]$port,
    [int]$timeoutSec
) {
    Write-Section "Smoke test (optional)"

    if (-not (Test-Path -LiteralPath $serverExe)) {
        Fail "Server exe not found: $serverExe"
    }
    if (-not (Test-Path -LiteralPath $modelJsonPath)) {
        Fail "Model JSON not found: $modelJsonPath"
    }

    Assert-EnvVar 'CUDA_PATH' "Set CUDA_PATH (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x)."
    Assert-EnvVar 'TENSORRT_ROOT_DIR' "Set TENSORRT_ROOT_DIR (e.g. C:\TensorRT-10.x.y.z)."

    if ($port -eq 0) {
        $port = Get-FreeTcpPort
    }

    $oldPath = $env:PATH
    $logDir = Join-Path $env:TEMP "a2f-inference-server-smoketest"
    Ensure-Directory $logDir
    $stdout = Join-Path $logDir "stdout.txt"
    $stderr = Join-Path $logDir "stderr.txt"
    if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force }
    if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force }

    try {
        $env:PATH = @(
            $audio2xBinDir
            (Join-Path $env:CUDA_PATH 'bin')
            (Join-Path $env:TENSORRT_ROOT_DIR 'lib')
            (Join-Path $env:TENSORRT_ROOT_DIR 'bin')
            $oldPath
        ) -join ';'

        $args = @('--host', $host, '--port', $port, '--model', $modelJsonPath)
        Write-Host ">> $serverExe $($args -join ' ')"
        $proc = Start-Process -FilePath $serverExe -ArgumentList $args -WorkingDirectory $RepoRoot -PassThru -NoNewWindow `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr

        $deadline = (Get-Date).AddSeconds($timeoutSec)
        $connected = $false
        while ((Get-Date) -lt $deadline) {
            if ($proc.HasExited) {
                break
            }
            try {
                $client = New-Object System.Net.Sockets.TcpClient
                $client.Connect($host, $port)
                $client.Close()
                $connected = $true
                break
            } catch {
                Start-Sleep -Milliseconds 200
            }
        }

        if (-not $connected) {
            $stderrText = ""
            if (Test-Path -LiteralPath $stderr) {
                $stderrText = Get-Content -LiteralPath $stderr -Raw
            }
            $stdoutText = ""
            if (Test-Path -LiteralPath $stdout) {
                $stdoutText = Get-Content -LiteralPath $stdout -Raw
            }
            $extra = ""
            if (-not [string]::IsNullOrWhiteSpace($stderrText)) { $extra += "`n--- stderr ---`n$stderrText" }
            if (-not [string]::IsNullOrWhiteSpace($stdoutText)) { $extra += "`n--- stdout ---`n$stdoutText" }
            Fail "Smoke test failed: server did not accept TCP connections on ${host}:${port} within ${timeoutSec}s.$extra"
        }

        Write-Host "OK: server accepted TCP connections on ws://$host`:$port"
    } finally {
        $env:PATH = $oldPath
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

function Copy-FileToBundle([string]$src, [string]$dst) {
    Ensure-Directory (Split-Path -Parent $dst)
    Copy-Item -LiteralPath $src -Destination $dst -Force
}

function Copy-ModelToBundle(
    [string]$modelJsonPath,
    [string[]]$referencedRelPaths,
    [string]$bundleRoot
) {
    $modelDir = Split-Path -Parent $modelJsonPath
    $identityName = Split-Path -Leaf $modelDir
    $bundleModelDir = Join-Path $bundleRoot (Join-Path 'models' $identityName)
    Ensure-Directory $bundleModelDir

    Copy-FileToBundle $modelJsonPath (Join-Path $bundleModelDir 'model.json')
    foreach ($rel in $referencedRelPaths) {
        $src = Resolve-FullPath $rel $modelDir
        $dst = Join-Path $bundleModelDir $rel
        Copy-FileToBundle $src $dst
    }

    return @{
        IdentityName = $identityName
        BundleModelJson = (Join-Path (Join-Path 'models' $identityName) 'model.json')
    }
}

function Write-StartScripts(
    [string]$bundleRoot,
    [string]$defaultModelRelPath
) {
    $ps1Path = Join-Path $bundleRoot 'start_server.ps1'
    $batPath = Join-Path $bundleRoot 'start_server.bat'

    $ps1Template = @'
[CmdletBinding()]
param(
    [string]\$BindHost = '127.0.0.1',
    [int]\$Port = 8765,
    [string]\$Model = "__DEFAULT_MODEL__",
    [string[]]\$ExtraArgs = @()
)

Set-StrictMode -Version Latest
\$ErrorActionPreference = 'Stop'

function Fail([string]\$message) { throw \$message }

if ([string]::IsNullOrWhiteSpace(\$env:CUDA_PATH)) {
    Fail 'CUDA_PATH is not set (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x).'
}
if ([string]::IsNullOrWhiteSpace(\$env:TENSORRT_ROOT_DIR)) {
    Fail 'TENSORRT_ROOT_DIR is not set (e.g. C:\TensorRT-10.x.y.z).'
}

\$root = Split-Path -Parent \$MyInvocation.MyCommand.Path
\$bin = Join-Path \$root 'bin'
\$modelPath = if ([System.IO.Path]::IsPathRooted(\$Model)) { \$Model } else { Join-Path \$root \$Model }

if (-not (Test-Path -LiteralPath \$modelPath)) {
    Fail \"Model not found: \$modelPath\"
}

\$oldPath = \$env:PATH
try {
    \$env:PATH = @(
        \$bin
        (Join-Path \$env:CUDA_PATH 'bin')
        (Join-Path \$env:TENSORRT_ROOT_DIR 'lib')
        (Join-Path \$env:TENSORRT_ROOT_DIR 'bin')
        \$oldPath
    ) -join ';'

    \$exe = Join-Path \$bin 'audio2face-inference-server.exe'
    & \$exe --host \$BindHost --port \$Port --model \$modelPath @ExtraArgs
} finally {
    \$env:PATH = \$oldPath
}
'@
    $ps1 = $ps1Template.Replace('__DEFAULT_MODEL__', $defaultModelRelPath)

    $bat = @"
@echo off
setlocal

set "ROOT=%~dp0"
set "BIN=%ROOT%bin"

if not "%CUDA_PATH%"=="" goto cuda_ok
echo CUDA_PATH is not set (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x)
exit /b 1
:cuda_ok

if not "%TENSORRT_ROOT_DIR%"=="" goto trt_ok
echo TENSORRT_ROOT_DIR is not set (e.g. C:\TensorRT-10.x.y.z)
exit /b 1
:trt_ok

set "PATH=%BIN%;%CUDA_PATH%\bin;%TENSORRT_ROOT_DIR%\lib;%TENSORRT_ROOT_DIR%\bin;%PATH%"

"%BIN%\audio2face-inference-server.exe" --host 127.0.0.1 --port 8765 --model "%ROOT%$defaultModelRelPath" %*
"@

    Set-Content -LiteralPath $ps1Path -Value $ps1 -Encoding UTF8
    Set-Content -LiteralPath $batPath -Value $bat -Encoding ASCII
}

function Write-BundleManifest(
    [string]$bundleRoot,
    [string]$config,
    [string]$buildTarget,
    [string]$sourceModelJson,
    [string]$bundleModelJsonRel,
    [string[]]$modelRelPaths,
    [bool]$includeHashes
) {
    $manifestPath = Join-Path $bundleRoot 'bundle_manifest.json'

    $gitCommit = $null
    try {
        $gitCommit = (& git rev-parse HEAD 2>$null).Trim()
    } catch {
        $gitCommit = $null
    }

    $files = @()
    $bundleRootFull = [System.IO.Path]::GetFullPath($bundleRoot)
    foreach ($f in Get-ChildItem -LiteralPath $bundleRootFull -Recurse -File -Force) {
        $rel = $f.FullName.Substring($bundleRootFull.Length).TrimStart('\', '/')
        $entry = [ordered]@{
            path = $rel
            size_bytes = [int64]$f.Length
        }
        if ($includeHashes) {
            $entry.sha256 = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        $files += $entry
    }

    $manifest = [ordered]@{
        created_utc = (Get-Date).ToUniversalTime().ToString('o')
        git_commit = $gitCommit
        config = $config
        build_target = $buildTarget
        source_model_json = $sourceModelJson
        bundle = [ordered]@{
            model_json = $bundleModelJsonRel
        }
        model_references = $modelRelPaths
        files = $files
    }

    $json = $manifest | ConvertTo-Json -Depth 20
    Set-Content -LiteralPath $manifestPath -Value $json -Encoding UTF8
}

function Prune-RepoToBundle([string]$repoRoot, [string]$bundleRoot) {
    Write-Section "In-place prune (optional, destructive)"

    if (-not $Force) {
        Fail "-InPlacePrune requires -Force."
    }

    $repoRootFull = [System.IO.Path]::GetFullPath($repoRoot)
    $bundleRootFull = [System.IO.Path]::GetFullPath($bundleRoot)

    if (-not (Test-Path -LiteralPath $bundleRootFull)) {
        Fail "Bundle root does not exist: $bundleRootFull"
    }

    $backupRoot = Join-Path $repoRootFull (Join-Path 'dist' (Join-Path 'prune-backup' (Get-Date -Format 'yyyyMMdd-HHmmss')))
    Ensure-Directory $backupRoot

    Write-Host "Repo root:   $repoRootFull"
    Write-Host "Keeping:     $bundleRootFull"
    Write-Host "Backup dir:  $backupRoot"
    Write-Host "Mode:        $((if ($DeleteInsteadOfMove) { 'delete' } else { 'move to backup' }))"
    Write-Host "Dry run:     $DryRun"

    $items = Get-ChildItem -LiteralPath $repoRootFull -Force
    foreach ($item in $items) {
        $full = $item.FullName
        if ($full -eq $bundleRootFull) { continue }
        if ($full.StartsWith($bundleRootFull + [System.IO.Path]::DirectorySeparatorChar)) { continue }

        if ($full.StartsWith($backupRoot + [System.IO.Path]::DirectorySeparatorChar)) { continue }
        if ($full -eq $backupRoot) { continue }

        # Never prune the running script folder if executed from inside the repo.
        if ($full -eq (Join-Path $repoRootFull 'tools')) { continue }
        if ($full.StartsWith((Join-Path $repoRootFull 'tools') + [System.IO.Path]::DirectorySeparatorChar)) { continue }

        $action = if ($DeleteInsteadOfMove) { 'DELETE' } else { 'MOVE' }
        Write-Host "$action $full"
        if ($DryRun) { continue }

        if ($DeleteInsteadOfMove) {
            Remove-Item -LiteralPath $full -Recurse -Force
        } else {
            $dest = Join-Path $backupRoot $item.Name
            Move-Item -LiteralPath $full -Destination $dest -Force
        }
    }

    Write-Host "Prune complete."
}

try {
    $RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    if ([string]::IsNullOrWhiteSpace($OutputBundle)) {
        $OutputBundle = Join-Path $RepoRoot (Join-Path 'dist' (Join-Path 'inference-server-bundle' $Config))
    }

    $OutputBundleFull = Resolve-FullPath $OutputBundle $RepoRoot
    $ModelJsonFull = Resolve-FullPath $ModelJson $RepoRoot

    Push-Location $RepoRoot
    try {
        Write-Section "Preflight"
        Write-Host "Repo root:     $RepoRoot"
        Write-Host "Config:        $Config"
        Write-Host "Build target:  $BuildTarget"
        Write-Host "Model JSON:    $ModelJsonFull"
        Write-Host "Output bundle: $OutputBundleFull"

        if (-not $SkipDownloadModels) {
            Assert-CommandExists 'hf' "Install the Hugging Face CLI (e.g. 'pip install huggingface_hub') and ensure 'hf' is available."
        }
        if (-not $SkipGenTestdata) {
            Assert-CommandExists 'python' "Install Python 3.x and ensure 'python' is available."
            Assert-EnvVar 'TENSORRT_ROOT_DIR' "Required to generate TRT engines via gen_testdata.bat."
        }
        if ($SmokeTest) {
            Assert-EnvVar 'CUDA_PATH' "Required to run the server with CUDA."
            Assert-EnvVar 'TENSORRT_ROOT_DIR' "Required to run the server with TensorRT."
        }

        if (-not $SkipBuild -and -not $SkipDeps) {
            if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot '_deps'))) {
                Write-Section "Fetch dependencies"
                Invoke-CheckedBat (Join-Path $RepoRoot 'fetch_deps.bat') @($Config)
            }
        }

        if (-not $SkipBuild) {
            Write-Section "Build"
            Invoke-CheckedBat (Join-Path $RepoRoot 'build.bat') @($BuildTarget, $Config)
        }

        if (-not $SkipDownloadModels) {
            Write-Section "Download models"
            Invoke-CheckedBat (Join-Path $RepoRoot 'download_models.bat') @()
        }

        if (-not $SkipGenTestdata) {
            Write-Section "Generate test/sample data (TRT engines)"
            Invoke-CheckedBat (Join-Path $RepoRoot 'gen_testdata.bat') @()
        }

        $buildDir = Join-Path $RepoRoot (Join-Path '_build' $Config)
        $serverExe = Join-Path $buildDir 'audio2face-sdk\bin\audio2face-inference-server.exe'
        $serverManifest = Join-Path $buildDir 'audio2face-sdk\bin\audio2face-inference-server.exe.manifest'
        $audio2xDll = Join-Path $buildDir 'audio2x-sdk\bin\audio2x.dll'
        $audio2xManifest = Join-Path $buildDir 'audio2x-sdk\bin\audio2x.dll.manifest'
        $audio2xBinDir = Split-Path -Parent $audio2xDll

        $modelRelPaths = @()
        if (-not $SkipValidation) {
            Write-Section "Validate outputs"
            if (-not (Test-Path -LiteralPath $serverExe)) { Fail "Missing server exe: $serverExe" }
            if (-not (Test-Path -LiteralPath $audio2xDll)) { Fail "Missing audio2x DLL: $audio2xDll" }
            $modelRelPaths = Validate-ModelClosure $ModelJsonFull
        } else {
            Write-Section "Validate outputs (skipped)"
        }

        if ($SmokeTest) {
            Invoke-ServerSmokeTest -serverExe $serverExe -audio2xBinDir $audio2xBinDir -modelJsonPath $ModelJsonFull -host $BindHost -port $Port -timeoutSec $SmokeTestTimeoutSec
        }

        if (-not $SkipBundle) {
            Write-Section "Create bundle"
            Assert-SafeToOverwriteBundleDir $OutputBundleFull $RepoRoot
            New-CleanDirectory $OutputBundleFull

            $bundleBinDir = Join-Path $OutputBundleFull 'bin'
            Ensure-Directory $bundleBinDir
            Copy-FileToBundle $serverExe (Join-Path $bundleBinDir 'audio2face-inference-server.exe')
            if (Test-Path -LiteralPath $serverManifest) {
                Copy-FileToBundle $serverManifest (Join-Path $bundleBinDir 'audio2face-inference-server.exe.manifest')
            }
            Copy-FileToBundle $audio2xDll (Join-Path $bundleBinDir 'audio2x.dll')
            if (Test-Path -LiteralPath $audio2xManifest) {
                Copy-FileToBundle $audio2xManifest (Join-Path $bundleBinDir 'audio2x.dll.manifest')
            }

            if ($SkipValidation) {
                $modelRelPaths = Get-ModelReferencedRelativePaths $ModelJsonFull
            }
            $modelCopyInfo = Copy-ModelToBundle -modelJsonPath $ModelJsonFull -referencedRelPaths $modelRelPaths -bundleRoot $OutputBundleFull

            if ($IncludeLicenses) {
                $licenseTxt = Join-Path $RepoRoot 'LICENSE.txt'
                if (Test-Path -LiteralPath $licenseTxt) {
                    Copy-FileToBundle $licenseTxt (Join-Path $OutputBundleFull 'LICENSE.txt')
                }
                $licensesDir = Join-Path $RepoRoot 'licenses'
                if (Test-Path -LiteralPath $licensesDir) {
                    Copy-Item -LiteralPath $licensesDir -Destination (Join-Path $OutputBundleFull 'licenses') -Recurse -Force
                }
            }

            Write-StartScripts -bundleRoot $OutputBundleFull -defaultModelRelPath $modelCopyInfo.BundleModelJson

            Write-BundleManifest -bundleRoot $OutputBundleFull -config $Config -buildTarget $BuildTarget `
                -sourceModelJson $ModelJsonFull -bundleModelJsonRel $modelCopyInfo.BundleModelJson `
                -modelRelPaths $modelRelPaths -includeHashes ([bool]$IncludeHashes)

            Write-Host "Bundle created: $OutputBundleFull"
        } else {
            Write-Section "Create bundle (skipped)"
        }

        if ($InPlacePrune) {
            Prune-RepoToBundle -repoRoot $RepoRoot -bundleRoot $OutputBundleFull
        }
    } finally {
        Pop-Location
    }
} catch {
    Write-Host ""
    Write-Error $_
    exit 1
}
