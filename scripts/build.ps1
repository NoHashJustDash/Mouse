# SPDX-License-Identifier: GPL-3.0-or-later
[CmdletBinding()]
param(
    [ValidateSet('windows-debug', 'windows-release')]
    [string]$Preset = 'windows-release',
    [switch]$SkipPython,
    [switch]$GpuPython,
    [switch]$Offline
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskBaseline = 'a5f736224e85b32e25a8ffc52562333180396642'
$taskOrtHash = '5b5241716b2628c1ab5e79ee620be767531021149ee68f30fc46c16263fb94dd'

function Invoke-MouseCommand {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code {0}: {1} {2}" -f $LASTEXITCODE, $File, ($Arguments -join ' '))
    }
}

function Get-VerifiedDownload {
    param([string]$Url, [string]$Path, [string]$Sha256)
    if (Test-Path -LiteralPath $Path) {
        if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {
            throw ('Cached download has the wrong SHA-256; inspect and remove it: ' + $Path)
        }
        return
    }
    $taskPartial = $Path + '.' + [guid]::NewGuid().ToString('N') + '.partial'
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $taskPartial
        if ((Get-FileHash -LiteralPath $taskPartial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {
            throw ('Downloaded archive SHA-256 mismatch: ' + $Url)
        }
        Move-Item -LiteralPath $taskPartial -Destination $Path
    } finally {
        if (Test-Path -LiteralPath $taskPartial) {
            Remove-Item -LiteralPath $taskPartial
        }
    }
}

Push-Location $taskRoot
try {
    $taskGit = $null
    $taskGitCandidates = @(
        (Join-Path $env:ProgramFiles 'Git\cmd\git.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd\git.exe')
    )
    $taskGitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($taskGitCommand -and $taskGitCommand.Source -notmatch 'codex-runtimes') {
        $taskGitCandidates += $taskGitCommand.Source
    }
    foreach ($taskCandidate in $taskGitCandidates) {
        if (Test-Path -LiteralPath $taskCandidate -PathType Leaf) {
            $taskGit = $taskCandidate
            break
        }
    }
    if (-not $taskGit) {
        throw 'Install system Git for Windows from git-scm.com, then rerun this command.'
    }
    $env:PATH = (Split-Path -Parent $taskGit) + ';' + $env:PATH
    . (Join-Path $PSScriptRoot 'Enter-DeveloperShell.ps1')
    $env:VCPKG_DISABLE_METRICS = '1'
    $env:VCPKG_ROOT = Join-Path $taskRoot '.toolchain\vcpkg'
    [IO.Directory]::CreateDirectory((Join-Path $taskRoot '.toolchain\downloads')) | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $env:VCPKG_ROOT '.git'))) {
        Invoke-MouseCommand $taskGit @('clone', '--filter=blob:none', '--no-checkout', 'https://github.com/microsoft/vcpkg.git', $env:VCPKG_ROOT)
        Invoke-MouseCommand $taskGit @('-C', $env:VCPKG_ROOT, 'fetch', '--depth=1', 'origin', $taskBaseline)
        Invoke-MouseCommand $taskGit @('-C', $env:VCPKG_ROOT, 'checkout', '--detach', $taskBaseline)
    }
    $taskActualBaseline = & $taskGit -C $env:VCPKG_ROOT rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $taskActualBaseline -ne $taskBaseline) {
        throw ('vcpkg checkout must be at pinned baseline ' + $taskBaseline)
    }
    if (-not (Test-Path -LiteralPath (Join-Path $env:VCPKG_ROOT 'vcpkg.exe'))) {
        Invoke-MouseCommand (Join-Path $env:VCPKG_ROOT 'bootstrap-vcpkg.bat') @('-disableMetrics')
    }

    if ($env:MOUSE_ONNXRUNTIME_ROOT -and -not (Test-Path -LiteralPath (Join-Path $env:MOUSE_ONNXRUNTIME_ROOT 'VERSION_NUMBER'))) {
        throw ('Configured MOUSE_ONNXRUNTIME_ROOT is not an SDK directory: ' + $env:MOUSE_ONNXRUNTIME_ROOT)
    }
    if (-not $env:MOUSE_ONNXRUNTIME_ROOT) {
        $env:MOUSE_ONNXRUNTIME_ROOT = Join-Path $taskRoot '.toolchain\sdks\onnxruntime\onnxruntime-win-x64-gpu-1.22.0'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $env:MOUSE_ONNXRUNTIME_ROOT 'VERSION_NUMBER'))) {
        $taskArchive = Join-Path $taskRoot '.toolchain\downloads\onnxruntime-win-x64-gpu-1.22.0.zip'
        # This is an observed hash of the official asset; the release published no digest.
        Get-VerifiedDownload 'https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-gpu-1.22.0.zip' $taskArchive $taskOrtHash
        $taskSdkParent = Split-Path -Parent $env:MOUSE_ONNXRUNTIME_ROOT
        [IO.Directory]::CreateDirectory($taskSdkParent) | Out-Null
        Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskSdkParent
    }

    Invoke-MouseCommand 'cmake' @('--preset', $Preset)
    Invoke-MouseCommand 'cmake' @('--build', '--preset', $Preset)
    Invoke-MouseCommand 'ctest' @('--preset', $Preset)
    if (-not $SkipPython) {
        $taskPython = $null
        foreach ($taskCandidate in @(
            (Join-Path $env:ProgramFiles 'Python312\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
        )) {
            if (Test-Path -LiteralPath $taskCandidate -PathType Leaf) {
                $taskVersion = & $taskCandidate -I -c 'import sys; print(sys.version_info[:2] == (3, 12))'
                if ($LASTEXITCODE -eq 0 -and $taskVersion -eq 'True') {
                    $taskPython = $taskCandidate
                    break
                }
            }
        }
        $taskPythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if (-not $taskPython -and $taskPythonCommand -and $taskPythonCommand.Source -notmatch 'codex-runtimes|WindowsApps') {
            $taskPythonCandidate = & $taskPythonCommand.Source -I -c 'import sys; print(sys.executable if sys.version_info[:2] == (3, 12) else "")'
            if ($LASTEXITCODE -eq 0 -and $taskPythonCandidate) {
                $taskPython = $taskPythonCandidate
            }
        }
        if (-not $taskPython) {
            $taskLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
            if ($taskLauncher) {
                $taskPython = & $taskLauncher.Source -3.12 -I -c 'import sys; print(sys.executable)'
                if ($LASTEXITCODE -ne 0) {
                    $taskPython = $null
                }
            }
        }
        if (-not $taskPython -or $taskPython -match 'codex-runtimes') {
            throw 'Install Python 3.12 from python.org with the launcher, then rerun. No private agent interpreter is accepted.'
        }
        if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
            Invoke-MouseCommand $taskPython @('-I', '-m', 'venv', '.venv')
        }
        $taskVenvPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
        $taskExpectedPrefix = & $taskPython -I -c 'import sys; print(sys.base_prefix)'
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the selected Python interpreter.' }
        $taskVenvPrefix = & $taskVenvPython -I -c 'import sys; print(sys.base_prefix if sys.version_info[:2] == (3, 12) else "")'
        if ($LASTEXITCODE -ne 0 -or $taskVenvPrefix -ne $taskExpectedPrefix) {
            throw 'Existing .venv uses a different interpreter. Remove/recreate it with the selected official Python 3.12 before rerunning.'
        }
        $taskLock = if ($GpuPython) { 'tools/requirements-converter-windows-gpu.lock' } else { 'tools/requirements-converter-windows-cpu.lock' }
        $taskPipArguments = @('-I', '-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes', '-r', $taskLock)
        $taskWheelCache = Join-Path $taskRoot '.toolchain\downloads\python-wheels'
        if (Test-Path -LiteralPath $taskWheelCache -PathType Container) {
            $taskPipArguments += @('--find-links', $taskWheelCache)
        }
        if ($Offline) {
            $taskPipArguments += '--no-index'
        }
        Invoke-MouseCommand $taskVenvPython $taskPipArguments
        Invoke-MouseCommand $taskVenvPython @('-I', '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', './tools')
        Invoke-MouseCommand $taskVenvPython @('-I', '-m', 'pip', 'check')
        Invoke-MouseCommand $taskVenvPython @('-I', '-m', 'pytest', 'tools')
        Invoke-MouseCommand $taskVenvPython @('-I', '-m', 'ruff', 'check', 'tools')
        Invoke-MouseCommand $taskVenvPython @('-I', '-m', 'mypy', '--strict', '--config-file', 'tools/pyproject.toml', 'tools')
    }
} finally {
    Pop-Location
}
