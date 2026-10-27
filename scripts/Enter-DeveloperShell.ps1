# SPDX-License-Identifier: GPL-3.0-or-later
# Dot-source to set process-local VS 2022, CMake and Ninja paths.
$taskPreviousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Stop'
    $taskInstaller = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer'
    $taskVswhere = Join-Path $taskInstaller 'vswhere.exe'
    if (-not (Test-Path -LiteralPath $taskVswhere -PathType Leaf)) {
        throw 'Install Visual Studio 2022 Build Tools with Desktop development with C++ first.'
    }
    $taskVsRoot = & $taskVswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $taskVsRoot) {
        throw 'Visual Studio 2022 C++ tools were not found. Install the C++ workload in Visual Studio Installer.'
    }
    # VsDevCmd itself invokes vswhere by name in a clean Windows shell.
    $env:PATH = $taskInstaller + ';' + $env:PATH
    # CUDA's supported side-by-side toolset can change v143's default. The engine
    # explicitly uses VS 2022's current default toolset, independent of nvcc.
    $taskToolsetFile = Join-Path $taskVsRoot 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt'
    $taskToolsetVersion = (Get-Content -LiteralPath $taskToolsetFile -Raw).Trim()
    if ($taskToolsetVersion -notmatch '^14\.[0-9]+\.[0-9]+$') {
        throw ('Invalid VS default toolset version: ' + $taskToolsetVersion)
    }
    $taskVsCommand = 'call "' + (Join-Path $taskVsRoot 'Common7\Tools\VsDevCmd.bat') + '" -no_logo -arch=x64 -host_arch=x64 -vcvars_ver=' + $taskToolsetVersion + ' && set'
    $taskEnvironment = & $env:ComSpec /d /c $taskVsCommand
    if ($LASTEXITCODE -ne 0) {
        throw 'VS 2022 x64 developer environment initialization failed.'
    }
    foreach ($taskLine in $taskEnvironment) {
        if ($taskLine -match '^([A-Za-z_][A-Za-z0-9_()]*)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
    if (-not $env:VCToolsVersion -or $env:VCToolsVersion.TrimEnd('\') -ne $taskToolsetVersion) {
        throw ('VS selected a different MSVC toolset; expected ' + $taskToolsetVersion + ', got ' + $env:VCToolsVersion)
    }
    $taskCmakeDirectory = Join-Path $taskVsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
    $taskNinjaDirectory = Join-Path $taskVsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
    foreach ($taskDirectory in @($taskCmakeDirectory, $taskNinjaDirectory)) {
        if (-not (Test-Path -LiteralPath $taskDirectory -PathType Container)) {
            throw ('Required VS CMake component missing: ' + $taskDirectory)
        }
    }
    $env:PATH = $taskCmakeDirectory + ';' + $taskNinjaDirectory + ';' + $env:PATH
    $env:VCPKG_VISUAL_STUDIO_PATH = $taskVsRoot
} finally {
    $ErrorActionPreference = $taskPreviousPreference
}
