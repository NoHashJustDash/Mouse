# SPDX-License-Identifier: GPL-3.0-or-later
$taskPreviousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    . (Join-Path $PSScriptRoot 'Enter-DeveloperShell.ps1')
    if ($ErrorActionPreference -ne 'Continue') {
        throw 'Developer shell changed the caller ErrorActionPreference.'
    }
    $ErrorActionPreference = 'Stop'
    $taskVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $taskVsRoot = & $taskVswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $taskExpectedVersion = (Get-Content -LiteralPath (Join-Path $taskVsRoot 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt') -Raw).Trim()
    $taskExpectedCompiler = Join-Path $taskVsRoot ("VC\Tools\MSVC\{0}\bin\Hostx64\x64\cl.exe" -f $taskExpectedVersion)
    $taskActualCompiler = (Get-Command cl.exe -ErrorAction Stop).Source
    if ($taskActualCompiler -ne $taskExpectedCompiler) {
        throw ("Compiler mismatch: expected {0}, selected {1}" -f $taskExpectedCompiler, $taskActualCompiler)
    }
    Write-Output ("Developer shell regression passed: MSVC {0}; caller preference preserved." -f $taskExpectedVersion)
} finally {
    $ErrorActionPreference = $taskPreviousPreference
}
