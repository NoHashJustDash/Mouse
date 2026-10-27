# SPDX-License-Identifier: GPL-3.0-or-later
[CmdletBinding()]
param([string]$ClangFormat = 'clang-format')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    $taskFiles = @(& git ls-files --cached --others --exclude-standard -- 'engine/*.h' 'engine/*.cpp' 'tests/*.h' 'tests/*.cpp')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate source files with Git.' }
    if ($taskFiles.Count -eq 0) { throw 'No C++ sources found.' }
    & $ClangFormat --dry-run --Werror @taskFiles
    if ($LASTEXITCODE -ne 0) { throw 'clang-format found formatting errors.' }
} finally {
    Pop-Location
}
