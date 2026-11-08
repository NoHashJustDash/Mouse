# SPDX-License-Identifier: GPL-3.0-or-later
# Assemble the pinned ONNX Runtime DirectML SDK that the windows-dml preset builds against,
# from two NuGet packages verified by SHA-256 (observed hashes; nuget.org's own SHA-512
# catalog hashes matched on 2026-10-27). The result has the layout of ORT's release
# archives (include/, lib/, VERSION_NUMBER, LICENSE), so cmake/OnnxRuntime.cmake treats
# it like any other pinned ORT SDK; lib/ also carries the DirectML.dll it needs.
[CmdletBinding()]
param([switch]$Offline)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = Split-Path -Parent $PSScriptRoot
$downloads = Join-Path $root '.toolchain\downloads'
$sdk = Join-Path $root '.toolchain\sdks\onnxruntime\onnxruntime-win-x64-directml-1.22.0'
$packages = @(
    @{ Name = 'microsoft.ml.onnxruntime.directml.1.22.0.nupkg'
       Url = 'https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.22.0/microsoft.ml.onnxruntime.directml.1.22.0.nupkg'
       Sha256 = '29f9872d786236b79aa83f94482f3a17c14297e4833768d6d0ed4883ee732e60' },
    @{ Name = 'microsoft.ai.directml.1.15.4.nupkg'
       Url = 'https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg'
       Sha256 = '4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9' }
)

if (Test-Path -LiteralPath (Join-Path $sdk 'VERSION_NUMBER')) {
    Write-Output "DirectML ORT SDK already staged: $sdk"
    return
}
[IO.Directory]::CreateDirectory($downloads) | Out-Null
foreach ($package in $packages) {
    $path = Join-Path $downloads $package.Name
    if (-not (Test-Path -LiteralPath $path)) {
        if ($Offline) { throw ('Missing offline package: ' + $path) }
        $partial = $path + '.' + [guid]::NewGuid().ToString('N') + '.partial'
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $package.Url -OutFile $partial
            Move-Item -LiteralPath $partial -Destination $path
        } finally {
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
        }
    }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $package.Sha256) {
        throw ('Package SHA-256 mismatch; inspect and remove it: ' + $path)
    }
}

$work = Join-Path $root ('.toolchain\sdks\onnxruntime\directml-staging-' + [guid]::NewGuid().ToString('N'))
try {
    $ort = Join-Path $work 'ort'
    $dml = Join-Path $work 'directml'
    $staged = Join-Path $work 'sdk'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $downloads $packages[0].Name), $ort)
    [IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $downloads $packages[1].Name), $dml)
    foreach ($directory in 'include', 'lib') {
        [IO.Directory]::CreateDirectory((Join-Path $staged $directory)) | Out-Null
    }
    Copy-Item -Path (Join-Path $ort 'build\native\include\*') -Destination (Join-Path $staged 'include')
    Copy-Item -LiteralPath (Join-Path $dml 'include\DirectML.h'), (Join-Path $dml 'include\DirectMLConfig.h') `
        -Destination (Join-Path $staged 'include')
    foreach ($file in 'onnxruntime.dll', 'onnxruntime.lib', 'onnxruntime_providers_shared.dll') {
        Copy-Item -LiteralPath (Join-Path $ort ('runtimes\win-x64\native\' + $file)) -Destination (Join-Path $staged 'lib')
    }
    Copy-Item -LiteralPath (Join-Path $dml 'bin\x64-win\DirectML.dll') -Destination (Join-Path $staged 'lib')
    Copy-Item -LiteralPath (Join-Path $ort 'LICENSE') -Destination (Join-Path $staged 'LICENSE')
    Copy-Item -LiteralPath (Join-Path $ort 'ThirdPartyNotices.txt') -Destination (Join-Path $staged 'ThirdPartyNotices.txt')
    Copy-Item -LiteralPath (Join-Path $dml 'LICENSE.txt') -Destination (Join-Path $staged 'DirectML-LICENSE.txt')
    Copy-Item -LiteralPath (Join-Path $dml 'ThirdPartyNotices.txt') -Destination (Join-Path $staged 'DirectML-ThirdPartyNotices.txt')
    [IO.File]::WriteAllText((Join-Path $staged 'VERSION_NUMBER'), "1.22.0`n")
    [IO.Directory]::CreateDirectory((Split-Path -Parent $sdk)) | Out-Null
    Move-Item -LiteralPath $staged -Destination $sdk
    Write-Output "Staged DirectML ORT SDK: $sdk"
} finally {
    if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force }
}
