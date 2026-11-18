# SPDX-License-Identifier: GPL-3.0-or-later
<#
.SYNOPSIS
Downloads the NVIDIA libraries Mouse needs into a "gpu" folder beside mouse.exe.

.DESCRIPTION
Mouse's TensorRT build needs NVIDIA's CUDA, cuDNN and TensorRT libraries. This script
downloads the exact versions Mouse is built for from NVIDIA's own servers, checks them
against the SHA-256 hashes in gpu-packages.json, and puts the libraries in a "gpu"
folder. Mouse uses that folder by itself when GpuDllDirectories is empty.

Nothing is installed system-wide, no NVIDIA account is needed, and deleting the "gpu"
folder undoes everything. The libraries are NVIDIA's and come under NVIDIA's licences,
which you must accept with -AcceptNvidiaLicenses.

.PARAMETER Destination
Folder to fill. Default: "gpu" beside this script.

.PARAMETER ArchiveDirectory
Folder for the downloaded archives. An archive already there is used if its size and
hash are right, so a download can be resumed or done on another machine. Archives in a
folder you name are kept; the default folder is deleted when the setup succeeds.

.PARAMETER AcceptNvidiaLicenses
Confirms that you have read and accept the NVIDIA licences the script lists.

.PARAMETER SkipGpuCheck
Do not check that this computer has an NVIDIA GPU (for preparing a folder for another one).
#>
[CmdletBinding()]
param(
    [string]$Destination,
    [string]$ArchiveDirectory,
    [switch]$AcceptNvidiaLicenses,
    [switch]$SkipGpuCheck
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProgressPreference = 'SilentlyContinue'

function Get-Sha256 {
    param([string]$Path)
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-OptionalProperty {
    param($Object, [string]$Name)
    if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
    return $null
}

# Whether every file of a package is in the destination, with the right hash where one is pinned.
function Test-PackageInstalled {
    param($Package, [string]$Folder)
    foreach ($file in $Package.files) {
        $target = Join-Path $Folder (Split-Path -Leaf $file.entry)
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { return $false }
        $expected = Get-OptionalProperty $file 'sha256'
        if ($expected -and (Get-Sha256 $target) -ne $expected) { return $false }
    }
    return $true
}

function Test-Archive {
    param($Package, [string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne [long]$Package.size) { return $false }
    $expected = Get-OptionalProperty $Package 'sha256'
    if ($expected -and (Get-Sha256 $Path) -ne $expected) {
        throw ("{0}: the archive's SHA-256 does not match gpu-packages.json. Delete it and run the setup again: {1}" -f $Package.name, $Path)
    }
    return $true
}

function Get-Archive {
    param($Package, [string]$Path)
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (-not (Test-Path -LiteralPath $curl -PathType Leaf)) {
        throw 'curl.exe was not found in System32 (it ships with Windows 10 version 1803 and later). Download the archives listed in gpu-packages.json yourself and pass their folder as -ArchiveDirectory.'
    }
    $partial = $Path + '.part'
    Write-Host ("Downloading {0} ({1:N0} MB)" -f $Package.name, ($Package.size / 1MB))
    # -C - resumes a partial download; --fail turns an HTTP error into a failure.
    # curl runs as its own process: called with "&", Windows PowerShell 5.1 turns its
    # progress output into errors whenever this script's output is redirected, and the
    # setup would stop at the first download.
    $arguments = '--fail --location --retry 3 --continue-at - --user-agent mouse-setup-gpu --output "{0}" "{1}"' -f $partial, $Package.url
    $process = Start-Process -FilePath $curl -ArgumentList $arguments -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw ("{0}: download failed (curl exit code {1}). Run the setup again to resume. URL: {2}" -f $Package.name, $process.ExitCode, $Package.url)
    }
    Move-Item -LiteralPath $partial -Destination $Path -Force
    if (-not (Test-Archive $Package $Path)) {
        throw ("{0}: the downloaded archive has the wrong size ({1} bytes, expected {2}). Delete it and run the setup again: {3}" -f $Package.name, (Get-Item -LiteralPath $Path).Length, $Package.size, $Path)
    }
}

function Install-Package {
    param($Package, [string]$Archive, [string]$Folder)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        foreach ($file in $Package.files) {
            $entry = $zip.GetEntry($file.entry)
            if ($null -eq $entry) {
                throw ("{0}: the archive has no file {1}" -f $Package.name, $file.entry)
            }
            $name = Split-Path -Leaf $file.entry
            $target = Join-Path $Folder $name
            $staged = $target + '.new'
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $staged, $true)
            $expected = Get-OptionalProperty $file 'sha256'
            if ($expected -and (Get-Sha256 $staged) -ne $expected) {
                Remove-Item -LiteralPath $staged -Force
                throw ("{0}: {1} does not match its SHA-256 in gpu-packages.json, so it was not installed" -f $Package.name, $name)
            }
            Move-Item -LiteralPath $staged -Destination $target -Force
        }
    } finally {
        $zip.Dispose()
    }
}

$manifestPath = Join-Path $PSScriptRoot 'gpu-packages.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw ('gpu-packages.json must be beside this script: ' + $manifestPath)
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.schema -ne 1) { throw 'gpu-packages.json has an unknown schema version' }

if (-not $Destination) { $Destination = Join-Path $PSScriptRoot 'gpu' }
$Destination = [System.IO.Path]::GetFullPath($Destination)
$keepArchives = [bool]$ArchiveDirectory
if (-not $ArchiveDirectory) { $ArchiveDirectory = Join-Path $PSScriptRoot 'gpu-downloads' }
$ArchiveDirectory = [System.IO.Path]::GetFullPath($ArchiveDirectory)

if (-not $SkipGpuCheck) {
    $adapters = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
    if (-not ($adapters | Where-Object { $_ -match 'NVIDIA' })) {
        throw ("No NVIDIA GPU was found (found: {0}). These libraries only work on NVIDIA GPUs. On an AMD or Intel GPU use Mouse's DirectML download, which needs no setup." -f ($adapters -join ', '))
    }
}

$missing = @($manifest.packages | Where-Object { -not (Test-PackageInstalled $_ $Destination) })
if ($missing.Count -eq 0) {
    Write-Host ('Every NVIDIA library is already installed in ' + $Destination)
    exit 0
}

Write-Host 'This setup downloads NVIDIA software under these licences:'
foreach ($license in $manifest.licenses) { Write-Host ('  {0}: {1}' -f $license.name, $license.url) }
if (-not $AcceptNvidiaLicenses) {
    throw 'Read the licences above, then run the setup again with -AcceptNvidiaLicenses to accept them. Nothing was downloaded or installed.'
}

$toDownload = @($missing | Where-Object { -not (Test-Archive $_ (Join-Path $ArchiveDirectory $_.archive)) })
$downloadBytes = [long]0
foreach ($package in $toDownload) { $downloadBytes += [long]$package.size }
Write-Host ('To download: {0:N1} GB in {1} archive(s).' -f ($downloadBytes / 1GB), $toDownload.Count)

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
New-Item -ItemType Directory -Force -Path $ArchiveDirectory | Out-Null
# The archives and the extracted libraries are on disk together until the setup ends.
$archiveBytes = [long]0
foreach ($package in $missing) { $archiveBytes += [long]$package.size }
$needed = $downloadBytes + 1.5 * $archiveBytes
$drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($Destination))
if ($drive.AvailableFreeSpace -lt $needed) {
    throw ('Not enough free disk space on {0}: {1:N1} GB free, about {2:N1} GB needed.' -f $drive.Name, ($drive.AvailableFreeSpace / 1GB), ($needed / 1GB))
}

foreach ($package in $missing) {
    $archive = Join-Path $ArchiveDirectory $package.archive
    if (-not (Test-Archive $package $archive)) { Get-Archive $package $archive }
    Write-Host ('Installing ' + $package.name)
    Install-Package $package $archive $Destination
    if (-not (Test-PackageInstalled $package $Destination)) {
        throw ($package.name + ': the files are not all in place after installing')
    }
}

if (-not $keepArchives) { Remove-Item -LiteralPath $ArchiveDirectory -Recurse -Force }
Write-Host ('Done. NVIDIA libraries are in ' + $Destination)
Write-Host 'Leave GpuDllDirectories empty and Mouse will use this folder. Run "mouse.exe doctor" to check.'
