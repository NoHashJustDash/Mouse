#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_baseline=a5f736224e85b32e25a8ffc52562333180396642
task_ort_hash=8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3
cd -- "$task_root"
export VCPKG_ROOT="$task_root/.toolchain/vcpkg"
export VCPKG_DISABLE_METRICS=1
if [[ ! -d "$VCPKG_ROOT/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
    git -C "$VCPKG_ROOT" fetch --depth=1 origin "$task_baseline"
    git -C "$VCPKG_ROOT" checkout --detach "$task_baseline"
fi
if [[ "$(git -C "$VCPKG_ROOT" rev-parse HEAD)" != "$task_baseline" ]]; then
    printf 'vcpkg must be checked out at %s\n' "$task_baseline" >&2
    exit 1
fi
if [[ ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
fi
task_archive="$task_root/.toolchain/downloads/onnxruntime-linux-x64-1.22.0.tgz"
task_sdk_parent="$task_root/.toolchain/sdks/onnxruntime"
mkdir -p -- "$(dirname -- "$task_archive")" "$task_sdk_parent"
if [[ ! -f "$task_archive" ]]; then
    task_partial="$(mktemp "$task_archive.XXXXXX")"
    trap 'rm -f -- "$task_partial"' EXIT
    curl --fail --location --retry 3 https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz -o "$task_partial"
    printf '%s  %s\n' "$task_ort_hash" "$task_partial" | sha256sum --check --status
    mv -- "$task_partial" "$task_archive"
    trap - EXIT
fi
# Observed hash of the official release asset; GitHub published no asset digest.
printf '%s  %s\n' "$task_ort_hash" "$task_archive" | sha256sum --check
if [[ ! -d "$task_sdk_parent/onnxruntime-linux-x64-1.22.0" ]]; then
    tar -xzf "$task_archive" -C "$task_sdk_parent"
fi
printf 'Pinned vcpkg and CPU ONNX Runtime 1.22.0 are ready.\n'
