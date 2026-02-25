#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/outputs/.cache/matplotlib" "$ROOT_DIR/outputs/.cache/fontconfig"
export MPLCONFIGDIR="$ROOT_DIR/outputs/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT_DIR/outputs/.cache"

python3 examples/01_forward_poisson_basis.py
