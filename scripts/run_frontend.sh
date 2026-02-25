#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/outputs/.cache/matplotlib" "$ROOT_DIR/outputs/.cache/fontconfig"
export MPLCONFIGDIR="$ROOT_DIR/outputs/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT_DIR/outputs/.cache"
export MPLBACKEND="Agg"

PYTHON_BIN="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

if ! "$PYTHON_BIN" -c "import PySide6" >/dev/null 2>&1; then
  echo "PySide6 is not installed for: $PYTHON_BIN"
  echo "Install it with:"
  echo "  $PYTHON_BIN -m pip install PySide6"
  exit 1
fi

"$PYTHON_BIN" frontend/app.py
