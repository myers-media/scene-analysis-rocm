#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8501}"

venv_python() {
  if [[ -x .venv/bin/python ]]; then
    echo ".venv/bin/python"
  elif [[ -f .venv/Scripts/python.exe ]]; then
    echo ".venv/Scripts/python.exe"
  else
    return 1
  fi
}

host_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    echo "Python 3.10+ is required to create .venv" >&2
    exit 1
  fi
}

if ! PY="$(venv_python)"; then
  "$(host_python)" -m venv .venv
  PY="$(venv_python)"
fi

"$PY" -m pip install --upgrade pip
"$PY" scripts/install_torch.py --run
"$PY" -m pip install -e .
exec "$PY" scripts/run_ui.py --port "$PORT"
