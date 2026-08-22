#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8501}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install --upgrade pip
python scripts/install_torch.py --run
python -m pip install -e .
exec python -m streamlit run app.py --server.port "$PORT"
