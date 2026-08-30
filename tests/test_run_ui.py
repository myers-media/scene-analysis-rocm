from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

_RUN_UI = Path(__file__).resolve().parents[1] / "scripts" / "run_ui.py"
_spec = importlib.util.spec_from_file_location("run_ui", _RUN_UI)
assert _spec and _spec.loader
run_ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_ui)


def test_stop_child_returns_if_already_exited():
    proc = MagicMock()
    proc.poll.return_value = 0
    run_ui.stop_child(proc)
    proc.kill.assert_not_called()


def test_stop_child_force_kills_when_wait_times_out(monkeypatch):
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 4242
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="streamlit", timeout=1)

    monkeypatch.setattr(run_ui.sys, "platform", "linux")
    monkeypatch.setattr(proc, "send_signal", MagicMock())
    run_ui.stop_child(proc, grace_s=0.1)
    proc.kill.assert_called()
