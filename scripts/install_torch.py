#!/usr/bin/env python3
"""Print (or run) the PyTorch install command that matches this machine."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_analysis.device import read_rocm_version  # noqa: E402


ROCM_INDEX = {
    "6.1": "https://download.pytorch.org/whl/rocm6.1",
    "6.2": "https://download.pytorch.org/whl/rocm6.2",
    "6.3": "https://download.pytorch.org/whl/rocm6.3",
    "6.4": "https://download.pytorch.org/whl/rocm6.3",
    "7.0": "https://download.pytorch.org/whl/rocm7.0",
    "7.1": "https://download.pytorch.org/whl/rocm7.1",
    "7.2": "https://download.pytorch.org/whl/rocm7.0",
}


def major_minor(version: str) -> str:
    parts = version.replace("-", ".").split(".")
    return ".".join(parts[:2])


def command_for_host() -> list[str]:
    system = platform.system()
    if system == "Windows":
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
    version = read_rocm_version()
    if version:
        key = major_minor(version)
        index = ROCM_INDEX.get(key, "https://download.pytorch.org/whl/rocm6.3")
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            index,
        ]
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute the pip command")
    args = parser.parse_args()
    cmd = command_for_host()
    print(" ".join(cmd))
    if not args.run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
