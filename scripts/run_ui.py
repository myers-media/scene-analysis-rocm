#!/usr/bin/env python3
"""Generate a LAN HTTPS cert if needed, then start Streamlit.

Streamlit is launched as a child process so Ctrl+C can kill the whole tree
(OpenCV/DirectShow and the script runner often ignore SIGINT on Windows).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_analysis.ssl_cert import access_urls, write_self_signed_cert  # noqa: E402


def stop_child(proc: subprocess.Popen, grace_s: float = 6.0) -> None:
    """Stop Streamlit and any camera/worker children. Force-kill if they hang."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    sigs = [signal.SIGINT, signal.SIGTERM]
    sigkill = getattr(signal, "SIGKILL", None)
    if sigkill is not None:
        sigs.append(sigkill)
    for sig in sigs:
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(sig)
        except OSError:
            return
        try:
            proc.wait(timeout=3.0 if sigkill is not None and sig == sigkill else 2.0)
            return
        except subprocess.TimeoutExpired:
            continue
    proc.kill()


def run_streamlit(cmd: list[str]) -> int:
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        print("\nStopping Streamlit (releasing cameras)…", flush=True)
        stop_child(proc)
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Streamlit with a generated HTTPS certificate.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("STREAMLIT_SERVER_PORT", "8501")))
    parser.add_argument("--address", default="0.0.0.0", help="Bind address (0.0.0.0 for remote devices)")
    parser.add_argument("--http", action="store_true", help="Skip TLS (localhost-only camera snapshot)")
    args = parser.parse_args()
    app = ROOT / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        "true",
        "--runner.fastReruns",
        "true",
    ]
    if not args.http:
        cert, key = write_self_signed_cert()
        cmd.extend(
            [
                "--server.sslCertFile",
                str(cert),
                "--server.sslKeyFile",
                str(key),
            ]
        )
        print("HTTPS is on. Browsers will warn about the self-signed certificate — continue once.")
        print("Remote camera snapshot (browser webcam) needs one of:")
        for url in access_urls(args.port):
            print(f"  {url}")
        print("Open a firewall hole for TCP {0} if another machine cannot connect.".format(args.port))
    print("Ctrl+C stops the server (kills Streamlit and OpenCV camera children).")
    return run_streamlit(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
