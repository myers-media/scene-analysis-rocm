#!/usr/bin/env python3
"""Generate a LAN HTTPS cert if needed, then start Streamlit."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_analysis.ssl_cert import access_urls, write_self_signed_cert  # noqa: E402


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
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
