from __future__ import annotations

import ipaddress
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_CERT_DIR = Path(__file__).resolve().parents[2] / "certs"
CERT_NAME = "cert.pem"
KEY_NAME = "key.pem"
VALID_DAYS = 365


def collect_sans(extra_hosts: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Hostnames and IPs to put on a self-signed cert for LAN HTTPS."""
    names: set[str] = {"localhost"}
    ips: set[str] = {"127.0.0.1"}
    for raw in (
        socket.gethostname(),
        socket.getfqdn(),
        os.environ.get("COMPUTERNAME"),
        os.environ.get("HOSTNAME"),
        *(extra_hosts or ()),
    ):
        if not raw:
            continue
        host = raw.strip().rstrip(".")
        if host:
            names.add(host)
            names.add(host.split(".")[0])
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if ":" in addr:
                if not addr.lower().startswith("fe80"):
                    ips.add(addr)
            elif not addr.startswith("127."):
                ips.add(addr)
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ips.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    return sorted(names, key=str.lower), sorted(ips)


def cert_paths(cert_dir: Path | None = None) -> tuple[Path, Path]:
    root = Path(cert_dir) if cert_dir else DEFAULT_CERT_DIR
    return root / CERT_NAME, root / KEY_NAME


def _load_san_text(cert_path: Path) -> str:
    from cryptography import x509

    data = cert_path.read_bytes()
    cert = x509.load_pem_x509_certificate(data)
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return ""
    return ",".join(str(n) for n in san)


def cert_covers_current_sans(cert_path: Path, names: list[str], ips: list[str]) -> bool:
    if not cert_path.is_file():
        return False
    try:
        from cryptography import x509

        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        now = datetime.now(timezone.utc)
        expiry = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after.replace(tzinfo=timezone.utc)
        if expiry < now + timedelta(days=14):
            return False
        blob = _load_san_text(cert_path).lower()
        for host in names:
            if host.lower() not in blob:
                return False
        for ip in ips:
            if ip.lower() not in blob:
                return False
        return True
    except Exception:
        return False


def write_self_signed_cert(
    cert_dir: Path | None = None,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Create certs/cert.pem and certs/key.pem if missing, expired, or SANs changed."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    cert_path, key_path = cert_paths(cert_dir)
    names, ips = collect_sans()
    if not force and cert_path.is_file() and key_path.is_file() and cert_covers_current_sans(cert_path, names, ips):
        return cert_path, key_path

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    cn = names[0] if names else "localhost"
    for candidate in names:
        if candidate.lower() not in {"localhost"}:
            cn = candidate
            break
    san_entries: list[x509.GeneralName] = []
    for host in names:
        san_entries.append(x509.DNSName(host))
    for ip in ips:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(ip)))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def access_urls(port: int = 8501) -> list[str]:
    names, ips = collect_sans()
    hosts = []
    for host in names + ips:
        if ":" in host and not host.startswith("["):
            hosts.append(f"https://[{host}]:{port}")
        else:
            hosts.append(f"https://{host}:{port}")
    return hosts


def main() -> None:
    cert, key = write_self_signed_cert()
    print(f"certificate: {cert}")
    print(f"key: {key}")
    for url in access_urls():
        print(url)


if __name__ == "__main__":
    main()
