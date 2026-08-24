from __future__ import annotations

from scene_analysis.ssl_cert import access_urls, collect_sans


def test_collect_sans_includes_localhost():
    names, ips = collect_sans()
    assert "localhost" in names
    assert "127.0.0.1" in ips
    assert names == sorted(names, key=str.lower)


def test_write_self_signed_cert(tmp_path):
    pytest = __import__("pytest")
    cryptography = pytest.importorskip("cryptography")
    from scene_analysis.ssl_cert import cert_paths, write_self_signed_cert

    cert, key = write_self_signed_cert(tmp_path)
    assert cert.is_file() and key.is_file()
    assert b"BEGIN CERTIFICATE" in cert.read_bytes()
    assert b"BEGIN" in key.read_bytes()
    again, _ = write_self_signed_cert(tmp_path)
    assert again == cert


def test_access_urls_are_https():
    urls = access_urls(8501)
    assert urls
    assert all(u.startswith("https://") for u in urls)
    assert any(":8501" in u for u in urls)
    assert any("localhost" in u for u in urls)
