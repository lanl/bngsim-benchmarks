"""`bench fetch-models`: every pinned model file lands in the cache, sha256-checked.

The download is faked, except in the one `network` test, which fetches the real
pilot pins into a fresh cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bench import cli, fetch, specs

REPO = Path(__file__).resolve().parents[1]
BODY = b"begin model\nend model\n"
SHA = hashlib.sha256(BODY).hexdigest()


@pytest.fixture
def problems(tmp_path, spec_data, write_spec):
    """Four problems, three cached files: model_c names model_a's bytes under the
    same file name (one entry), model_d names them under another (its own entry)."""
    a = dict(spec_data, id="model_a")
    a["source"] = dict(spec_data["source"], path="models/a/a.bngl", sha256=SHA)
    b = dict(spec_data, id="model_b")
    b_body = BODY + b"# b\n"
    b["source"] = dict(spec_data["source"], path="models/b/b.bngl",
                       sha256=hashlib.sha256(b_body).hexdigest())
    c = dict(spec_data, id="model_c")
    c["source"] = dict(a["source"], path="models/c/a.bngl")
    d = dict(spec_data, id="model_d")
    d["source"] = dict(a["source"], path="models/d/renamed.bngl")
    for spec in (a, b, c, d):
        write_spec(spec)
    bodies = {"a.bngl": BODY, "b.bngl": b_body, "renamed.bngl": BODY}
    return tmp_path / "problems", bodies


@pytest.fixture
def downloads(monkeypatch, problems):
    """Serve the fixture's bodies; record each URL fetched."""
    _, bodies = problems
    calls: list[str] = []

    def fake_download(url, dest, timeout=120):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(bodies[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(fetch, "_download", fake_download)
    return calls


def _fetch(problems_dir, cache, *patterns):
    return cli.main(["fetch-models", "--problems", str(problems_dir),
                     "--cache", str(cache), *patterns])


def test_cache_state_is_missing_cached_or_corrupt(tmp_path):
    src = {"repo": "o/r", "commit": "a" * 40, "path": "m/m.bngl", "sha256": SHA}
    assert fetch.model_cache_state(src, tmp_path) == "missing"
    dest = fetch.model_cache_path(src, tmp_path)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(BODY)
    assert fetch.model_cache_state(src, tmp_path) == "cached"
    dest.write_bytes(b"bit rot")
    assert fetch.model_cache_state(src, tmp_path) == "corrupt"


def test_a_fresh_cache_gets_every_file_once(tmp_path, problems, downloads, monkeypatch,
                                            capsys):
    problems_dir, bodies = problems
    assert _fetch(problems_dir, tmp_path / "cache") == 0
    assert len(downloads) == 3  # model_c shares model_a's entry
    assert "3 model files for 4 problems: 0 already cached, 3 fetched, 0 failed" in (
        capsys.readouterr().out)
    # Every problem's job now finds its file in the cache: no network at run time.
    def no_download(*a, **k):
        raise AssertionError("a job would have gone to the network")

    monkeypatch.setattr(fetch, "_download", no_download)
    for p in specs.load_all(problems_dir):
        got = fetch.model_file(p.source, tmp_path / "cache")
        assert got.name == Path(p.source["path"]).name
        assert hashlib.sha256(got.read_bytes()).hexdigest() == p.source["sha256"]


def test_a_second_run_downloads_nothing(tmp_path, problems, downloads, capsys):
    problems_dir, _ = problems
    _fetch(problems_dir, tmp_path / "cache")
    downloads.clear()
    assert _fetch(problems_dir, tmp_path / "cache") == 0
    assert downloads == []
    assert "3 already cached, 0 fetched" in capsys.readouterr().out


def test_a_tampered_cache_file_is_fetched_again(tmp_path, problems, downloads, capsys):
    problems_dir, _ = problems
    _fetch(problems_dir, tmp_path / "cache")
    src = next(p.source for p in specs.load_all(problems_dir) if p.id == "model_b")
    cached = fetch.model_cache_path(src, tmp_path / "cache")
    cached.write_bytes(b"edited in place")
    downloads.clear()
    capsys.readouterr()
    assert _fetch(problems_dir, tmp_path / "cache") == 0
    assert len(downloads) == 1 and downloads[0].endswith("/models/b/b.bngl")
    assert hashlib.sha256(cached.read_bytes()).hexdigest() == src["sha256"]
    assert "refetched" in capsys.readouterr().out


def test_a_wrong_sha256_fails_loudly_and_caches_nothing(
        tmp_path, problems, monkeypatch, capsys):
    problems_dir, _ = problems

    def upstream_changed(url, dest, timeout=120):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"not the pinned bytes")

    monkeypatch.setattr(fetch, "_download", upstream_changed)
    assert _fetch(problems_dir, tmp_path / "cache") == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "sha256 mismatch" in out
    assert "model_a, model_c" in out  # names every problem the bad file breaks
    assert "0 fetched, 3 failed" in out
    assert not list((tmp_path / "cache" / "models").rglob("*.bngl"))


def test_patterns_select_problems_by_their_jobs(tmp_path, problems, downloads):
    problems_dir, _ = problems
    assert _fetch(problems_dir, tmp_path / "cache", "model_b") == 0
    assert len(downloads) == 1 and downloads[0].endswith("/models/b/b.bngl")


def test_doctor_reports_a_full_cache_as_ok(tmp_path, problems, downloads, capsys):
    problems_dir, _ = problems
    _fetch(problems_dir, tmp_path / "cache")
    capsys.readouterr()
    cli._report_model_cache(specs.load_all(problems_dir), tmp_path / "cache")
    assert capsys.readouterr().out.startswith("OK   3 model files cached and sha256-checked")


def test_doctor_warns_on_missing_and_corrupt_files(tmp_path, problems, downloads, capsys):
    problems_dir, _ = problems
    _fetch(problems_dir, tmp_path / "cache", "model_a")  # model_b, model_d stay missing
    src = next(p.source for p in specs.load_all(problems_dir) if p.id == "model_a")
    fetch.model_cache_path(src, tmp_path / "cache").write_bytes(b"bit rot")
    capsys.readouterr()
    cli._report_model_cache(specs.load_all(problems_dir), tmp_path / "cache")
    out = capsys.readouterr().out
    assert out.startswith("WARN 3 model files: 2 not cached, 1 with the wrong sha256")
    assert "pixi run fetch-models" in out


@pytest.mark.network
def test_the_committed_pins_fetch_byte_identical_into_a_fresh_cache(tmp_path):
    """Acceptance for pin-and-fetch: from nothing, every committed spec's file
    downloads at its pinned commit and hashes to its pinned sha256."""
    problems_dir = REPO / "problems"
    assert _fetch(problems_dir, tmp_path / "cache") == 0
    for p in specs.load_all(problems_dir):
        assert fetch.model_cache_state(p.source, tmp_path / "cache") == "cached", p.id
