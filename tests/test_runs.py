"""dipgpe.runs: permanent run records, index, queries and backup."""

import json

import pytest

from dipgpe import runs


def test_run_records_meta_log_result_data_and_manifest(tmp_path, capsys):
    with runs.Run("demo", {"a_s": 92.0, "cells": [4.2, 4.5], "out": tmp_path}, root=tmp_path) as run:
        print("hello from the run")
        run.result(c_plus=2.03)
        run.file("scan.json").write_text("{}")
        run.data("psi.bin").write_bytes(b"\x00" * 16)
    d = run.dir
    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "finished" and meta["params"]["a_s"] == 92.0 and meta["wall_s"] >= 0
    assert meta["params"]["out"] == str(tmp_path) and "python" in meta["environment"]
    assert "hello from the run" in (d / "log.txt").read_text()
    assert json.loads((d / "result.json").read_text()) == {"c_plus": 2.03}
    manifest = (d / "MANIFEST").read_text()
    for name in ("meta.json", "log.txt", "result.json", "scan.json", "data/psi.bin"):
        assert name in manifest
    entry = runs.index(tmp_path)[-1]
    assert entry["id"] == run.id and entry["result"]["c_plus"] == 2.03


def test_failed_run_is_marked_and_ids_never_collide(tmp_path):
    with pytest.raises(ValueError):
        with runs.Run("demo", {"x": 1}, root=tmp_path):
            raise ValueError("boom")
    with runs.Run("demo", {"x": 1}, root=tmp_path) as r2:
        pass
    with runs.Run("demo", {"x": 1}, root=tmp_path) as r3:
        pass
    rows = runs.index(tmp_path)
    assert [r["status"] for r in rows] == ["failed", "finished", "finished"]
    assert "boom" in json.loads((tmp_path / rows[0]["id"] / "meta.json").read_text())["error"]
    assert len({r["id"] for r in rows}) == 3 and r2.id != r3.id


def test_find_latest_resolve_and_backup(tmp_path):
    root, dest = tmp_path / "runs", tmp_path / "backup"
    for a in (91.0, 92.0):
        with runs.Run("scan", {"a_s": a, "dx": 0.25}, root=root) as r:
            r.data("big.bin").write_bytes(bytes(100))
            r.result(a=a)
    assert [x["params"]["a_s"] for x in runs.find("scan", root=root)] == [91.0, 92.0]
    assert runs.find("scan", root=root, a_s=92) and not runs.find("scan", root=root, a_s=93.0)
    last = runs.latest("scan", root=root, dx=0.25)
    assert runs.resolve(last.name, "big.bin", root=root) == last / "data" / "big.bin"
    assert runs.backup(dest, root=root)[0] > 0
    copied, skipped = runs.backup(dest, root=root)
    assert copied == 0 and skipped > 0
    assert (dest / last.name / "data" / "big.bin").exists()


def test_register_existing_files(tmp_path):
    src = tmp_path / "old"
    src.mkdir()
    (src / "a.json").write_text('{"x": 1}')
    (src / "big.bin").write_bytes(bytes(2048))
    (src / "frames").mkdir()
    (src / "frames" / "f0.bin").write_bytes(b"1")
    d = runs.register("legacy_a", [src / "a.json", src / "big.bin", src / "frames"], params={"N": 4},
                      root=tmp_path / "runs", large_bytes=1024, commit="abc")
    assert (d / "a.json").exists() and (d / "data" / "big.bin").exists() and (d / "data" / "frames" / "f0.bin").exists()
    assert "data/big.bin" in (d / "MANIFEST").read_text()
    row = runs.find("legacy_a", root=tmp_path / "runs", N=4)[0]
    assert row["git_commit"] == "abc" and row["registered"]


def test_repair_marks_killed_runs(tmp_path):
    run = runs.Run("killed", {"x": 1}, root=tmp_path, quiet=True)
    run.__enter__()                                   # simulate a process killed inside the run
    import sys
    sys.stdout, sys.stderr = run._stdout, run._stderr
    run._log.close()
    assert runs.repair(tmp_path) == [run.id]
    assert runs.index(tmp_path)[-1]["status"] == "interrupted"
    assert runs.repair(tmp_path) == []


def test_annotate_invalid_run_is_skipped_by_find(tmp_path):
    with runs.Run("calc", {"a": 1}, root=tmp_path, quiet=True) as bad:
        pass
    with runs.Run("calc", {"a": 1}, root=tmp_path, quiet=True) as good:
        pass
    runs.annotate(bad.id, "kernel bug", invalid=True, root=tmp_path)
    assert [r["id"] for r in runs.find("calc", root=tmp_path, a=1)] == [good.id]
    row = [r for r in runs.index(tmp_path) if r["id"] == bad.id][0]
    assert row["invalid"] and row["notes"] == ["kernel bug"]
    assert "kernel bug" in (bad.dir / "meta.json").read_text()
