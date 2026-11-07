import json

import pytest

from gqteashmd.errors import (
    ConfigError,
    ProjectError,
    ShmdError,
    StageFailedError,
    StaleStageError,
    TaskError,
)
from gqteashmd.util.fileio import read_json, write_json_atomic, write_text_atomic


def test_exit_codes_distinguish_error_classes():
    assert ShmdError("x").exit_code == 1
    assert ConfigError("x").exit_code == 2
    assert ProjectError("x").exit_code == 3
    assert StaleStageError("x").exit_code == 3
    assert StageFailedError("x").exit_code == 4


def test_stale_stage_is_a_project_error():
    assert issubclass(StaleStageError, ProjectError)


def test_task_error_carries_retryable_flag():
    err = TaskError("scf failed", retryable=True)
    assert err.retryable is True
    assert str(err) == "scf failed"
    assert isinstance(err, ShmdError)


def test_write_text_atomic_creates_parents_and_leaves_no_temp(tmp_path):
    target = tmp_path / "a" / "b" / "out.txt"
    write_text_atomic(target, "hello\n")
    assert target.read_text() == "hello\n"
    assert [p.name for p in target.parent.iterdir()] == ["out.txt"]


def test_json_roundtrip_is_sorted_and_newline_terminated(tmp_path):
    target = tmp_path / "data.json"
    write_json_atomic(target, {"b": 1, "a": [1, 2]})
    text = target.read_text()
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')
    assert read_json(target) == {"a": [1, 2], "b": 1}


def test_read_json_propagates_decode_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        read_json(bad)


def test_write_bytes_atomic(tmp_path):
    from gqteashmd.util.fileio import write_bytes_atomic

    target = tmp_path / "sub" / "state.npz"
    write_bytes_atomic(target, b"\x00\x01")
    assert target.read_bytes() == b"\x00\x01"
    assert [p.name for p in target.parent.iterdir()] == ["state.npz"]


def test_failed_atomic_write_leaves_no_temp_file(tmp_path, monkeypatch):
    import os

    from gqteashmd.util.fileio import write_bytes_atomic

    def broken_replace(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", broken_replace)
    target = tmp_path / "out.json"
    with pytest.raises(OSError, match="No space"):
        write_bytes_atomic(target, b"data")
    assert list(tmp_path.iterdir()) == []
