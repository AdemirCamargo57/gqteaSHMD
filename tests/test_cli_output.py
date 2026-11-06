"""Console output of real `python -m gqteashmd` processes (capsys cannot see these problems)."""

import json
import os
import subprocess
import sys

from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"


def _cli(args, cwd, env=None):
    return subprocess.run(
        [sys.executable, "-m", "gqteashmd", *args],
        cwd=cwd,
        env=env or dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_config_error_is_printed_exactly_once(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("project = [")
    proc = _cli(["validate", str(bad)], tmp_path)
    assert proc.returncode == 2
    assert proc.stderr.count("is not valid TOML") == 1
    assert proc.stderr.startswith("gqteashmd: error:")


def test_internal_error_has_no_traceback_on_console(tmp_path):
    script = (
        "import sys, gqteashmd.cli as cli\n"
        "def boom(args):\n"
        "    raise RuntimeError('kaboom')\n"
        "cli._cmd_validate = boom\n"
        "sys.exit(cli.main(['validate', 'x.toml']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 1
    assert proc.stderr.count("internal error: RuntimeError: kaboom") == 1
    assert "Traceback" not in proc.stderr


def test_stage_failure_is_printed_once_and_logged_to_file(tmp_path, write_input):
    path = write_input(tmp_path, execution=FAST)
    env = dict(os.environ)
    env[BEHAVIOUR_ENV] = json.dumps({"fail": ["sampling:t000"]})
    proc = _cli(["run", str(path), "--pipeline", FAKE], tmp_path, env=env)
    assert proc.returncode == 4
    assert proc.stderr.count("Stage 'sampling' failed") == 1
    log_text = (tmp_path / "run" / "logs" / "gqteashmd.log").read_text()
    assert "Stage 'sampling' failed" in log_text
