"""scripts/run_then_stop_pod.sh, against a stand-in for runpodctl that records its calls.

Nothing here talks to RunPod: the wrapper is pointed at the stand-in by path, and at a fake
copy of the main process's environment.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_then_stop_pod.sh"

# Fails on the current syntax ("pod stop ID") and accepts the older one ("stop pod ID").
OLD_CLI = 'if [ "$1" = pod ]; then exit 1; fi'


def _fake_runpodctl(path: Path, calls: Path, body: str = "") -> Path:
    path.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\n{body}\nexit ${{RUNPODCTL_EXIT:-0}}\n')
    path.chmod(0o755)
    return path


class FakePod:
    """Where the stand-in lives, and the environment the wrapper runs in.

    The environment is built from scratch, never copied from this process: pytest prints
    fixtures when a test fails, and the real environment holds credentials.
    """

    def __init__(self, directory: Path):
        self.dir = directory
        self.calls = directory / "runpodctl.calls"
        self.log = directory / "job.log"
        self.env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "RUNPODCTL": str(_fake_runpodctl(directory / "runpodctl", self.calls)),
            "RUNPOD_POD_ID": "pod123",
            "PID1_ENVIRON": str(directory / "no-such-file"),
            "STOP_ATTEMPTS": "2",
            "STOP_RETRY_SECONDS": "0",
        }

    def __repr__(self) -> str:
        return f"FakePod({self.dir})"


@pytest.fixture
def pod(tmp_path):
    return FakePod(tmp_path)


def _run(pod, *args):
    command = ["bash", str(SCRIPT), "--log", str(pod.log), *args]
    return subprocess.run(command, env=pod.env, capture_output=True, text=True, timeout=60)


def _calls(pod) -> list[str]:
    return pod.calls.read_text().splitlines() if pod.calls.exists() else []


def test_stops_the_pod_after_success(pod):
    result = _run(pod, "--", "sh", "-c", "echo hello from the job")
    assert result.returncode == 0
    assert _calls(pod) == ["pod stop pod123"]
    log = pod.log.read_text()
    assert "hello from the job" in log
    assert log.index("hello from the job") < log.index("stopping this pod")


def test_stops_the_pod_after_failure_and_passes_the_status_on(pod):
    result = _run(pod, "--", "sh", "-c", "exit 3")
    assert result.returncode == 3
    assert _calls(pod) == ["pod stop pod123"]
    assert "job ended with exit status 3" in pod.log.read_text()


def test_deadline_kills_the_job_then_stops_the_pod(pod):
    marker = pod.dir / "job-finished"
    started = time.monotonic()
    result = _run(pod, "--max-hours", "0.0003", "--", "sh", "-c", f"sleep 30; touch {marker}")
    assert time.monotonic() - started < 15
    assert result.returncode != 0
    assert "deadline" in pod.log.read_text()
    assert not marker.exists()
    assert _calls(pod) == ["pod stop pod123"]


def test_output_reaches_the_log_while_the_job_runs(pod):
    # No flush in the job: the wrapper has to make Python unbuffered.
    job = "print('first line'); import time; time.sleep(5)"
    command = ["bash", str(SCRIPT), "--log", str(pod.log), "--", sys.executable, "-c", job]
    process = subprocess.Popen(command, env=pod.env)
    try:
        deadline = time.monotonic() + 4
        seen = False
        while time.monotonic() < deadline and not seen:
            seen = pod.log.exists() and "first line" in pod.log.read_text()
            time.sleep(0.1)
        still_running = process.poll() is None
    finally:
        process.wait(timeout=30)
    assert seen and still_running
    assert _calls(pod) == ["pod stop pod123"]


@pytest.mark.parametrize("missing", ["runpodctl", "pod id"])
def test_refuses_to_start_a_job_it_could_not_stop(pod, missing):
    if missing == "runpodctl":
        pod.env["RUNPODCTL"] = str(pod.dir / "not-installed")
    else:
        del pod.env["RUNPOD_POD_ID"]
    marker = pod.dir / "job-ran"
    result = _run(pod, "--", "touch", str(marker))
    assert result.returncode == 2
    assert "not starting a job" in result.stderr
    assert not marker.exists()
    assert _calls(pod) == []


def test_reads_the_pod_id_from_the_main_process_and_keeps_the_key_quiet(pod):
    del pod.env["RUNPOD_POD_ID"]
    environ = pod.dir / "environ"
    environ.write_bytes(b"HOME=/root\0RUNPOD_POD_ID=podXYZ\0RUNPOD_API_KEY=sk-not-for-logs\0")
    pod.env["PID1_ENVIRON"] = str(environ)
    result = _run(pod, "--", "sh", "-c", 'test -n "$RUNPOD_API_KEY" && echo key reached the job')
    assert result.returncode == 0
    assert _calls(pod) == ["pod stop podXYZ"]
    assert "key reached the job" in pod.log.read_text()
    assert "sk-not-for-logs" not in result.stdout + result.stderr + pod.log.read_text()


def test_falls_back_to_the_older_runpodctl_syntax(pod):
    _fake_runpodctl(Path(pod.env["RUNPODCTL"]), pod.calls, OLD_CLI)
    result = _run(pod, "--", "true")
    assert result.returncode == 0
    assert _calls(pod) == ["pod stop pod123", "stop pod pod123"]
    assert "stop requested" in pod.log.read_text()


def test_keeps_trying_and_says_so_when_the_pod_will_not_stop(pod):
    pod.env["RUNPODCTL_EXIT"] = "1"
    result = _run(pod, "--", "true")
    assert result.returncode == 0  # still the job's status
    assert _calls(pod) == ["pod stop pod123", "stop pod pod123"] * 2
    assert "COULD NOT STOP THE POD" in pod.log.read_text()


def test_check_reports_without_running_or_stopping_anything(pod):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"], env=pod.env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0
    assert result.stdout.startswith("ok:")
    assert _calls(pod) == []
    assert not pod.log.exists()
