"""Shell-free, cancellable process execution with disk-backed stdout/stderr."""
from __future__ import annotations

import codecs
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .models import JobContext

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def resolve_executable(value: str) -> Path:
    value = value.strip()
    if not value:
        raise ValueError("Select the FreeCT v2 fct-wfbp executable first.")
    candidate = Path(value).expanduser()
    if not candidate.is_file():
        located = shutil.which(value)
        if not located:
            raise ValueError(f"FreeCT executable does not exist: {value}")
        candidate = Path(located)
    candidate = candidate.resolve()
    if os.name == "nt" and candidate.suffix.lower() != ".exe":
        raise ValueError("Select a native FreeCT .exe, not a batch file or shell command.")
    if os.name != "nt" and not os.access(candidate, os.X_OK):
        raise ValueError(f"FreeCT file is not executable: {candidate}")
    return candidate


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            # FreeCT is a standalone executable. /T also covers any child processes.
            taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/taskkill.exe"
            subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=5)


def run_process(
    command: list[str], directory: Path, stem: str, context: JobContext,
    *, log_stdout: bool = True, timeout: float | None = None,
) -> Path:
    """Return stdout's path. Stderr is separate so it cannot corrupt YAML output.

    Tailing files rather than readline() on a pipe avoids deadlock on verbose
    processes and lets cancellation interrupt a completely silent executable.
    """
    context.check()
    stdout_path = directory / f"{stem}.stdout.log"
    stderr_path = directory / f"{stem}.stderr.log"
    options: dict = {"cwd": str(directory), "stdin": subprocess.DEVNULL, "shell": False}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    context.report(f"Starting FreeCT ({stem}); logs: {directory}")
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, **options)
        started = time.monotonic()
        try:
            with stdout_path.open("rb") as out_reader, stderr_path.open("rb") as err_reader:
                readers = [(out_reader, log_stdout), (err_reader, True)]
                decoders = [codecs.getincrementaldecoder("utf-8")("replace") for _ in readers]
                pending = ["", ""]
                while True:
                    context.check()
                    if timeout is not None and time.monotonic() - started > timeout:
                        raise TimeoutError(f"FreeCT {stem} exceeded {timeout:g} seconds.")
                    finished = process.poll() is not None
                    remaining = False
                    for index, (reader, display) in enumerate(readers):
                        data = reader.read(65536)
                        remaining = remaining or bool(data)
                        if not display:
                            continue
                        pending[index] += decoders[index].decode(data)
                        lines = pending[index].replace("\r", "\n").split("\n")
                        pending[index] = lines.pop()
                        for line in lines:
                            if line.strip():
                                context.report(_ANSI.sub("", line))
                        if len(pending[index]) > 16384 or (finished and not data):
                            if pending[index].strip():
                                context.report(_ANSI.sub("", pending[index]))
                            pending[index] = ""
                    if finished and not remaining:
                        break
                    context.cancel_event.wait(0.1)
            if process.returncode != 0:
                raise RuntimeError(
                    f"FreeCT exited with code {process.returncode}. See {stderr_path} "
                    f"and {stdout_path}."
                )
            context.check()
        finally:
            _stop(process)
    return stdout_path
