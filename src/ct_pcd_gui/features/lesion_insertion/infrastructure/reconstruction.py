from __future__ import annotations

import os
import queue
import shlex
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from threading import Event

from ..domain.models import ValidationError


class ReconstructionCancelled(RuntimeError):
    pass


class ExternalReconstructionRunner:
    """Run an explicitly configured reconstruction executable without a shell."""

    def run(
        self,
        command_template: str,
        *,
        input_directory: str | Path,
        output_directory: str | Path,
        session_path: str | Path,
        cancel_event: Event,
        log: Callable[[str], None],
    ) -> Path:
        output = Path(output_directory).expanduser().resolve(strict=False)
        if output.exists() and any(output.iterdir()):
            raise ValidationError(
                "The reconstruction output directory is not empty; choose a new location."
            )
        output.mkdir(parents=True, exist_ok=True)
        formatted = command_template.format(
            input=str(Path(input_directory).resolve()),
            output=str(output),
            session=str(Path(session_path).resolve()),
        )
        arguments = shlex.split(formatted, posix=os.name != "nt")
        if os.name == "nt":
            arguments = [
                item[1:-1]
                if len(item) >= 2 and item[0] == item[-1] and item[0] in {'"', "'"}
                else item
                for item in arguments
            ]
        if not arguments:
            raise ValidationError("The reconstruction command is empty.")
        log(f"Starting configured reconstruction command: {arguments[0]}")
        process = subprocess.Popen(  # noqa: S603 - user-configured executable.
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        lines: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    lines.put(line.rstrip())
            finally:
                lines.put(None)

        reader = threading.Thread(
            target=read_output,
            name="lesion-reconstruction-output",
            daemon=True,
        )
        reader.start()
        stream_closed = False
        try:
            while True:
                if cancel_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise ReconstructionCancelled("Reconstruction was cancelled.")
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    line = ""
                if line is None:
                    stream_closed = True
                elif line:
                    log(line)
                return_code = process.poll()
                if return_code is not None and stream_closed:
                    if return_code != 0:
                        raise RuntimeError(
                            f"The reconstruction command exited with code {return_code}."
                        )
                    break
        finally:
            if process.poll() is None:
                process.terminate()
            reader.join(timeout=1)
            process.stdout.close()
        return output
