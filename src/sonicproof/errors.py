"""Errors Sonicproof raises on purpose: each has a stable ``code`` and a one-line, actionable message.

``InputError`` is about what the user passed (exit code 2); ``SetupError`` is about the machine — a missing
extra, a model that is not downloaded, ffmpeg not installed (exit code 3). Both subclass the builtin they
replace, so older ``except ValueError`` / ``except RuntimeError`` handlers keep working.
"""

from __future__ import annotations


class SonicproofError(Exception):
    code = "error"
    exit_code = 2

    def __init__(self, message: str, *, code: str | None = None, file: str | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.file = file

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "file": self.file}


class InputError(SonicproofError, ValueError):
    """Bad input. Codes: not_found, is_dir, empty, unreadable, no_audio, bad_value, no_prompt."""

    code = "bad_input"
    exit_code = 2


class SetupError(SonicproofError, RuntimeError):
    """The environment can't do it.

    Codes: missing_extra, model_unavailable, needs_ffmpeg, bad_device, port_busy.
    """

    code = "setup"
    exit_code = 3
