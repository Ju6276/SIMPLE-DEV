"""Compatibility shim for the JEPAMEMORY macro-action encoder source tree."""

from __future__ import annotations

from pathlib import Path

_SOURCE = Path("/home/d013/桌面/JEPAMEMORY/starVLA/model/modules/world_model/macro_action_encoder.py")
exec(compile(_SOURCE.read_text(encoding="utf-8"), str(_SOURCE), "exec"), globals(), globals())
