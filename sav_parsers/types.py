from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedField:
  value: str | bool | int | None
  confidence: float
