from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DocType(StrEnum):
  EM       = "exame_medico"
  FPB_MOD1 = "fpb_modelo_1"
  FPB_MOD4 = "fpb_modelo_4"
  OUTROS   = "outros"


@dataclass
class ParsedField:
  value: str | bool | int | None
  confidence: float
