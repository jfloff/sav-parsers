from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DocType(StrEnum):
  EXAME_MEDICO            = "exame_medico"
  FPB_MODELO_1            = "fpb_modelo_1"
  FPB_MODELO_4            = "fpb_modelo_4"
  ATESTADO_RESIDENCIA     = "atestado_residencia"
  CERTIDAO_MATRICULA      = "certidao_matricula"
  DOCUMENTO_IDENTIFICACAO = "documento_identificacao"
  OUTROS                  = "outros"


@dataclass
class BBox:
  """Entity location. `page` is 0-indexed; `vertices` are (x, y) in [0, 1],
  top-left origin, normalized to the page — multiply by page width/height
  (PDF points or image pixels) to place overlays.
  """
  page: int
  vertices: list[tuple[float, float]]


@dataclass
class ParsedField:
  value: str | bool | int | None
  confidence: float
  bbox: BBox | None = None
