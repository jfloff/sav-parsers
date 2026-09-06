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
  """Entity location. `page` is 0-indexed; `vertices` are (x, y), top-left
  origin, normalized to the page as Document AI rendered it. A corrected slot
  may extend past the page edge.
  """
  page: int
  vertices: list[tuple[float, float]]


@dataclass
class ParsedField:
  value: str | bool | int | None
  confidence: float
  bbox: BBox | None = None
