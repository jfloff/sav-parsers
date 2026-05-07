from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class DocType(StrEnum):
  FPB_MOD1 = "fpb-mod1"
  FPB_MOD4 = "fpb-mod4"
  UNKNOWN  = "unknown"


def classify(pdf_path: str | Path) -> DocType:
  """Return the document type for pdf_path.

  TODO: real Document AI classifier — currently always returns FPB_MOD1.
  """
  return DocType.FPB_MOD1
