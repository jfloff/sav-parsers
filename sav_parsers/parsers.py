from __future__ import annotations

from pathlib import Path

from .classify import classify
from .em import parse_em
from .fpb_mod1 import parse_fpb_mod1
from .fpb_mod4 import parse_fpb_mod4
from .types import DocType

_PARSER_MAP = {
  DocType.EXAME_MEDICO: parse_em,
  DocType.FPB_MODELO_1: parse_fpb_mod1,
  DocType.FPB_MODELO_4: parse_fpb_mod4,
}


def parse_document(pdf_path: str | Path, doc_type: DocType) -> dict | None:
  """Dispatch to the parser registered for `doc_type`, or None if unmapped."""
  if parser := _PARSER_MAP.get(doc_type):
    return parser(pdf_path)
  return None


def classify_and_parse(pdf_path: str | Path) -> tuple[DocType, dict | None]:
  """Classify then parse. Returns (doc_type, fields) or (doc_type, None)."""
  doc_type = classify(pdf_path)
  return doc_type, parse_document(pdf_path, doc_type)
