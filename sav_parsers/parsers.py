from __future__ import annotations

from pathlib import Path

from .classify import classify
from .em import parse_em
from .fpb_mod1 import parse_fpb_mod1
from .fpb_mod4 import parse_fpb_mod4
from .types import DocType

_PARSER_MAP = {
    DocType.EM: parse_em,
    DocType.FPB_MOD1: parse_fpb_mod1,
    DocType.FPB_MOD4: parse_fpb_mod4,
}


def parse_document(pdf_path: str | Path, doc_type: DocType) -> dict | None:
    """
    Parses a document with the appropriate parser based on the given doc_type.

    Args:
        pdf_path: The path to the PDF file.
        doc_type: The type of the document.

    Returns:
        A dictionary with the parsed data, or None if no parser is found.
    """
    if parser := _PARSER_MAP.get(doc_type):
        return parser(pdf_path)
    return None


def classify_and_parse(pdf_path: str | Path) -> tuple[DocType, dict | None]:
    """
    Classifies a document and then parses it with the appropriate parser.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A tuple containing the detected DocType and the parsed data,
        or the DocType and None if no parser is found for the detected type.
    """
    doc_type = classify(pdf_path)
    return doc_type, parse_document(pdf_path, doc_type)
