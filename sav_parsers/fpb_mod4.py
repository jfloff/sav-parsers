from __future__ import annotations

from pathlib import Path

from .document_ai import _processor_id_for, mime_type_for, process_document
from .postprocess import (
  apply_postprocess_to_doc,
  clean_ocr_text,
  entity_bbox,
  extract_value,
  pair_region_bboxes,
  RegionPair,
)
from .processing import start_processing
from .schema import load_schema
from .types import DocType, ParsedField

# Modelo 4 carries two signature slots: the holder's (assinatura_detentor) and
# the club's (club_signature). Each `_presente` is the inked/uninked signal; the
# sibling `_anchor` is its bbox locator (it never carries a presence value).
# presence_value turns the signal into a true/false, and pair_region_bboxes folds
# the anchor's bbox onto the _presente sibling — exactly the _region/_presente
# pairing used in mod1.
_PRESENCE_FIELDS = frozenset({
  "assinatura_detentor_presente",
  "club_signature_presente",
})

_REGION_PAIRS = (
  RegionPair("assinatura_detentor_anchor", "assinatura_detentor_presente"),
  RegionPair("club_signature_anchor", "club_signature_presente"),
)


def _postprocess(entity_type: str, value):
  """Light OCR cleanup. Modelo 4 carries only name/license/tier text plus the
  signature presence fields — no dates, NIFs, or postal codes to normalize."""
  if not isinstance(value, str):
    return value
  return clean_ocr_text(value)


def parse_fpb_mod4(pdf_path: str | Path) -> dict:
  """Send `pdf_path` through the FPB Modelo 4 processor.

  Returns {"processing_id": str, "fields": dict[str, ParsedField]}, matching
  the other FPB parsers. See parse_fpb_mod1 for the processing-session contract.
  """
  processor_id = _processor_id_for(DocType.FPB_MODELO_4.value)
  pdf_bytes = Path(pdf_path).read_bytes()
  document = process_document(
    pdf_bytes, processor_id=processor_id, mime_type=mime_type_for(pdf_path),
  )

  auto_corrections = apply_postprocess_to_doc(document, _postprocess)
  processing_id = start_processing(
    pdf_bytes, DocType.FPB_MODELO_4.value, document, auto_corrections=auto_corrections,
  )

  fields = {
    e.type_: ParsedField(
      value=extract_value(e, _postprocess, presence_types=_PRESENCE_FIELDS),
      confidence=e.confidence,
      bbox=entity_bbox(e, document),
    )
    for e in document.entities
  }
  for entity_type in load_schema(DocType.FPB_MODELO_4.value):
    # DocAI omits a presence entity when its box is empty (unsigned form) and
    # doesn't always emit the paired anchor either, so default presence fields
    # to False — None would read as "unknown" instead of "not signed".
    default = False if entity_type in _PRESENCE_FIELDS else None
    fields.setdefault(entity_type, ParsedField(value=default, confidence=0.0))
  # Signature anchors are already on the writable ink slots, so the identity
  # pairs fold their bboxes onto the presence fields.
  pair_region_bboxes(fields, _REGION_PAIRS)
  return {"processing_id": processing_id, "fields": fields}
