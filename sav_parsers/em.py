from __future__ import annotations

from pathlib import Path

from .document_ai import _processor_id_for, process_document
from .postprocess import (
  apply_postprocess_to_doc,
  clean_ocr_text,
  presence_value,
  try_iso_date,
)
from .processing import start_processing
from .schema import load_schema
from .types import DocType, ParsedField


def _postprocess(entity_type: str, value):
  if not isinstance(value, str):
    return value

  cleaned = clean_ocr_text(value)
  if not cleaned:
    return None

  if entity_type == "exam_date":
    return try_iso_date(cleaned) or cleaned

  return cleaned


def parse_em(pdf_path: str | Path) -> dict:
  processor_id = _processor_id_for(DocType.EM.value)
  pdf_bytes = Path(pdf_path).read_bytes()
  document = process_document(pdf_bytes, processor_id=processor_id)

  auto_corrections = apply_postprocess_to_doc(document, _postprocess)
  processing_id = start_processing(
    pdf_bytes,
    DocType.EM.value,
    document,
    auto_corrections=auto_corrections,
  )

  def _value(entity):
    if entity.type_ == "doctor_validation_present":
      return presence_value(entity, _postprocess)

    nv = entity.normalized_value
    which = nv._pb.WhichOneof("structured_value")
    if which == "boolean_value":
      return nv.boolean_value
    if which == "integer_value":
      return nv.integer_value
    if which == "date_value":
      return nv.text
    raw = nv.text or entity.mention_text
    return _postprocess(entity.type_, (raw or "").strip())

  fields = {
    e.type_: ParsedField(value=_value(e), confidence=e.confidence)
    for e in document.entities
  }
  for entity_type in load_schema(DocType.EM.value):
    fields.setdefault(entity_type, ParsedField(value=None, confidence=0.0))
  return {"processing_id": processing_id, "fields": fields}
