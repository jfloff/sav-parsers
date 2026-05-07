from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from wordfreq import zipf_frequency

from .document_ai import _processor_id_for, process_document
from .processing import start_processing
from .schema import load_schema
from .types import ParsedField

_NINE_DIGIT_FIELDS = {"nif", "telemovel", "telefone", "telefone_encarregado"}
_DATE_FIELDS       = {"validade_doc", "validade_doc_encarregado", "data_nascimento"}

# Text fields where handwritten OCR sometimes drops inter-word spaces
# ("NOGUEIRAGUEDES" → "NOGUEIRA GUEDES"). Structured fields (NIF, postal,
# email, dates, doc numbers, policy numbers, season) stay out — splitting
# digit/code blobs by word frequency makes no sense.
_WORD_BOUNDARY_FIELDS = {
  "nome_completo", "nome_encarregado",
  "morada", "localidade", "concelho", "distrito",
  "pais_nascimento", "nacionalidade",
  "clube", "associacao", "companhia_seguro",
}

# Frequency-based mash-splitter (wordfreq Portuguese corpus). Tuned to be
# conservative: only triggers on tokens that are themselves essentially
# absent from the corpus AND admit a decomposition into well-attested
# parts of >= 4 letters each. VITORINO (zipf 2.96, single token) stays
# whole; NOGUEIRAGUEDES (zipf 0.00) splits into NOGUEIRA + GUEDES.
_WB_MIN_PART_LEN  = 4
_WB_MIN_PART_ZIPF = 2.5
_WB_MAX_TOKEN_ZIPF = 1.5
_WB_MIN_TOKEN_LEN = 2 * _WB_MIN_PART_LEN


@lru_cache(maxsize=8192)
def _pt_zipf(token: str) -> float:
  return zipf_frequency(token.lower(), "pt")


@lru_cache(maxsize=8192)
def _decompose_token(token: str) -> tuple[str, ...] | None:
  """Greedy-best decomposition into parts each with zipf >= MIN_PART_ZIPF.

  Returns the parts (>=1) if the token is itself common or splits cleanly,
  else None. Score = min(zipf) across parts, so among valid decompositions
  we prefer the one whose weakest part is strongest.
  """
  if _pt_zipf(token) >= _WB_MIN_PART_ZIPF:
    return (token,)
  if len(token) < 2 * _WB_MIN_PART_LEN:
    return None
  best: tuple[str, ...] | None = None
  best_score = -1.0
  for i in range(_WB_MIN_PART_LEN, len(token) - _WB_MIN_PART_LEN + 1):
    left  = _decompose_token(token[:i])
    right = _decompose_token(token[i:])
    if left is None or right is None:
      continue
    parts = left + right
    score = min(_pt_zipf(p) for p in parts)
    if score > best_score:
      best_score = score
      best = parts
  return best


def _recover_word_boundaries(text: str) -> str:
  out: list[str] = []
  for token in text.split(" "):
    if (not token.isalpha()
        or len(token) < _WB_MIN_TOKEN_LEN
        or _pt_zipf(token) >= _WB_MAX_TOKEN_ZIPF):
      out.append(token)
      continue
    parts = _decompose_token(token)
    if parts is None or len(parts) <= 1:
      out.append(token)
      continue
    out.append(" ".join(parts))
  return " ".join(out)

# (regex, indices of year/month/day groups). YYYYMMDD ordered before DDMMYYYY
# because most date-as-int OCR cases come from machine-printed YYYYMMDD.
_DATE_PATTERNS: list[tuple[str, tuple[int, int, int]]] = [
  (r"^(\d{4})[/.\-](\d{2})[/.\-](\d{2})$", (1, 2, 3)),
  (r"^(\d{2})[/.\-](\d{2})[/.\-](\d{4})$", (3, 2, 1)),
  (r"^(\d{4})(\d{2})(\d{2})$",             (1, 2, 3)),
  (r"^(\d{2})(\d{2})(\d{4})$",             (3, 2, 1)),
]


def _try_date(value: str) -> str | None:
  s = value.strip()
  for pattern, (yi, mi, di) in _DATE_PATTERNS:
    m = re.match(pattern, s)
    if not m:
      continue
    y, mo, d = m.group(yi), m.group(mi), m.group(di)
    try:
      datetime(int(y), int(mo), int(d))
      return f"{y}-{mo}-{d}"
    except ValueError:
      continue
  return None


def _postprocess_email(value: str) -> str | None:
  cleaned = re.sub(r"\s+", "", value).lower()
  if "@" not in cleaned or "." not in cleaned.rsplit("@", 1)[-1]:
    return value or None
  return cleaned


def _postprocess(entity_type: str, value):
  """Light, low-risk cleanups applied to the value emitted by Document AI.

  Booleans, ints, etc. pass through unchanged. Cleanups that produce a
  substring of the original OCR text (e.g. leading-char strip) can also be
  applied to the labeled doc by _apply_postprocess_to_doc — see that hook
  for the label-side update.
  """
  if not isinstance(value, str):
    return value

  if entity_type in {"email_jogador", "email_encarregado"}:
    return _postprocess_email(value)

  # Generic: strip leading non-alphanumeric + collapse internal whitespace
  cleaned = re.sub(r"^[^A-Za-zÀ-ÿ0-9]+", "", value)
  cleaned = re.sub(r"\s+", " ", cleaned).strip()
  if not cleaned:
    return None

  if entity_type == "codigo_postal":
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) == 7:
      return f"{digits[:4]}-{digits[4:]}"
    return cleaned

  if entity_type in _NINE_DIGIT_FIELDS:
    digits = re.sub(r"\D", "", cleaned)
    return digits if len(digits) == 9 else cleaned

  if entity_type in _DATE_FIELDS:
    return _try_date(cleaned) or cleaned

  if entity_type in _WORD_BOUNDARY_FIELDS:
    return _recover_word_boundaries(cleaned)

  return cleaned


def _apply_postprocess_to_doc(document) -> list[str]:
  """Mutate the Document so each entity's textAnchor/mentionText reflect
  what _postprocess produces, when the cleaned value is locatable inside
  the original entity span.

  This means the cached docai.json (and any labeled training doc derived
  from it) already has narrowed labels — no caller-supplied correction
  needed to teach the retrained model to drop the leading '[' on morada
  or similar artefacts. Cleanups whose output is not a substring of the
  original span (whitespace collapse, hyphen insertion, date reformatting)
  are display-only and don't touch the label.

  Returns the list of entity types whose label was actually patched.
  """
  text = document.text
  changed: list[str] = []
  for entity in document.entities:
    original = entity.mention_text or ""
    cleaned = _postprocess(entity.type_, original)
    if not isinstance(cleaned, str) or cleaned == original:
      continue
    if not entity.text_anchor.text_segments:
      continue
    seg = entity.text_anchor.text_segments[0]
    orig_start = int(seg.start_index) if seg.start_index else 0
    orig_end   = int(seg.end_index)
    orig_text  = text[orig_start:orig_end]
    idx = orig_text.find(cleaned)
    if idx < 0:
      continue
    seg.start_index = orig_start + idx
    seg.end_index   = seg.start_index + len(cleaned)
    entity.mention_text = cleaned
    if entity.normalized_value:
      entity.normalized_value.Clear()
    changed.append(entity.type_)
  return changed


def parse_fpb_mod1(pdf_path: str | Path) -> dict:
  """Send `pdf_path` through the FPB mod 1 processor.

  Returns:
    {
      "processing_id": str,                     # pass to close_processing later
      "fields":        dict[str, ParsedField],  # entity type → ParsedField
    }

  Post-processing is applied to both the displayed values and the cached
  labeled doc (where the cleaned text maps cleanly to a substring of the
  OCR text). The processing_id is the handle to a session under
  files/processing/<id>/ holding the original PDF + cleaned DocAI response.
  """
  processor_id = _processor_id_for("fpb-mod1")
  pdf_bytes = Path(pdf_path).read_bytes()
  document = process_document(pdf_bytes, processor_id=processor_id)

  auto_corrections = _apply_postprocess_to_doc(document)
  processing_id = start_processing(
    pdf_bytes, "fpb-mod1", document, auto_corrections=auto_corrections,
  )

  def _value(entity):
    nv = entity.normalized_value
    which = nv._pb.WhichOneof("structured_value")
    if which == "boolean_value":
      return nv.boolean_value
    if which == "signature_value":
      return nv.signature_value
    if which == "integer_value":
      return nv.integer_value
    if which == "date_value":
      return nv.text  # DocAI already gives ISO YYYY-MM-DD here
    # Prefer DocAI's normalized text when present, fall back to OCR mention.
    raw = nv.text or entity.mention_text
    return _postprocess(entity.type_, (raw or "").strip())

  fields = {
    e.type_: ParsedField(value=_value(e), confidence=e.confidence)
    for e in document.entities
  }
  for entity_type in load_schema("fpb-mod1"):
    fields.setdefault(entity_type, ParsedField(value=None, confidence=0.0))
  return {"processing_id": processing_id, "fields": fields}
