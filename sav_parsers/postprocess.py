from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

# YYYYMMDD stays before DDMMYYYY because most date-as-int OCR cases come
# from machine-printed YYYYMMDD.
_DATE_PATTERNS: list[tuple[str, tuple[int, int, int]]] = [
  (r"^(\d{4})[/.\-](\d{2})[/.\-](\d{2})$", (1, 2, 3)),
  (r"^(\d{2})[/.\-](\d{2})[/.\-](\d{4})$", (3, 2, 1)),
  (r"^(\d{4})(\d{2})(\d{2})$",             (1, 2, 3)),
  (r"^(\d{2})(\d{2})(\d{4})$",             (3, 2, 1)),
]

TRUE_MARKERS = {
  "1", "s", "sim", "true", "v", "x", "y", "yes",
}
FALSE_MARKERS = {
  "0", "n", "nao", "não", "false", "no",
}


def try_iso_date(value: str) -> str | None:
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


def clean_ocr_text(value: str) -> str | None:
  cleaned = re.sub(r"^[^A-Za-zÀ-ÿ0-9]+", "", value)
  cleaned = re.sub(r"\s+", " ", cleaned).strip()
  return cleaned or None


def presence_value(entity, postprocess: Callable[[str, object], object]):
  nv = entity.normalized_value
  which = nv._pb.WhichOneof("structured_value")
  if which == "boolean_value":
    return nv.boolean_value
  if which == "signature_value":
    return True

  raw = postprocess(entity.type_, (nv.text or entity.mention_text or "").strip())
  if raw is None:
    return None
  if isinstance(raw, bool):
    return raw

  lowered = raw.casefold()
  if lowered in TRUE_MARKERS:
    return True
  if lowered in FALSE_MARKERS:
    return False
  return True


def apply_postprocess_to_doc(document, postprocess: Callable[[str, object], object]) -> list[str]:
  """Apply `postprocess` to OCR mentions and narrow labels when possible.

  Only substring-preserving cleanups can update the underlying label in the
  cached Document AI response. Cleanups that change the text itself
  (whitespace collapse, hyphen insertion, date reformatting) remain
  display-only.
  """
  text = document.text
  changed: list[str] = []
  for entity in document.entities:
    original = entity.mention_text or ""
    cleaned = postprocess(entity.type_, original)
    if not isinstance(cleaned, str) or cleaned == original:
      continue
    if not entity.text_anchor.text_segments:
      continue
    seg = entity.text_anchor.text_segments[0]
    orig_start = int(seg.start_index) if seg.start_index else 0
    orig_end = int(seg.end_index)
    orig_text = text[orig_start:orig_end]
    idx = orig_text.find(cleaned)
    if idx < 0:
      continue
    seg.start_index = orig_start + idx
    seg.end_index = seg.start_index + len(cleaned)
    entity.mention_text = cleaned
    if entity.normalized_value:
      entity.normalized_value.Clear()
    changed.append(entity.type_)
  return changed
