from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

from .types import BBox, ParsedField

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

# Locator entities (`<base>_region`, `<base>_anchor`) carry only a bbox; their
# value is folded onto the `<base>_presente` sibling by pair_region_bboxes.
REGION_SUFFIXES = ("_region", "_anchor")
PRESENCE_SUFFIX = "_presente"


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


def entity_bbox(entity, document) -> BBox | None:
  refs = entity.page_anchor.page_refs
  if not refs:
    return None
  # Form fields are single-page; if a multi-page entity ever appears, only
  # the first page's box is kept. Return list[BBox] here if that changes.
  ref = refs[0]
  page_idx = int(ref.page)
  poly = ref.bounding_poly
  if poly.normalized_vertices:
    vertices = [(v.x, v.y) for v in poly.normalized_vertices]
  elif poly.vertices and page_idx < len(document.pages):
    dim = document.pages[page_idx].dimension
    if not dim.width or not dim.height:
      return None
    vertices = [(v.x / dim.width, v.y / dim.height) for v in poly.vertices]
  else:
    return None
  return BBox(page=page_idx, vertices=vertices)


def extract_value(
  entity,
  postprocess: Callable[[str, object], object],
  *,
  presence_types: frozenset[str] = frozenset(),
):
  """Pull a value out of a DocAI entity, applying parser-specific rules.

  Entities in `presence_types` go through `presence_value` (true/false from
  marker text or a signature). Everything else falls through DocAI's
  normalized structured_value when present, otherwise the raw OCR mention is
  cleaned via `postprocess`.
  """
  if entity.type_ in presence_types:
    return presence_value(entity, postprocess)
  nv = entity.normalized_value
  which = nv._pb.WhichOneof("structured_value")
  if which == "boolean_value":
    return nv.boolean_value
  if which == "signature_value":
    return nv.signature_value
  if which == "integer_value":
    return nv.integer_value
  if which == "date_value":
    return nv.text
  raw = nv.text or entity.mention_text
  return postprocess(entity.type_, (raw or "").strip())


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
  # Any unrecognized non-empty text counts as present — the field had ink in it.
  return True


def apply_postprocess_to_doc(document, postprocess: Callable[[str, object], object]) -> list[str]:
  """Apply `postprocess` to OCR mentions and narrow labels when possible.

  Only substring-preserving cleanups can update the underlying label in the
  cached Document AI response. Cleanups that change the text itself
  (whitespace collapse, hyphen insertion, date reformatting) remain
  display-only.

  `postprocess` must be idempotent: this mutates `mention_text` and clears
  `normalized_value`, and downstream value extraction re-applies the same
  callback to the cleaned text.
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


def pair_region_bboxes(fields: dict[str, ParsedField]) -> None:
  """Move each locator bbox onto its `<base>{PRESENCE_SUFFIX}` sibling.

  DocAI's derived presence entities (e.g. `carimbo_clube_presente`) never
  carry a page anchor, so the paired locator entity — `<base>_region` or
  `<base>_anchor` (see REGION_SUFFIXES) — exists only to capture the area
  DocAI inspected. After this step, the presence field carries the locator's
  bbox and the locator key is dropped from `fields`.
  """
  for region_key in [k for k in fields if k.endswith(REGION_SUFFIXES)]:
    suffix = next(s for s in REGION_SUFFIXES if region_key.endswith(s))
    base = region_key[: -len(suffix)]
    region_field = fields.pop(region_key)
    presence_key = base + PRESENCE_SUFFIX
    pf = fields.get(presence_key)
    if pf is None:
      # DocAI inspected the region but emitted no presence entity: create
      # the presence field as False (uninked) carrying the region's bbox.
      fields[presence_key] = ParsedField(
        value=False, confidence=region_field.confidence, bbox=region_field.bbox,
      )
    else:
      # If DocAI has a region for this slot but no presence signal, treat
      # the slot as uninked (False) rather than unknown (None).
      if pf.value is None:
        pf.value = False
      pf.bbox = region_field.bbox
