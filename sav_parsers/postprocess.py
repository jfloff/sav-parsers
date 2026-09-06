from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
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
      # proto-plus wrappers expose no Clear() — that is the raw protobuf API.
      # `del` is the idiomatic clear here (equivalent to _pb.ClearField).
      del entity.normalized_value
    changed.append(entity.type_)
  return changed


@dataclass(frozen=True)
class SlotOffset:
  """Where the writable slot sits relative to the labelled anchor.

  Units are anchor widths/heights; `right`/`up` move the centre, `width`/
  `height` scale about it. All-defaults is the identity.
  """
  right: float = 0.0
  up: float = 0.0
  width: float = 1.0
  height: float = 1.0


def slot_from_anchor(
  vertices: list[tuple[float, float]], offset: SlotOffset,
) -> list[tuple[float, float]]:
  min_x = min(x for x, _ in vertices)
  max_x = max(x for x, _ in vertices)
  min_y = min(y for _, y in vertices)
  max_y = max(y for _, y in vertices)
  aw = max_x - min_x
  ah = max_y - min_y
  cx = (min_x + max_x) / 2 + offset.right * aw
  cy = (min_y + max_y) / 2 - offset.up * ah
  half_w = aw * offset.width / 2
  half_h = ah * offset.height / 2
  return [
    (cx - half_w, cy - half_h),
    (cx + half_w, cy - half_h),
    (cx + half_w, cy + half_h),
    (cx - half_w, cy + half_h),
  ]


@dataclass(frozen=True)
class RegionPair:
  """One locator entity and the presence field its bbox belongs to."""
  region: str
  presence: str
  value: str | None = None
  slot: SlotOffset = SlotOffset()


def pair_region_bboxes(
  fields: dict[str, ParsedField], pairs: tuple[RegionPair, ...],
) -> None:
  """Pair explicit locator entities with presence fields.

  Each locator is removed from `fields`; its bbox is transplanted to the
  presence field after correcting it to the writable slot rather than keeping
  the labelled anchor. A real presence value is preserved, while a missing or
  unknown presence is derived from the configured value entity or defaults to
  false. Missing locator entities leave their presence fields untouched.
  """
  for pair in pairs:
    region_field = fields.pop(pair.region, None)
    if region_field is None:
      continue
    slot_bbox = None
    if region_field.bbox is not None:
      slot_bbox = BBox(
        page=region_field.bbox.page,
        vertices=slot_from_anchor(region_field.bbox.vertices, pair.slot),
      )
    # The value entity is missing entirely when DocAI read nothing there and
    # the doc-type's schema cache does not list it.
    value_field = fields.get(pair.value) if pair.value else None
    inked = value_field is not None and value_field.value is not None
    presence_field = fields.get(pair.presence)
    if presence_field is None:
      fields[pair.presence] = ParsedField(
        value=inked, confidence=region_field.confidence, bbox=slot_bbox,
      )
      continue
    if presence_field.value is None:
      presence_field.value = inked
    presence_field.bbox = slot_bbox
