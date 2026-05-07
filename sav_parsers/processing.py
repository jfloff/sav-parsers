from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import documentai
from google.protobuf import json_format

PROCESSING_ROOT = Path("files/processing")
DATASET_ROOT    = Path("files/dataset")


def _processing_dir(processing_id: str) -> Path:
  return PROCESSING_ROOT / processing_id


def start_processing(
  pdf_bytes: bytes, doc_type: str, document: documentai.Document,
  *, auto_corrections: list[str] | None = None,
) -> str:
  """Persist PDF + Document AI response under files/processing/<id>/.

  `auto_corrections` lists entity names whose labels were already improved
  by parser-side post-processing (e.g. leading-char strip on morada). These
  count as "worth training on" alongside any caller-supplied corrections at
  close_processing time, so the labeled doc gets staged even if the caller
  has nothing to add.

  Returns the processing_id, formatted `YYYYMMDDTHHMMSS-<12-hex>` (UTC).
  The timestamp prefix makes ids lexicographically chronological — useful
  for `list_processing` ordering and at-a-glance debugging.
  """
  now = datetime.now(timezone.utc)
  processing_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:12]}"
  d = _processing_dir(processing_id)
  d.mkdir(parents=True, exist_ok=True)

  (d / "original.pdf").write_bytes(pdf_bytes)

  docai_dict = json_format.MessageToDict(document._pb)
  (d / "docai.json").write_text(
    json.dumps(docai_dict, ensure_ascii=False, indent=2),
  )

  meta = {
    "doc_type":         doc_type,
    "created_at":       now.isoformat(),
    "auto_corrections": list(auto_corrections or []),
  }
  (d / "meta.json").write_text(json.dumps(meta, indent=2))

  return processing_id


def _find_entity(doc: dict, entity_type: str) -> dict | None:
  for e in doc.get("entities", []):
    if e.get("type") == entity_type:
      return e
  return None


def _find_text_span(
  haystack: str, needle: str, orig_start: int, orig_end: int,
) -> tuple[int, int] | None:
  """Find `needle` in `haystack`, return only spans that overlap [orig_start, orig_end].

  Tries case-sensitive then case-insensitive. The overlap requirement avoids
  re-labeling the entity onto a different field that happens to contain the
  same text (e.g. the form has both a localidade and a concelho field that
  may name the same town).
  """
  for casefold in (False, True):
    h = haystack.casefold() if casefold else haystack
    n = needle.casefold() if casefold else needle
    candidates: list[int] = []
    start = 0
    while True:
      i = h.find(n, start)
      if i < 0:
        break
      candidates.append(i)
      start = i + 1
    overlapping = [p for p in candidates if p <= orig_end and p + len(needle) >= orig_start]
    if overlapping:
      best = min(overlapping, key=lambda p: abs(p - orig_start))
      return best, best + len(needle)
  return None


def _patch_entity(doc: dict, entity_type: str, corrected: str) -> bool:
  """Rewrite an entity's textAnchor + mentionText to point at `corrected`.

  Returns True if the corrected value was located in document.text. Returns
  False when the value isn't in the OCR text at all (an OCR limitation that
  CDE retraining cannot fix) or when the entity is missing from the response.
  """
  entity = _find_entity(doc, entity_type)
  if entity is None:
    return False

  text = doc.get("text", "")
  segments = entity.get("textAnchor", {}).get("textSegments", [])
  if not segments:
    return False
  orig_start = int(segments[0].get("startIndex", 0))
  orig_end   = int(segments[0].get("endIndex", orig_start))

  span = _find_text_span(text, corrected, orig_start, orig_end)
  if span is None:
    return False

  start, end = span
  entity["textAnchor"] = {
    "textSegments": [{"startIndex": str(start), "endIndex": str(end)}],
  }
  entity["mentionText"] = text[start:end]
  entity.pop("normalizedValue", None)
  return True


def close_processing(
  processing_id: str,
  corrections: dict[str, str] | None = None,
) -> dict:
  """Finalize a processing session and (optionally) stage a labeled training doc.

  `corrections` is keyed by entity name (e.g. "morada", "validade_doc") with the
  user-confirmed truth. For each correction:
    - found inside the original entity span → that entity's textAnchor +
      mentionText are rewritten to point at the corrected substring (a
      "patched" entity).
    - not found → recorded as an OCR limitation; CDE retraining cannot fix
      OCR misreads, so it's surfaced for visibility but doesn't anchor a label.

  Staging happens whenever the labeled doc is materially better than the raw
  model output — i.e. there's at least one patched entity OR at least one
  parser-side auto-correction (recorded at start_processing time, e.g. the
  leading-'[' strip on morada). When that's the case, files/dataset/<doc_type>/
  receives <id>.{pdf,json,meta.json}; otherwise nothing is persisted and the
  processing dir is just removed. The return value always reports both the
  staged signals and any ocr_limitations the caller passed in.

  Note: staging is local. Once a GCS dataset bucket is wired in, this is the
  hook to call DocumentService.ImportDocuments. The actual training run
  (TrainProcessorVersion) stays a manual console action.
  """
  d = _processing_dir(processing_id)
  if not d.exists():
    raise FileNotFoundError(f"Unknown processing_id: {processing_id!r}")

  meta = json.loads((d / "meta.json").read_text())
  doc_type = meta["doc_type"]
  doc = json.loads((d / "docai.json").read_text())

  corrections = corrections or {}
  patched: list[str] = []
  ocr_limited: dict[str, str] = {}

  for entity_type, corrected in corrections.items():
    if _patch_entity(doc, entity_type, corrected):
      patched.append(entity_type)
    else:
      ocr_limited[entity_type] = corrected

  dataset_dir = DATASET_ROOT / doc_type
  auto_corrections = meta.get("auto_corrections", [])
  staged = bool(patched or auto_corrections)

  # Only persist when there's something trainable. ocr_limitations alone
  # aren't useful — the corrected value isn't in the OCR text, so there's
  # no span to label and CDE retraining can't fix it. The caller still
  # sees them in the return value for logging/UX purposes.
  if staged:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(d / "original.pdf", dataset_dir / f"{processing_id}.pdf")
    (dataset_dir / f"{processing_id}.json").write_text(
      json.dumps(doc, ensure_ascii=False, indent=2),
    )
    sidecar = {
      "doc_type":                 doc_type,
      "staged_at":                datetime.now(timezone.utc).isoformat(),
      "patched_entities":         patched,
      "auto_correction_entities": auto_corrections,
      "ocr_limitations":          ocr_limited,
    }
    (dataset_dir / f"{processing_id}.meta.json").write_text(
      json.dumps(sidecar, ensure_ascii=False, indent=2),
    )

  shutil.rmtree(d)

  return {
    "doc_type":                 doc_type,
    "staged":                   staged,
    "patched_entities":         patched,
    "auto_correction_entities": auto_corrections,
    "ocr_limitations":          ocr_limited,
  }


def list_processing() -> list[dict]:
  """Return all pending processing sessions sorted by id, with age in hours."""
  if not PROCESSING_ROOT.exists():
    return []
  out = []
  now = datetime.now(timezone.utc)
  for d in sorted(PROCESSING_ROOT.iterdir()):
    if not d.is_dir():
      continue
    meta_path = d / "meta.json"
    if not meta_path.exists():
      continue
    meta = json.loads(meta_path.read_text())
    created = datetime.fromisoformat(meta["created_at"])
    out.append({
      "processing_id": d.name,
      "doc_type":      meta["doc_type"],
      "created_at":    meta["created_at"],
      "age_hours":     round((now - created).total_seconds() / 3600, 1),
    })
  return out


def list_staged(doc_type: str) -> list[dict]:
  """Return staged labeled docs under files/dataset/<doc_type>/, sorted by
  total correction count (most-corrected first) so the most-interesting
  candidates for manual review/upload surface at the top.
  """
  dataset_dir = DATASET_ROOT / doc_type
  if not dataset_dir.exists():
    return []
  out: list[dict] = []
  for meta_path in dataset_dir.glob("*.meta.json"):
    sidecar = json.loads(meta_path.read_text())
    processing_id = meta_path.name.removesuffix(".meta.json")
    patched = sidecar.get("patched_entities", [])
    auto    = sidecar.get("auto_correction_entities", [])
    limits  = sidecar.get("ocr_limitations", {})
    out.append({
      "id":              processing_id,
      "staged_at":       sidecar.get("staged_at"),
      "corrections":     len(patched) + len(auto),
      "patched":         patched,
      "auto":            auto,
      "ocr_limitations": list(limits.keys()),
    })
  out.sort(key=lambda r: (-r["corrections"], r.get("staged_at") or ""))
  return out


def gc_processing(older_than_days: int = 7) -> list[str]:
  """Delete processing dirs older than `older_than_days`. Returns deleted ids."""
  deleted: list[str] = []
  cutoff_ts = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
  for entry in list_processing():
    created_ts = datetime.fromisoformat(entry["created_at"]).timestamp()
    if created_ts < cutoff_ts:
      shutil.rmtree(_processing_dir(entry["processing_id"]))
      deleted.append(entry["processing_id"])
  return deleted
