# AGENTS.md

Editing rules + design rationale for sav-parsers. Loaded automatically by
Claude Code, Cursor, Cline, Aider, etc. Read [README.md](README.md) first
for the user-facing flow.

## Style

- 2-space indentation. Match existing alignment patterns in dicts/columns.
- Default to writing no comments. Only add a comment when the *why* is
  non-obvious: a hidden constraint, a workaround, surprising behavior. Don't
  narrate *what* the code does — well-named identifiers handle that.
- No emoji in code, docs, or commit messages.
- Don't add error handling, fallbacks, or validation for scenarios that
  can't happen. Only validate at system boundaries (user input, external
  APIs). Trust internal code.
- Don't add backwards-compatibility shims. If something is unused, delete it.
- Don't introduce abstractions beyond what the task requires. Three similar
  lines is better than a premature helper.

## Module map

```
sav_parsers/
  classify.py      # classifier inference + classifier-training import helper
  document_ai.py   # GCP DocAI client + processor lookup helpers
  parsers.py       # classify_and_parse dispatcher + DocType -> parser routing
  fpb_mod1.py      # parse_fpb_mod1 + post-processing for mod 1 forms
  fpb_mod4.py      # placeholder parser for mod 4
  em.py            # placeholder parser for exame medico
  processing.py    # start/close/list/gc lifecycle (doc-type agnostic)
  schema.py        # fetch/load/save schema cache
  types.py         # DocType StrEnum + ParsedField dataclass
  schemas/         # committed schema snapshots (entity lists per doc-type)
cli.py             # argparse CLI wrapping the package
sav-parsers        # bash wrapper that calls .venv/bin/python cli.py
```

## Caller boundary — do not cross

sav-parsers returns `ParsedField`s keyed by Document AI entity name and
nothing else. Caller-specific translation (mapping entities into a
domain object, reconciling OCR against a stored profile, deciding what
to surface for human review) belongs in the caller, not here. Don't grow
domain logic into this package.

The `*_presente` fields are a geometry exception: their bboxes identify the
corrected writable slot, while value entities keep the raw ink location. The
anchor-to-slot calibration lives here because it corrects our own labelling
choices, not caller domain translation.

## Common edits

### Add a schema entity

1. Add the entity in the GCP Document AI console (Schema tab).
2. `./sav-parsers schema <doc-type> --save` — refreshes `sav_parsers/schemas/<doc-type>.json`.
   (Without `--save`, it just prints the live schema to stdout.) Commit the
   diff so it's visible in code review.
3. If the entity needs special post-processing (date, 9-digit, email, postal,
   word-boundary recovery), route it in `_postprocess` in `fpb_mod1.py`.
4. If the entity is a `_region` or `_anchor` locator, register it in that
   doc-type's `_REGION_PAIRS`; locator entities are no longer paired by suffix.

### Add a doc-type (e.g. `fpb_modelo_4`)

The canonical doc-type string is `DocType.<NAME>.value` (e.g. `fpb_modelo_4`).
That string drives the env-var name, the schema filename, the dataset directory,
and the `staged` CLI argument — keep them aligned.

1. Add `DOCAI_FPB_MODELO_4_PROCESSOR_ID=...` to `.env`.
2. Add `FPB_MOD4 = "fpb_modelo_4"` to `DocType` in `types.py` (already there).
3. Create `sav_parsers/fpb_mod4.py` mirroring `fpb_mod1.py`: a
   `parse_fpb_mod4(pdf_path)` that processes via DocAI, applies
   doc-type-specific post-processing, calls `start_processing(...,
   doc_type=DocType.FPB_MOD4.value, auto_corrections=...)`, and returns
   `{"processing_id", "fields"}`.
4. Export `parse_fpb_mod4` from `sav_parsers/__init__.py`.
5. `./sav-parsers schema fpb_modelo_4 --save` to seed the entity cache.
6. Update `sav_parsers/parsers.py` if the new doc-type should be parseable
   via `classify_and_parse`.

### Add classifier training data

Use `./sav-parsers classify --type <doc-type> <pdf>`. This uploads the PDF
to a temporary GCS staging location, imports it into the classifier dataset's
training split with `document_type=<doc-type>`, then deletes the temp object
after the import operation completes.

### Add a post-processor

`_postprocess(entity_type, value)` in `fpb_mod1.py`. Contract: take the raw
OCR text (or DocAI-normalized text) and return a cleaned string, or `None`
if cleanup leaves nothing useful. Booleans/ints short-circuit at the top
and pass through unchanged.

If the cleanup produces a *substring* of the original entity span (leading
char strip, trailing char strip, suffix trim), `_apply_postprocess_to_doc`
will also rewrite the labeled doc's `textAnchor` so retraining inherits the
cleaner label. Cleanups that produce text not present in the original span
(whitespace collapse, hyphen insertion, date reformatting) only update the
displayed value — the labeled doc keeps the raw OCR characters.

## Don't

- Don't change the env-var naming convention. `_processor_id_for(doc_type)`
  derives `DOCAI_<DOC_TYPE>_PROCESSOR_ID` mechanically.
- Don't bypass the `start_processing` → `close_processing` lifecycle. Every
  parse creates a session that must be closed (caller responsibility) or
  garbage-collected later.
- Don't auto-import staged extractor docs to Document AI's training dataset.
  `train_classifier` / `classify --type` is the separate, intentional path
  for classifier training examples only. See "Why we stage locally" below.
- Don't expose `mention_text` on `ParsedField`. Callers want a single
  authoritative `value`; the raw mention is an implementation detail kept
  inside the cached docai.json.

## Design rationale

### Why labels auto-update with a substring-only rule

Document AI labels are spans of `document.text` (textAnchor offsets). For
the labeled doc to point at a corrected value, that value must literally
exist in `document.text`. So:

- Strip leading `[` from morada → cleaned value is a substring of the
  original entity span → `textAnchor` narrowed by 1 char, retraining
  inherits the cleaner boundary.
- Collapse `RIO  MAIOR` → `RIO MAIOR` → cleaned value isn't in the
  original text (the OCR has the double space) → display value updates,
  labeled doc stays untouched.

The substring requirement also prevents a pathological case: the form has
both `localidade` and `concelho` fields that may both contain the same town
name. We require the corrected occurrence to *overlap* the original entity's
span, which keeps each label anchored to its own physical region instead of
silently re-pointing to the neighbor.

### Why we stage locally instead of auto-importing to GCS

`close_processing` writes labeled docs to `files/dataset/<doc-type>/`. It
does NOT call `DocumentService.ImportDocuments` to push them into the
extractor processor's training dataset.

Document AI's `ImportDocuments` requires a GCS prefix; the processor's
managed storage makes the wiring possible. But every closed session would
silently land in the dataset, and some manual review is healthy before
letting unsupervised labels drift the model. `./sav-parsers staged <doc-type>`
lists candidates sorted by correction count so reviewers can pick the
highest-signal ones first; upload at your own cadence.

A `sync` subcommand could automate the upload step later. Skipped for now
per project preference.

Classifier training is different: `train_classifier` intentionally imports a
single caller-labeled PDF immediately via a temporary GCS object, then
deletes that object after the import finishes. That flow is safe here
because the classifier dataset is Google-managed, so the temp GCS location is
only an import staging area, not the dataset's source of truth.

### Only forward user-verified corrections to `close_processing`

`close_processing` accepts a `corrections` dict; whatever you pass in
gets baked into the labeled training doc when the cleaned value anchors
into the OCR text. Callers should pass *only* corrections the user
explicitly typed or confirmed — not values that were auto-accepted
because the model's confidence happened to be high. Auto-accepted values
that turn out to be wrong would teach the retrained model to keep
emitting them. When in doubt, leave them out.

### Why `parse_fpb_mod1` returns a dict, not a dataclass

The return value is meant to cross JSON boundaries (e.g. an MCP server
serializing to an LLM agent). A plain dict with `processing_id` (string)
and `fields` (dict of dataclasses) is more transparent than a wrapper
class. The fields dict itself uses `ParsedField` because a nested
dataclass works fine over JSON via `dataclasses.asdict`.
