# sav-parsers

Parse FPB (Federação Portuguesa de Basquetebol) basketball registration PDFs
via Google Document AI; produce typed, post-processed fields keyed by
Document AI entity name.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

The `./sav-parsers` script at the repo root is a thin wrapper that runs
`.venv/bin/python cli.py "$@"` — no shell activation needed.

## Configure

Create `.env`:

```
DOCAI_PROJECT_ID=<your-gcp-project>
DOCAI_LOCATION=<eu|us|...>
DOCAI_EXAME_MEDICO_PROCESSOR_ID=<processor-id>
DOCAI_FPB_MOD1_PROCESSOR_ID=<processor-id>
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
```

Processor env vars follow the convention `DOCAI_<DOC_TYPE>_PROCESSOR_ID`
(hyphens become underscores, uppercased). Adding a new doc-type only needs a
new env var.

## Mental model — the parse → close lifecycle

A parse run creates a *processing session*. Three phases:

1. **parse** — sends the PDF to Document AI, runs post-processing (strip
   leading `[`, hyphenate postal codes, normalize emails, etc.), persists the
   PDF + cleaned DocAI response under `files/processing/<id>/`. Returns
   `{processing_id, fields}` where `fields` is a dict of `entity_name →
   ParsedField(value, confidence)`.

2. **caller's flow** — the caller does whatever it needs with the fields
   (reconcile against another data source, surface low-confidence values for
   human review, etc.) and may collect user-verified corrections to pass to
   `close_processing`.

3. **close** — finalizes the session. If the labeled doc is materially better
   than raw OCR (parser-side auto-corrections OR caller-supplied corrections
   that anchor onto the original entity span), it's staged to
   `files/dataset/<doc-type>/<id>.{pdf,json,meta.json}` for later manual
   upload to Document AI's training dataset. The processing dir is always
   removed at close.

Caller-supplied corrections whose value can't be located in the OCR text are
recorded as `ocr_limitations` — visible in the close result but not
persisted, since CDE retraining can't fix OCR-layer misreads.

## CLI

```bash
./sav-parsers parse <pdf>                       # process a PDF, print fields
./sav-parsers classify <pdf>                    # returns DocType
./sav-parsers classify --type <doc-type> <pdf>  # import labeled classifier training data
./sav-parsers list                              # pending processing sessions
./sav-parsers close <id> [--correction k=v]...  # finalize a session
./sav-parsers staged <doc-type>                 # labeled docs ready to upload
./sav-parsers gc [--days 7]                     # sweep stale sessions
./sav-parsers schema <doc-type> [--save]        # fetch live schema (--save writes the cache)
```

`--correction` is repeatable; `k` is the entity name (e.g. `morada`), `v` is
the user-confirmed truth.
`classify --type` accepts any `DocType` value such as `fpb_modelo_1`.

## Filesystem

```
files/
  schemas/<doc-type>.json    # cached entity list, committed to git
  processing/<id>/           # active sessions (gitignored)
  dataset/<doc-type>/        # staged labeled docs (gitignored)
```

`schemas/<doc-type>.json` is the source of truth for the entity list at
runtime — `parse_<doc_type>` always returns one `ParsedField` per name in the
schema, padding missing entities with `value=None, confidence=0.0`. Refresh
with `./sav-parsers schema <doc-type> --save` after changing the processor
schema in the GCP console (omit `--save` to inspect without writing).

## Training data

Manual review + upload, for now. Run `./sav-parsers staged fpb-mod1` to list
labeled docs sorted by correction count, pick a candidate, upload to the
processor's GCS bucket (`sav-parsers--fpb-mod1`), and trigger training in the
Document AI console. See [AGENTS.md](AGENTS.md) for why auto-import is
intentionally not wired in yet.

## Acknowledgments

Built with [Claude Code](https://claude.com/claude-code).
