#!/usr/bin/env python
"""sav-parsers CLI — manual exercise + ops for the parsing pipeline.

Usage:
  ./sav-parsers parse <pdf>
  ./sav-parsers classify [--type doc-type] <pdf>
  ./sav-parsers close <processing_id> [--correction entity=value]...
  ./sav-parsers list
  ./sav-parsers staged <doc-type>
  ./sav-parsers gc [--days N]
  ./sav-parsers schema <doc-type>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from sav_parsers import (
  DocType,
  classify,
  close_processing,
  gc_processing,
  list_processing,
  list_staged,
  train_classifier,
)
from sav_parsers.parsers import classify_and_parse
from sav_parsers.schema import fetch_schema, load_schema, save_schema


def _print_json(obj) -> None:
  print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_parse(args) -> int:
  doc_type, result = classify_and_parse(args.pdf)
  if result is None:
    _print_json({
      "doc_type": doc_type,
      "error":    "unsupported_doc_type",
      "message":  f"no parser available for doc_type {doc_type.value!r}",
    })
    return 1

  _print_json({
    "doc_type": doc_type,
    "processing_id": result["processing_id"],
    "fields":        {k: asdict(v) for k, v in result["fields"].items()},
  })
  return 0


def cmd_classify(args) -> int:
  if args.doc_type:
    train_classifier(args.pdf, DocType(args.doc_type))
    _print_json({
      "trained":  True,
      "doc_type": args.doc_type,
      "pdf":      args.pdf,
    })
    return 0
  print(classify(args.pdf))
  return 0


def cmd_close(args) -> int:
  corrections: dict[str, str] = {}
  for item in args.correction or []:
    if "=" not in item:
      _print_json({
        "error":   "bad_correction_format",
        "message": f"--correction must be entity=value, got {item!r}",
      })
      return 2
    k, v = item.split("=", 1)
    corrections[k.strip()] = v.strip()
  try:
    result = close_processing(args.processing_id, corrections=corrections or None)
  except FileNotFoundError:
    _print_json({
      "error":         "unknown_processing_id",
      "message":       f"no pending processing session {args.processing_id!r}",
      "processing_id": args.processing_id,
    })
    return 1
  _print_json(result)
  return 0


def cmd_list(_args) -> int:
  _print_json(list_processing())
  return 0


def cmd_staged(args) -> int:
  _print_json(list_staged(args.doc_type))
  return 0


def cmd_gc(args) -> int:
  deleted = gc_processing(older_than_days=args.days)
  _print_json({"deleted": deleted, "older_than_days": args.days})
  return 0


def cmd_schema(args) -> int:
  fetched = fetch_schema(args.doc_type)
  new = sorted(fetched["entities"])
  if not args.save:
    _print_json({
      "doc_type": args.doc_type,
      "processor_type": fetched["processor_type"],
      "entity_count": len(new),
      "entities": new,
    })
    return 0
  existing = set(load_schema(args.doc_type))
  path = save_schema(
    args.doc_type,
    new,
    processor_type=fetched["processor_type"],
  )
  _print_json({
    "doc_type": args.doc_type,
    "path": str(path),
    "processor_type": fetched["processor_type"],
    "entity_count": len(new),
    "added": sorted(set(new) - existing),
    "removed": sorted(existing - set(new)),
  })
  return 0


def main() -> int:
  p = argparse.ArgumentParser(prog="sav-parsers")
  sub = p.add_subparsers(dest="cmd", required=True)

  pp = sub.add_parser("parse", help="Run the FPB mod1 parser on a PDF")
  pp.add_argument("pdf")
  pp.set_defaults(func=cmd_parse)

  pc = sub.add_parser("classify", help="Classify a PDF or submit labeled training data")
  pc.add_argument(
    "--type",
    dest="doc_type",
    choices=[dt.value for dt in DocType],
    help="If set, treat the PDF as labeled training data for this doc type.",
  )
  pc.add_argument("pdf")
  pc.set_defaults(func=cmd_classify)

  pcl = sub.add_parser("close", help="Close a processing session, optionally with corrections")
  pcl.add_argument("processing_id")
  pcl.add_argument(
    "--correction", action="append",
    metavar="entity=value",
    help="Repeatable. Entity name (e.g. morada) and the corrected value.",
  )
  pcl.set_defaults(func=cmd_close)

  pl = sub.add_parser("list", help="List pending processing sessions")
  pl.set_defaults(func=cmd_list)

  pst = sub.add_parser(
    "staged",
    help="List staged labeled docs under files/dataset/<doc-type>/ "
         "(sorted by correction count) — pick one to upload manually.",
  )
  pst.add_argument("doc_type", help="e.g. fpb_modelo_1")
  pst.set_defaults(func=cmd_staged)

  pg = sub.add_parser("gc", help="Delete stale processing sessions")
  pg.add_argument("--days", type=int, default=7)
  pg.set_defaults(func=cmd_gc)

  ps = sub.add_parser(
    "schema",
    help="Fetch processor schema from Document AI; prints to stdout. "
         "Use --save to also write files/schemas/<doc-type>.json.",
  )
  ps.add_argument("doc_type", help="e.g. fpb_modelo_1")
  ps.add_argument("--save", action="store_true", help="Write to files/schemas/<doc-type>.json (with diff vs cached)")
  ps.set_defaults(func=cmd_schema)

  args = p.parse_args()
  return args.func(args)


if __name__ == "__main__":
  sys.exit(main())
