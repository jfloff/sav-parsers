"""Schema cache: snapshot of a Document AI processor's entity list, on disk.

Run `./sav-parsers schema <doc-type>` to refresh the cache from the deployed
processor version. parse_<doc_type> uses the cache at runtime to ensure every
schema entity is present in the output (missing ones get value=None,
confidence=0.0). Keeping the snapshot under version control makes schema
changes visible in code review.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1beta3

from .document_ai import _processor_id_for, _required_env

SCHEMA_DIR = Path("files/schemas")


def schema_path(doc_type: str) -> Path:
  return SCHEMA_DIR / f"{doc_type}.json"


def fetch_schema(doc_type: str) -> list[str]:
  """Hit Document AI for the processor's dataset entity list.

  We use DatasetServiceClient.get_dataset_schema (v1beta3) because Custom
  Extractor processors expose their trained labels there. The v1
  get_processor_version().document_schema only returns the base entity
  ('custom_extraction_document_type') for CDE — useless for our purposes.
  """
  project_id   = _required_env("DOCAI_PROJECT_ID")
  location     = _required_env("DOCAI_LOCATION")
  processor_id = _processor_id_for(doc_type)

  client = documentai_v1beta3.DocumentServiceClient(
    client_options=ClientOptions(
      api_endpoint=f"{location}-documentai.googleapis.com",
    ),
  )
  schema = client.get_dataset_schema(
    name=f"projects/{project_id}/locations/{location}/processors/{processor_id}/dataset/datasetSchema",
  )
  # Trained labels live as `properties` of the synthetic
  # `custom_extraction_document_type` entity, not as top-level entity_types.
  return sorted(
    prop.name
    for et in schema.document_schema.entity_types
    for prop in et.properties
  )


def save_schema(doc_type: str, entities: list[str]) -> Path:
  SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
  payload = {
    "doc_type":   doc_type,
    "fetched_at": datetime.now(timezone.utc).isoformat(),
    "entities":   sorted(entities),
  }
  path = schema_path(doc_type)
  path.write_text(json.dumps(payload, indent=2))
  return path


def load_schema(doc_type: str) -> tuple[str, ...]:
  """Read the cached schema. Returns () if no cache file exists yet."""
  path = schema_path(doc_type)
  if not path.exists():
    return ()
  payload = json.loads(path.read_text())
  return tuple(payload.get("entities", []))
