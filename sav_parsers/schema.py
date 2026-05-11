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
from google.cloud import documentai_v1beta3 as documentai

from .document_ai import _processor_id_for, _required_env

SCHEMA_DIR = Path("files/schemas")
_CLASSIFIER_PROCESSOR_TYPE = "CUSTOM_CLASSIFICATION_PROCESSOR"


def schema_path(doc_type: str) -> Path:
  return SCHEMA_DIR / f"{doc_type}.json"


def _schema_processor_id_for(doc_type: str) -> str:
  if doc_type == "classifier":
    return _required_env("DOCAI_CLASSIFIER_PROCESSOR_ID")
  return _processor_id_for(doc_type)


def _processor_type(processor_id: str) -> str:
  project_id = _required_env("DOCAI_PROJECT_ID")
  location = _required_env("DOCAI_LOCATION")
  client = documentai.DocumentProcessorServiceClient(
    client_options=ClientOptions(
      api_endpoint=f"{location}-documentai.googleapis.com",
    ),
  )
  processor = client.get_processor(
    name=f"projects/{project_id}/locations/{location}/processors/{processor_id}",
  )
  return processor.type_


def fetch_schema(doc_type: str) -> dict[str, str | list[str]]:
  """Hit Document AI for the processor's dataset entity list.

  We use v1beta3 get_dataset_schema because Custom Extractor processors
  expose their trained labels there. get_processor_version().document_schema
  only returns the base entity ('custom_extraction_document_type') for CDE —
  useless for our purposes.
  """
  project_id = _required_env("DOCAI_PROJECT_ID")
  location = _required_env("DOCAI_LOCATION")
  processor_id = _schema_processor_id_for(doc_type)
  processor_type = _processor_type(processor_id)

  client = documentai.DocumentServiceClient(
    client_options=ClientOptions(
      api_endpoint=f"{location}-documentai.googleapis.com",
    ),
  )
  schema = client.get_dataset_schema(
    name=f"projects/{project_id}/locations/{location}/processors/{processor_id}/dataset/datasetSchema",
  )
  if processor_type == _CLASSIFIER_PROCESSOR_TYPE:
    entities = sorted(et.name for et in schema.document_schema.entity_types)
  else:
    # Trained labels live as `properties` of the synthetic
    # `custom_extraction_document_type` entity, not as top-level entity_types.
    entities = sorted(
      prop.name
      for et in schema.document_schema.entity_types
      for prop in et.properties
    )
  return {
    "processor_type": processor_type,
    "entities": entities,
  }


def save_schema(
  doc_type: str,
  entities: list[str],
  *,
  processor_type: str | None = None,
) -> Path:
  SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
  payload = {
    "doc_type": doc_type,
    "fetched_at": datetime.now(timezone.utc).isoformat(),
    "entities": sorted(entities),
  }
  if processor_type is not None:
    payload["processor_type"] = processor_type
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
