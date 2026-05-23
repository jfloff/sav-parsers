from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

load_dotenv()

# Mime types Document AI accepts for synchronous processing, keyed by extension.
_MIME_TYPES = {
  ".pdf": "application/pdf",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".tif": "image/tiff",
  ".tiff": "image/tiff",
  ".gif": "image/gif",
  ".bmp": "image/bmp",
  ".webp": "image/webp",
}


def mime_type_for(path: str | Path) -> str:
  """Resolve a Document AI mime type from a file extension."""
  ext = Path(path).suffix.lower()
  try:
    return _MIME_TYPES[ext]
  except KeyError:
    raise ValueError(
      f"Unsupported file type {ext or path!r}; "
      f"expected one of {', '.join(sorted(_MIME_TYPES))}",
    ) from None


def _required_env(name: str) -> str:
  value = os.getenv(name)
  if not value:
    raise RuntimeError(f"Missing required env var: {name}")
  return value


@lru_cache(maxsize=1)
def _client() -> documentai.DocumentProcessorServiceClient:
  location = _required_env("DOCAI_LOCATION")
  return documentai.DocumentProcessorServiceClient(
    client_options=ClientOptions(
      api_endpoint=f"{location}-documentai.googleapis.com",
    ),
  )


def process_document(
  content: bytes,
  processor_id: str,
  mime_type: str = "application/pdf",
) -> documentai.Document:
  """Send document bytes to a Document AI processor; return the parsed Document."""
  project_id = _required_env("DOCAI_PROJECT_ID")
  location = _required_env("DOCAI_LOCATION")
  name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
  request = documentai.ProcessRequest(
    name=name,
    raw_document=documentai.RawDocument(
      content=content,
      mime_type=mime_type,
    ),
  )
  return _client().process_document(request=request).document


def _processor_id_for(doc_type: str) -> str:
  env_var = "DOCAI_" + doc_type.upper().replace("-", "_") + "_PROCESSOR_ID"
  return _required_env(env_var)
