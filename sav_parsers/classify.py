from __future__ import annotations

from pathlib import Path
from time import sleep
from urllib.parse import quote
from uuid import uuid4

import google.auth
from google.api_core.client_options import ClientOptions
from google.auth.transport.requests import AuthorizedSession
from google.cloud import documentai_v1beta3 as documentai

from .document_ai import _required_env, mime_type_for, process_document
from .types import DocType


def _to_doc_type(label: str) -> DocType:
  """Map a classifier entity type to a DocType.

  Accepts the classifier's canonical label (`fpb_modelo_1`) or equivalent
  hyphen/underscore variants.
  """
  norm = label.strip().lower()
  if not norm:
    return DocType.OUTROS
  for dt in DocType:
    if dt.value == norm:
      return dt
  hyphenated = norm.replace("_", "-")
  for dt in DocType:
    if dt.value.replace("_", "-") == hyphenated:
      return dt
  return DocType.OUTROS


def classify(pdf_path: str | Path) -> DocType:
  """Run pdf_path through the Document AI classifier processor.

  CDC processors return one entity per candidate class with a confidence
  score; we take the highest-confidence one and map its type to DocType.
  Returns OUTROS if the processor returns no entities or the top label
  doesn't map to a known DocType.
  """
  processor_id = _required_env("DOCAI_CLASSIFIER_PROCESSOR_ID")
  pdf_bytes = Path(pdf_path).read_bytes()
  document = process_document(
    pdf_bytes, processor_id=processor_id, mime_type=mime_type_for(pdf_path),
  )

  if not document.entities:
    return DocType.OUTROS
  top = max(document.entities, key=lambda e: e.confidence)
  return _to_doc_type(top.type_)


def _wait_for_operation(client, operation) -> dict[str, str]:
  raw_operation = operation.operation
  operations_client = client.transport.operations_client
  while not raw_operation.done:
    sleep(1)
    raw_operation = operations_client.get_operation(raw_operation.name)
  if raw_operation.HasField("error"):
    raise RuntimeError(
      f"Document AI import failed [{raw_operation.error.code}]: "
      f"{raw_operation.error.message}",
    )
  return {"operation": raw_operation.name}


def train_classifier(pdf_path: str | Path, expected_doc_type: DocType):
  """Submit a PDF and its type to the processor as training data."""
  # 1. Resolve the classifier processor and the dataset upload prefix.
  project_id = _required_env("DOCAI_PROJECT_ID")
  location = _required_env("DOCAI_LOCATION")
  processor_id = _required_env("DOCAI_CLASSIFIER_PROCESSOR_ID")
  dataset_gcs_prefix = "gs://sav-parsers/tmp/"
  pdf_path = Path(pdf_path)
  pdf_bytes = pdf_path.read_bytes()

  # 2. Document AI imports training docs from GCS, so upload the PDF first.
  bucket_and_prefix = dataset_gcs_prefix.removeprefix("gs://")
  bucket, _, prefix = bucket_and_prefix.partition("/")
  object_name = f"{prefix.rstrip('/')}/{uuid4().hex}.pdf" if prefix else f"{uuid4().hex}.pdf"

  # Use ADC so the same service account can write the object and call DocAI.
  credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
  session = AuthorizedSession(credentials)
  response = session.post(
    f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={quote(object_name, safe='/')}",
    data=pdf_bytes,
    headers={"Content-Type": "application/pdf"},
  )
  response.raise_for_status()

  client = documentai.DocumentServiceClient(
    client_options=ClientOptions(
      api_endpoint=f"{location}-documentai.googleapis.com",
    ),
  )
  dataset = client.dataset_path(project=project_id, location=location, processor=processor_id)

  # 3. Import that uploaded PDF into the classifier dataset's training split.
  operation = client.import_documents(
    request=documentai.ImportDocumentsRequest(
      dataset=dataset,
      batch_documents_import_configs=[
        documentai.ImportDocumentsRequest.BatchDocumentsImportConfig(
          dataset_split=documentai.DatasetSplitType.DATASET_SPLIT_TRAIN,
          batch_input_config=documentai.BatchDocumentsInputConfig(
            gcs_documents=documentai.GcsDocuments(
              documents=[
                documentai.GcsDocument(
                  gcs_uri=f"gs://{bucket}/{object_name}",
                  mime_type="application/pdf",
                ),
              ],
            ),
          ),
          # Classifier imports can label the whole document via document_type,
          # so we don't need to construct entity spans ourselves.
          document_type=expected_doc_type.value,
        ),
      ],
    ),
  )

  # 4. Wait for the long-running import, then delete the temp GCS object.
  import_error = None
  try:
    return _wait_for_operation(client, operation)
  except Exception as exc:
    import_error = exc
    raise
  finally:
    delete_response = session.delete(
      f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(object_name, safe='')}",
    )
    if import_error is None:
      delete_response.raise_for_status()
