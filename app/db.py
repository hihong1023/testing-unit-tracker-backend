# app/db.py
import os
from pathlib import Path
from typing import Optional

from sqlmodel import SQLModel, create_engine, Session

from azure.storage.blob import BlobServiceClient, BlobClient


# ========= CONFIG & PATHS =========

# Where the SQLite DB file lives inside the app/container
# (Keep default same as your current file to avoid breaking anything)
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./testing_unit_tracker.db")

# Azure Storage env vars (set in Azure portal / local .env)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "db-backup")
AZURE_STORAGE_BLOB_NAME = os.getenv("AZURE_STORAGE_BLOB_NAME", "testing_unit_tracker.db")

db_path = Path(SQLITE_DB_PATH)
db_path.parent.mkdir(parents=True, exist_ok=True)


# ========= ENGINE =========

DATABASE_URL = f"sqlite:///{db_path.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},  # needed for sqlite+fastapi
)


# ========= BLOB HELPERS =========

def _get_blob_client() -> Optional[BlobClient]:
    """
    Return a BlobClient or None if storage is not configured.
    """
    if not AZURE_STORAGE_CONNECTION_STRING:
        # Likely local dev or not configured yet
        print("[DB] AZURE_STORAGE_CONNECTION_STRING not set, skipping blob sync.")
        return None

    service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    container_client = service_client.get_container_client(AZURE_STORAGE_CONTAINER)

    # Ensure container exists (idempotent)
    try:
        container_client.create_container()
    except Exception:
        # already exists, ignore
        pass

    blob_client = container_client.get_blob_client(AZURE_STORAGE_BLOB_NAME)
    return blob_client


def download_db_from_blob_if_needed() -> None:
    """
    On startup:
    - If local DB does NOT exist but there is a blob → download it.
    - If no blob yet → do nothing; a new DB will be created locally.
    """
    blob_client = _get_blob_client()
    if blob_client is None:
        return

    if db_path.exists():
        print(f"[DB] Local DB exists at {db_path}, no need to download.")
        return

    print("[DB] Local DB not found, checking blob storage...")
    try:
        if blob_client.exists():
            print("[DB] Remote DB blob found, downloading...")
            with open(db_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            print(f"[DB] Downloaded DB to {db_path}")
        else:
            print("[DB] No DB blob found, will create new SQLite file locally.")
    except Exception as e:
        print(f"[DB] Error downloading DB from blob: {e}")


def upload_db_to_blob() -> None:
    """
    On shutdown:
    - Upload the local DB to blob (overwrite=True).
    """
    blob_client = _get_blob_client()
    if blob_client is None:
        return

    if not db_path.exists():
        print(f"[DB] Local DB file {db_path} does not exist, nothing to upload.")
        return

    try:
        print("[DB] Uploading DB file to blob storage...")
        with open(db_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print("[DB] DB upload complete.")
    except Exception as e:
        print(f"[DB] Error uploading DB to blob: {e}")


# ========= DB INIT & SESSION =========

def init_db():
    """
    Create tables if they do not exist.
    Call this AFTER download_db_from_blob_if_needed() on startup.
    """
    # If your models are in app/models.py and imported elsewhere already,
    # you may not need this import. If tables are not created, uncomment:
    # from . import models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
