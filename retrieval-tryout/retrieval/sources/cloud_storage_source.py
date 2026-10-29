"""
Cloud Storage Source Handler
Supports: AWS S3, Google Cloud Storage, Azure Blob Storage
"""

import os
import boto3
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse

from azure.storage.blob import BlobServiceClient
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class CloudStorageSource:
    """
    Handle

s file discovery and download from cloud storage
    """

    @staticmethod
    def _sanitize_failed_filename_component(value: Any, default: str) -> str:
        """Convert cloud paths into safe filename-like values."""
        raw_value = str(value or "").strip()
        if not raw_value:
            return default

        normalized = raw_value.replace("\\", "/").rstrip("/")
        if "://" in normalized:
            normalized = normalized.split("://", 1)[1]
        if "/" in normalized:
            normalized = normalized.rsplit("/", 1)[-1]

        import re

        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
        return normalized or default

    @classmethod
    def build_discovery_failed_filename(cls, source_details: Dict[str, Any]) -> str:
        """Build the failed-files entry for cloud storage discovery failures."""
        return cls._sanitize_failed_filename_component(
            source_details.get("folder_path")
            or source_details.get("filename")
            or source_details.get("name"),
            "cloud_storage",
        )

    def __init__(self, request_data: Dict[str, Any], provider: str):
        """
        Initialize cloud storage source

        Args:
            request_data: Complete request JSON
            provider: Cloud provider (aws, gcp, azure)
        """
        self.request_data = request_data
        self.provider = provider.lower()
        self.source_details = request_data.get('source_details', {})

        # Parse folder path (e.g., s3://bucket/prefix/)
        self.folder_path = self.source_details.get('folder_path', '')
        self.bucket, self.prefix = self._parse_folder_path(self.folder_path)

        # Credentials
        self.secret = self.source_details.get('secret', {})
        self.region = self.source_details.get('region_name', 'us-east-1')

        # Temp directory for downloads
        self.temp_dir = Path(tempfile.mkdtemp(prefix="cloud_storage_"))

        logger.info(f"Cloud Storage Source initialized: {self.provider}")
        logger.info(f"  Bucket: {self.bucket}")
        logger.info(f"  Prefix: {self.prefix}")

    def _parse_folder_path(self, folder_path: str) -> tuple:
        """Parse s3://bucket/prefix or Azure URLs into bucket/container and prefix."""
        if not folder_path:
            return "", ""

        path = folder_path.strip()
        if path.startswith("azure://"):
            path = path[len("azure://") :]
        elif path.startswith(("http://", "https://")):
            parsed = urlparse(path)
            path = parsed.path.lstrip("/")
        elif path.startswith("s3://"):
            path = path[len("s3://") :]
        elif path.startswith("gs://"):
            path = path[len("gs://") :]

        parts = path.split('/', 1)
        bucket = parts[0] if parts else ""
        prefix = parts[1] if len(parts) > 1 else ''

        return bucket, prefix

    async def discover(self) -> List[Dict[str, Any]]:
        """Discover and download files from cloud storage"""
        if 'aws' in self.provider or 's3' in self.provider:
            return await self._discover_s3()
        elif 'gcp' in self.provider or 'google' in self.provider:
            return await self._discover_gcs()
        elif 'azure' in self.provider:
            return await self._discover_azure()
        else:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")

    async def _discover_s3(self) -> List[Dict[str, Any]]:
        """Discover files from AWS S3"""
        try:
            logger.info("🔍 Connecting to AWS S3...")

            # Create S3 client
            s3 = boto3.client(
                's3',
                aws_access_key_id=self.secret.get('access_key_id'),
                aws_secret_access_key=self.secret.get('secret_access_key'),
                region_name=self.region
            )

            # List objects (paginated — a single call returns at most 1000)
            paginator = s3.get_paginator('list_objects_v2')
            objects = (
                obj
                for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
                for obj in page.get('Contents', [])
            )

            files = []
            for obj in objects:
                key = obj['Key']

                # Skip folders
                if key.endswith('/'):
                    continue

                filename = os.path.basename(key)
                local_path = self._local_path_for(key)

                logger.info(f"⬇️  Downloading: {key}")
                s3.download_file(self.bucket, key, str(local_path))

                # Handle ZIP files (only real .zip — .docx/.pptx/.xlsx are zip
                # containers too and must go to their own extractors intact)
                if Path(filename).suffix.lower() == '.zip' and zipfile.is_zipfile(local_path):
                    extracted = self._extract_zip(local_path, filename)
                    files.extend(extracted)
                else:
                    files.append({
                        'filename': filename,
                        'file_path': str(local_path),
                        'file_type': Path(filename).suffix.lower(),
                        'content_size_bytes': local_path.stat().st_size,
                        'source': 's3',
                        's3_key': key
                    })

            logger.info(f"✅ S3 discovery complete: {len(files)} file(s)")
            return files

        except Exception as e:
            logger.error(f"S3 discovery failed: {e}", exc_info=True)
            raise

    async def _discover_gcs(self) -> List[Dict[str, Any]]:
        """Discover files from Google Cloud Storage"""
        try:
            logger.info("Connecting to Google Cloud Storage...")

            # Build credentials from request
            creds_dict = self._build_gcp_credentials()
            logger.info(
                "GCS credential context: project_id=%s, client_email=%s",
                creds_dict.get("project_id") or "N/A",
                creds_dict.get("client_email") or "N/A",
            )
            credentials = service_account.Credentials.from_service_account_info(creds_dict)

            # Create GCS client
            client = storage.Client(credentials=credentials)
            bucket = client.bucket(self.bucket)
            logger.info(
                "Listing GCS objects: bucket=%s, prefix=%r, folder_path=%r",
                self.bucket,
                self.prefix,
                self.folder_path,
            )
            blobs = bucket.list_blobs(prefix=self.prefix)

            files = []
            matched_blob_names = []
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue

                matched_blob_names.append(blob.name)
                filename = os.path.basename(blob.name)
                local_path = self._local_path_for(blob.name)

                logger.info("Downloading GCS object: %s", blob.name)
                blob.download_to_filename(str(local_path))

                # Handle ZIP files (only real .zip — see S3 branch)
                if Path(filename).suffix.lower() == '.zip' and zipfile.is_zipfile(local_path):
                    extracted = self._extract_zip(local_path, filename)
                    files.extend(extracted)
                else:
                    files.append({
                        'filename': filename,
                        'file_path': str(local_path),
                        'file_type': Path(filename).suffix.lower(),
                        'content_size_bytes': local_path.stat().st_size,
                        'source': 'gcs',
                        'gcs_blob': blob.name
                    })

            if matched_blob_names:
                logger.info(
                    "GCS matched %d blob(s): %s",
                    len(matched_blob_names),
                    matched_blob_names,
                )
            else:
                logger.warning(
                    "GCS listing returned no blobs for bucket=%s prefix=%r",
                    self.bucket,
                    self.prefix,
                )

            logger.info("GCS discovery complete: %d file(s)", len(files))
            return files

        except Exception as e:
            logger.error(f"GCS discovery failed: {e}", exc_info=True)
            raise

    def _resolve_azure_connection_details(self) -> tuple[str, str, str]:
        """Resolve Azure Blob Storage connection string, container, and prefix."""
        connection_string = (
            self.source_details.get("connection_string")
            or self.source_details.get("storage_connection_string")
            or self.source_details.get("conn_string")
            or self.secret.get("connection_string")
            or self.secret.get("storage_connection_string")
            or self.secret.get("conn_string")
            or ""
        )
        container_name = (
            self.source_details.get("container_name")
            or self.source_details.get("bucket_name")
            or self.source_details.get("container")
            or ""
        )

        folder_path = str(self.source_details.get("folder_path", "") or "").strip()
        prefix = (
            str(self.source_details.get("blob_prefix", "") or "").strip()
            or str(self.source_details.get("prefix", "") or "").strip()
            or str(self.source_details.get("filename", "") or "").strip()
        )

        azure_container = ""
        azure_prefix = ""
        if folder_path:
            if folder_path.startswith("azure://"):
                path = folder_path[len("azure://") :]
            elif folder_path.startswith(("http://", "https://")):
                parsed = urlparse(folder_path)
                path = parsed.path.lstrip("/")
            else:
                path = folder_path.lstrip("/")

            if path:
                path_parts = path.split("/", 1)
                azure_container = path_parts[0]
                azure_prefix = path_parts[1] if len(path_parts) > 1 else ""

        if not container_name and azure_container:
            container_name = azure_container

        if not prefix:
            if azure_prefix:
                prefix = azure_prefix
            elif folder_path and container_name and not folder_path.startswith(("azure://", "http://", "https://")):
                if folder_path.startswith(container_name + "/"):
                    prefix = folder_path[len(container_name) + 1 :]
                else:
                    prefix = folder_path

        if container_name and folder_path and folder_path.startswith(("azure://", "http://", "https://")) and azure_container == container_name:
            # Avoid preserving the container name twice when folder_path includes it.
            prefix = azure_prefix

        prefix = prefix.lstrip("/") if prefix else ""
        return connection_string, container_name, prefix

    async def _discover_azure(self) -> List[Dict[str, Any]]:
        """Discover files from Azure Blob Storage"""
        try:
            connection_string, container_name, prefix = self._resolve_azure_connection_details()
            if not connection_string:
                raise ValueError(
                    "Azure Blob Storage requires connection_string in source_details or source_details.secret"
                )
            if not container_name:
                raise ValueError(
                    "Azure Blob Storage requires container_name in source_details or an azure://container/path in folder_path"
                )

            logger.info(
                "Connecting to Azure Blob Storage: container=%s prefix=%r",
                container_name,
                prefix,
            )
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service.get_container_client(container_name)

            blobs = container_client.list_blobs(name_starts_with=prefix or None)
            files: List[Dict[str, Any]] = []
            matched_blob_names = []
            for blob in blobs:
                if blob.name.endswith("/"):
                    continue

                matched_blob_names.append(blob.name)
                filename = os.path.basename(blob.name)
                local_path = self._local_path_for(blob.name)

                logger.info("Downloading Azure blob: %s", blob.name)
                with open(local_path, "wb") as file_handle:
                    stream = container_client.get_blob_client(blob.name).download_blob()
                    file_handle.write(stream.readall())

                files.append({
                    "filename": filename,
                    "file_path": str(local_path),
                    "file_type": Path(filename).suffix.lower(),
                    "content_size_bytes": local_path.stat().st_size,
                    "source": "azure",
                    "azure_container": container_name,
                    "azure_blob": blob.name,
                })

            if matched_blob_names:
                logger.info(
                    "Azure matched %d blob(s): %s",
                    len(matched_blob_names),
                    matched_blob_names,
                )
            else:
                logger.warning(
                    "Azure listing returned no blobs for container=%s prefix=%r",
                    container_name,
                    prefix,
                )

            logger.info("Azure discovery complete: %d file(s)", len(files))
            return files
        except Exception as e:
            logger.error("Azure discovery failed: %s", e, exc_info=True)
            raise

    def _build_gcp_credentials(self) -> Dict[str, Any]:
        """Build GCP credentials dictionary from request"""
        secret = self.secret
        if "service_account_json" in secret:
            import json

            service_account_info = json.loads(secret["service_account_json"])

        # 🔥 FIX newline issue
            service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

            return service_account_info
        return {
            "type": secret.get("type", "service_account"),
            "project_id": secret.get("project_id"),
            "private_key_id": secret.get("private_key_id"),
            "private_key": (secret.get("private_key") or "").replace("\\n", "\n"),
            "client_email": secret.get("client_email"),
            "client_id": secret.get("client_id"),
            "auth_uri": secret.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": secret.get("token_uri", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": secret.get("auth_provider_cert_url"),
            "client_x509_cert_url": secret.get("client_cert_url"),
        }

    def _local_path_for(self, object_key: str) -> Path:
        """Map an object key to a local path under temp_dir.

        Keeps the key's folder structure so files with the same name in
        different folders don't overwrite each other on download.
        """
        parts = [
            p for p in object_key.replace("\\", "/").split("/")
            if p not in ("", ".", "..")
        ]
        local_path = self.temp_dir.joinpath(*parts) if parts else self.temp_dir / "unnamed"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return local_path

    def _extract_zip(self, zip_path: Path, zip_name: str) -> List[Dict[str, Any]]:
        """Extract ZIP file and return list of extracted files"""
        extract_dir = zip_path.parent / f"{zip_path.name}_extracted"
        extract_dir.mkdir(exist_ok=True)

        logger.info(f"📦 Extracting ZIP: {zip_name}")

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

        files = []
        for root, dirnames, filenames in os.walk(extract_dir):
            # Skip macOS archive metadata and hidden folders
            dirnames[:] = [d for d in dirnames if d != '__MACOSX' and not d.startswith('.')]
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                file_path = Path(root) / filename
                files.append({
                    'filename': filename,
                    'file_path': str(file_path),
                    'file_type': file_path.suffix.lower(),
                    'content_size_bytes': file_path.stat().st_size,
                    'source': 'zip',
                    'zip_archive': zip_name
                })

        logger.info(f"  ✓ Extracted {len(files)} file(s) from {zip_name}")
        return files

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("🧹 Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
