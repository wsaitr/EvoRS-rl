"""Huawei Cloud OBS data loader for RS-HMComm."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Iterator
from pathlib import Path
import os
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class OBSConfig:
    """OBS connection configuration."""
    endpoint: str = ""  # e.g. "obs.cn-north-4.myhuaweicloud.com"
    bucket: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""

    # Can also be set via environment variables
    @classmethod
    def from_env(cls) -> OBSConfig:
        return cls(
            endpoint=os.environ.get("OBS_ENDPOINT", ""),
            bucket=os.environ.get("OBS_BUCKET", "evors-data"),
            access_key_id=os.environ.get("OBS_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("OBS_SECRET_ACCESS_KEY", ""),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> OBSConfig:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_env(self) -> dict[str, str]:
        """Export as environment variables."""
        return {
            "OBS_ENDPOINT": self.endpoint,
            "OBS_BUCKET": self.bucket,
            "OBS_ACCESS_KEY_ID": self.access_key_id,
            "OBS_SECRET_ACCESS_KEY": self.secret_access_key,
        }

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.bucket and self.access_key_id and self.secret_access_key)


class OBSDataLoader:
    """
    Downloads and caches data from Huawei Cloud OBS.

    Usage:
        config = OBSConfig.from_env()
        loader = OBSDataLoader(config, cache_dir="./data/cache")

        # Download a single file
        local_path = loader.download("vrsbench/train/sample_001.jpg")

        # Download a directory/prefix
        loader.download_prefix("vrsbench/train/", local_dir="./data/vrsbench/train")

        # List objects
        keys = loader.list_objects("vrsbench/train/", max_keys=100)
    """

    def __init__(self, config: OBSConfig | None = None, cache_dir: str | Path = "./data/cache"):
        self.config = config or OBSConfig.from_env()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _get_client(self) -> Any:
        """Lazy init OBS client."""
        if self._client is not None:
            return self._client

        if not self.config.is_configured:
            raise RuntimeError(
                "OBS not configured. Set environment variables:\n"
                "  OBS_ENDPOINT, OBS_BUCKET, OBS_ACCESS_KEY_ID, OBS_SECRET_ACCESS_KEY\n"
                "Or create a config YAML and use OBSConfig.from_yaml()"
            )

        try:
            from obs import ObsClient
            self._client = ObsClient(
                access_key_id=self.config.access_key_id,
                secret_access_key=self.config.secret_access_key,
                server=self.config.endpoint,
            )
            return self._client
        except ImportError:
            raise ImportError(
                "esdk-obs-python is not installed. Install with:\n"
                "  pip install esdk-obs-python"
            )

    def _cache_path(self, key: str) -> Path:
        """Get local cache path for an OBS key."""
        safe_key = key.replace("/", os.sep)
        return self.cache_dir / safe_key

    def download(self, key: str, force: bool = False) -> Path:
        """Download a single file from OBS. Returns local path."""
        local_path = self._cache_path(key)

        if local_path.exists() and not force:
            logger.debug(f"Cache hit: {key} -> {local_path}")
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        client = self._get_client()

        logger.info(f"Downloading: {key}")
        resp = client.getObject(
            bucketName=self.config.bucket,
            objectKey=key,
            downloadPath=str(local_path),
        )

        if resp.status < 300:
            logger.info(f"Downloaded: {key} -> {local_path}")
            return local_path
        else:
            raise RuntimeError(f"OBS download failed: {resp.errorMessage} (status={resp.status})")

    def download_prefix(self, prefix: str, local_dir: str | Path | None = None,
                        max_keys: int = 10000, force: bool = False) -> list[Path]:
        """Download all objects under a prefix."""
        local_dir = Path(local_dir) if local_dir else self._cache_path(prefix.rstrip("/"))
        local_dir.mkdir(parents=True, exist_ok=True)

        keys = self.list_objects(prefix, max_keys=max_keys)
        paths = []
        for key in keys:
            try:
                p = self.download(key, force=force)
                paths.append(p)
            except Exception as e:
                logger.warning(f"Failed to download {key}: {e}")

        logger.info(f"Downloaded {len(paths)}/{len(keys)} files from {prefix}")
        return paths

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list[str]:
        """List object keys under a prefix."""
        client = self._get_client()
        keys = []

        resp = client.listObjects(
            bucketName=self.config.bucket,
            prefix=prefix,
            max_keys=max_keys,
        )

        if resp.status < 300:
            for obj in resp.body.contents:
                keys.append(obj.key)

        return keys

    def exists(self, key: str) -> bool:
        """Check if an object exists in OBS."""
        client = self._get_client()
        resp = client.getObjectMetadata(
            bucketName=self.config.bucket,
            objectKey=key,
        )
        return resp.status < 300

    def upload(self, local_path: str | Path, key: str) -> str:
        """Upload a local file to OBS."""
        client = self._get_client()
        local_path = Path(local_path)

        resp = client.putFile(
            bucketName=self.config.bucket,
            objectKey=key,
            file=str(local_path),
        )

        if resp.status < 300:
            logger.info(f"Uploaded: {local_path} -> obs://{self.config.bucket}/{key}")
            return f"obs://{self.config.bucket}/{key}"
        else:
            raise RuntimeError(f"OBS upload failed: {resp.errorMessage}")

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
