#!/usr/bin/env python3
"""Publish the EvoRS ModelArts image directly with the OCI Registry API.

This publisher is used when the build host is x86 and Docker Desktop is not
available.  It creates one deterministic application layer, mounts the existing
Ascend base layers inside SWR, writes a new ARM64 image config, and pushes the
manifest.  No cloud credential is copied into the image or evidence output.
"""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import BinaryIO, Iterable
from urllib.parse import urljoin

import requests
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkswr.v2 import CreateAuthorizationTokenRequest, SwrClient
from huaweicloudsdkswr.v2.region.swr_region import SwrRegion


ROOT = Path(__file__).resolve().parents[1]
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
OCI_LAYER_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
MANIFEST_ACCEPT = ", ".join(
    (
        OCI_MANIFEST,
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    )
)


def _load_key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith(("#", ";")) and "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _parse_image(value: str) -> tuple[str, str, str]:
    registry, remainder = value.split("/", 1)
    repository, tag = remainder.rsplit(":", 1)
    if not registry or not repository or not tag:
        raise ValueError(f"invalid image reference: {value}")
    return registry, repository, tag


class Registry:
    def __init__(self, registry: str, username: str, password: str):
        self.registry = registry
        self.base_url = f"https://{registry}"
        self.username = username
        self.password = password
        self.session = requests.Session()

    def token(self, scopes: Iterable[str]) -> str:
        params: list[tuple[str, str]] = [("service", "dockyard")]
        params.extend(("scope", scope) for scope in scopes)
        response = self.session.get(
            f"{self.base_url}/swr/auth/v2/registry/auth/",
            params=params,
            auth=(self.username, self.password),
            timeout=60,
        )
        response.raise_for_status()
        value = response.json().get("token") or response.json().get("access_token")
        if not value:
            raise RuntimeError("SWR did not return a registry bearer token")
        return value

    @staticmethod
    def _auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def get_manifest(self, repository: str, reference: str, token: str):
        return self.session.get(
            f"{self.base_url}/v2/{repository}/manifests/{reference}",
            headers={**self._auth(token), "Accept": MANIFEST_ACCEPT},
            timeout=60,
        )

    def get_blob(self, repository: str, digest: str, token: str) -> bytes:
        response = self.session.get(
            f"{self.base_url}/v2/{repository}/blobs/{digest}",
            headers=self._auth(token),
            timeout=120,
        )
        response.raise_for_status()
        return response.content

    def blob_exists(self, repository: str, digest: str, token: str) -> bool:
        response = self.session.head(
            f"{self.base_url}/v2/{repository}/blobs/{digest}",
            headers=self._auth(token),
            timeout=60,
        )
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def mount_blob(
        self, target: str, source: str, digest: str, token: str
    ) -> bool:
        if self.blob_exists(target, digest, token):
            return False
        response = self.session.post(
            f"{self.base_url}/v2/{target}/blobs/uploads/",
            params={"mount": digest, "from": source},
            headers=self._auth(token),
            allow_redirects=False,
            timeout=60,
        )
        if response.status_code == 201:
            return True
        if response.status_code == 202:
            location = response.headers.get("Location")
            if location:
                self.session.delete(
                    urljoin(self.base_url, location),
                    headers=self._auth(token),
                    timeout=60,
                )
            raise RuntimeError(
                f"SWR did not cross-mount {digest}; refusing a multi-GiB fallback copy"
            )
        response.raise_for_status()
        return False

    def upload_blob(
        self, repository: str, digest: str, payload: bytes | BinaryIO, token: str
    ) -> bool:
        if self.blob_exists(repository, digest, token):
            return False
        start = self.session.post(
            f"{self.base_url}/v2/{repository}/blobs/uploads/",
            headers=self._auth(token),
            allow_redirects=False,
            timeout=60,
        )
        if start.status_code != 202:
            start.raise_for_status()
        location = start.headers.get("Location")
        if not location:
            raise RuntimeError("SWR upload session did not return Location")
        upload_url = urljoin(self.base_url, location)
        separator = "&" if "?" in upload_url else "?"
        response = self.session.put(
            f"{upload_url}{separator}digest={digest}",
            data=payload,
            headers={**self._auth(token), "Content-Type": "application/octet-stream"},
            timeout=(60, 1800),
        )
        if response.status_code != 201:
            response.raise_for_status()
        return True

    def put_manifest(
        self, repository: str, tag: str, payload: bytes, token: str
    ) -> str:
        response = self.session.put(
            f"{self.base_url}/v2/{repository}/manifests/{tag}",
            data=payload,
            headers={
                **self._auth(token),
                "Content-Type": OCI_MANIFEST,
            },
            timeout=120,
        )
        if response.status_code not in (201, 202):
            response.raise_for_status()
        return response.headers.get("Docker-Content-Digest") or _sha256_bytes(payload)


def _iter_files(source: Path) -> Iterable[Path]:
    ignored_parts = {"__pycache__", ".pytest_cache"}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if ignored_parts.intersection(relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def _layer_mappings() -> list[tuple[Path, PurePosixPath, int]]:
    mappings: list[tuple[Path, PurePosixPath, int]] = []
    for local_name, image_name in (
        ("src", "opt/evors/src"),
        ("scripts", "opt/evors/scripts"),
        ("configs", "opt/evors/configs"),
        ("tests", "opt/evors/tests"),
        (".modelarts/vendor", "opt/evors/vendor"),
    ):
        source = ROOT / local_name
        for path in _iter_files(source):
            target = PurePosixPath(image_name) / path.relative_to(source).as_posix()
            mode = 0o755 if local_name == "scripts" and path.suffix in {".py", ".sh"} else 0o644
            mappings.append((path, target, mode))
    for name in (
        "pyproject.toml",
        "requirements-base.txt",
        "requirements-vlm.txt",
        "requirements-ascend.txt",
    ):
        mappings.append((ROOT / name, PurePosixPath("opt/evors") / name, 0o644))
    mappings.append(
        (
            ROOT / "evidence/2026-09-03_phase5_routing/checkpoint.json",
            PurePosixPath("opt/evors/checkpoints/phase5-routing.json"),
            0o644,
        )
    )
    mappings.append(
        (
            ROOT / "docker/modelarts-entrypoint.sh",
            PurePosixPath("usr/local/bin/evors-modelarts"),
            0o755,
        )
    )
    return sorted(mappings, key=lambda item: item[1].as_posix())


def _make_layer(tar_path: Path, gzip_path: Path) -> dict[str, object]:
    mappings = _layer_mappings()
    directories: set[PurePosixPath] = set()
    for _, target, _ in mappings:
        parent = target.parent
        while parent.as_posix() not in (".", ""):
            directories.add(parent)
            parent = parent.parent

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for directory in sorted(directories, key=lambda value: (len(value.parts), value.as_posix())):
            info = tarfile.TarInfo(directory.as_posix())
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid = 1000
            info.gid = 1000
            info.mtime = 0
            archive.addfile(info)
        for source, target, mode in mappings:
            info = tarfile.TarInfo(target.as_posix())
            info.size = source.stat().st_size
            info.mode = mode
            info.uid = 1000
            info.gid = 1000
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)

    diff_id = _sha256_file(tar_path)
    with tar_path.open("rb") as source, gzip_path.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, fileobj=raw_output, mtime=0
        ) as compressed:
            shutil.copyfileobj(source, compressed, length=1024 * 1024)
    return {
        "diff_id": diff_id,
        "digest": _sha256_file(gzip_path),
        "size": gzip_path.stat().st_size,
        "uncompressed_size": tar_path.stat().st_size,
        "file_count": len(mappings),
    }


def _updated_environment(existing: list[str]) -> list[str]:
    overrides = {
        "EVORS_ROOT": "/opt/evors",
        "EVORS_OUTPUT_DIR": "/home/ma-user/evors-output",
        "PYTHONPATH": "/opt/evors/src:/opt/evors/vendor",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ASCEND_VISIBLE_DEVICES": "all",
        "HCCL_WHITELIST_DISABLE": "1",
    }
    values = [item for item in existing if item.split("=", 1)[0] not in overrides]
    values.extend(f"{key}={value}" for key, value in overrides.items())
    return values


def _build_config(base: dict, layer: dict, created: str, revision: str) -> bytes:
    result = copy.deepcopy(base)
    result["created"] = created
    result["architecture"] = "arm64"
    result["os"] = "linux"
    runtime = result.setdefault("config", {})
    runtime["User"] = "1000:1000"
    runtime["WorkingDir"] = "/opt/evors"
    runtime["Entrypoint"] = ["/usr/local/bin/evors-modelarts"]
    runtime["Cmd"] = ["verify"]
    runtime["Env"] = _updated_environment(runtime.get("Env") or [])
    labels = runtime.setdefault("Labels", {})
    labels.update(
        {
            "org.opencontainers.image.title": "EvoRS-Comm Ascend experiments",
            "org.opencontainers.image.description": "EvoRS-Comm ARM64/Ascend SNT9B experiment runtime",
            "org.opencontainers.image.revision": revision,
            "org.opencontainers.image.created": created,
            "ai.modelarts.flavor": "ASCEND_SNT9B",
            "ai.modelarts.default_uid": "1000",
        }
    )
    result.setdefault("rootfs", {}).setdefault("diff_ids", []).append(layer["diff_id"])
    result.setdefault("history", []).append(
        {
            "created": created,
            "created_by": "EvoRS deterministic OCI packager",
            "comment": "EvoRS source, configuration, tests, and ARM64 offline dependencies",
        }
    )
    result.pop("container", None)
    result.pop("container_config", None)
    return json.dumps(result, sort_keys=True, separators=(",", ":")).encode()


def publish(plan_path: Path, credential_path: Path, evidence_dir: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    target_image = plan["image"]
    base_image = plan["base_image"]
    registry, target_repository, tag = _parse_image(target_image)
    base_registry, source_repository, base_tag = _parse_image(base_image)
    if registry != base_registry:
        raise ValueError("base and target must use the same SWR registry")

    credentials = _load_key_values(credential_path)
    ak = credentials.get("HUAWEICLOUD_ACCESS_KEY") or credentials.get("Access Key Id")
    sk = credentials.get("HUAWEICLOUD_SECRET_KEY") or credentials.get(
        "Secret Access Key"
    )
    if not ak or not sk:
        raise RuntimeError("Huawei access key fields were not found")
    swr = (
        SwrClient.new_builder()
        .with_credentials(BasicCredentials(ak, sk, plan["project_id"]))
        .with_region(SwrRegion.value_of(plan["region"]))
        .build()
    )
    authorization = swr.create_authorization_token(CreateAuthorizationTokenRequest())
    encoded = authorization.auths[registry].auth
    username, password = base64.b64decode(encoded).decode().split(":", 1)
    client = Registry(registry, username, password)
    token = client.token(
        (
            f"repository:{source_repository}:pull",
            f"repository:{target_repository}:pull,push",
        )
    )

    existing = client.get_manifest(target_repository, tag, token)
    if existing.status_code == 200:
        raise RuntimeError(f"immutable target tag already exists: {target_image}")
    if existing.status_code != 404:
        existing.raise_for_status()

    base_response = client.get_manifest(source_repository, base_tag, token)
    base_response.raise_for_status()
    base_manifest = base_response.json()
    if "manifests" in base_manifest:
        raise RuntimeError("multi-platform base indexes are not supported")
    base_digest = base_response.headers.get("Docker-Content-Digest")
    base_config = json.loads(
        client.get_blob(source_repository, base_manifest["config"]["digest"], token)
    )
    if base_config.get("architecture") != "arm64":
        raise RuntimeError("base image is not arm64")

    build_dir = ROOT / ".codex_work" / "oci-build"
    build_dir.mkdir(parents=True, exist_ok=True)
    tar_path = build_dir / "evors-application.tar"
    gzip_path = build_dir / "evors-application.tar.gz"
    layer = _make_layer(tar_path, gzip_path)
    print(
        f"APPLICATION_LAYER files={layer['file_count']} "
        f"compressed_bytes={layer['size']} digest={layer['digest']}"
    )

    mounted = 0
    reused = 0
    for index, descriptor in enumerate(base_manifest["layers"], start=1):
        if client.mount_blob(
            target_repository, source_repository, descriptor["digest"], token
        ):
            mounted += 1
        else:
            reused += 1
        if index % 10 == 0 or index == len(base_manifest["layers"]):
            print(f"BASE_LAYERS_READY {index}/{len(base_manifest['layers'])}")

    created = plan.get("build_date") or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    config_payload = _build_config(base_config, layer, created, plan["vcs_ref"])
    config_digest = _sha256_bytes(config_payload)
    client.upload_blob(target_repository, config_digest, config_payload, token)
    with gzip_path.open("rb") as stream:
        client.upload_blob(target_repository, str(layer["digest"]), stream, token)

    manifest = {
        "schemaVersion": 2,
        "mediaType": OCI_MANIFEST,
        "config": {
            "mediaType": OCI_CONFIG,
            "digest": config_digest,
            "size": len(config_payload),
        },
        "layers": [
            *base_manifest["layers"],
            {
                "mediaType": OCI_LAYER_GZIP,
                "digest": layer["digest"],
                "size": layer["size"],
            },
        ],
        "annotations": {
            "org.opencontainers.image.created": created,
            "org.opencontainers.image.revision": plan["vcs_ref"],
            "org.opencontainers.image.title": "EvoRS-Comm Ascend experiments",
        },
    }
    manifest_payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    remote_digest = client.put_manifest(
        target_repository, tag, manifest_payload, token
    )

    verify_response = client.get_manifest(target_repository, tag, token)
    verify_response.raise_for_status()
    verified_digest = verify_response.headers.get("Docker-Content-Digest")
    if verified_digest != remote_digest:
        raise RuntimeError(
            f"remote digest mismatch: put={remote_digest} get={verified_digest}"
        )
    remote_manifest = verify_response.json()
    total_size = sum(int(item["size"]) for item in remote_manifest["layers"])

    result = {
        "schema_version": "evors.modelarts-image-publish.v1",
        "image": target_image,
        "digest": remote_digest,
        "base_image": base_image,
        "base_digest": base_digest,
        "architecture": "arm64",
        "default_user": "1000:1000",
        "entrypoint": ["/usr/local/bin/evors-modelarts"],
        "default_command": ["verify"],
        "application_layer": layer,
        "base_layers_mounted": mounted,
        "base_layers_reused": reused,
        "layer_count": len(remote_manifest["layers"]),
        "compressed_layer_bytes": total_size,
        "content_sha256": plan["content_sha256"],
        "published_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "swr_publish.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tar_path.unlink(missing_ok=True)
    gzip_path.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / ".codex_work/modelarts-release-plan.json"
    )
    parser.add_argument("--credentials", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / "evidence/2026-09-19_modelarts_image",
    )
    args = parser.parse_args()
    result = publish(
        args.plan.resolve(), args.credentials.resolve(), args.evidence_dir.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
