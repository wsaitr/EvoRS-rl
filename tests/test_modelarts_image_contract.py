from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_modelarts_image_contract() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.modelarts-ascend").read_text(
        encoding="utf-8"
    )
    assert "USER 1000:1000" in dockerfile
    assert "ENTRYPOINT [\"/usr/local/bin/evors-modelarts\"]" in dockerfile
    assert "COPY --link --chown=1000:1000 src/" in dockerfile
    assert "COPY --link --chown=1000:1000 .modelarts/vendor/" in dockerfile
    assert "torch_npu" not in dockerfile
    assert not any(line.startswith("RUN ") for line in dockerfile.splitlines())


def test_modelarts_entrypoint_exposes_required_modes() -> None:
    entrypoint = (ROOT / "docker" / "modelarts-entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert 'verify)' in entrypoint
    assert 'routing)' in entrypoint
    assert 'test)' in entrypoint
    assert '--require-npu' in entrypoint
    assert 'exec "${mode}" "$@"' in entrypoint


def test_docker_context_excludes_credentials_and_local_models() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    }
    assert ".env" in patterns
    assert ".models/" in patterns
    assert ".codex_work/" in patterns
    assert "data/" in patterns
    assert "outputs/" in patterns
