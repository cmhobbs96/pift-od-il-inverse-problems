"""Modal serverless GPU backend for PIFT pipelines.

This module is imported lazily by the run manager so that users without the
``modal`` extra installed are unaffected.

Design
------
A single generic Modal function (``_run_pipeline_remote``) executes any
pipeline against a temp output directory inside the container, then walks the
directory and packs all written files into a ``{relpath: bytes}`` dict that is
returned to the client alongside the pipeline result. The client unpacks the
bytes into the local ``run_output_root``, so downstream code (storage, GUI)
sees the same on-disk layout as a local run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import io

import numpy as np

try:
    import modal
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'modal' package is required for the Modal backend. "
        "Install with: pip install -e '.[modal]'"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = modal.App("pift-od-il")

# Build the image from the project's pyproject so the remote env matches local.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject(
        str(PROJECT_ROOT / "pyproject.toml"),
        optional_dependencies=["gpu"],
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/root/src")
)


def _serialize_obs(obs_data: tuple[np.ndarray, np.ndarray] | None) -> bytes | None:
    if obs_data is None:
        return None
    buf = io.BytesIO()
    np.savez(buf, x=obs_data[0], y=obs_data[1])
    return buf.getvalue()


def _deserialize_obs(blob: bytes | None) -> tuple[np.ndarray, np.ndarray] | None:
    if blob is None:
        return None
    with np.load(io.BytesIO(blob)) as data:
        return np.asarray(data["x"]), np.asarray(data["y"])


def _pack_dir(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


def _unpack_dir(files: dict[str, bytes], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for rel, blob in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)


def _strip_local_paths(result: dict[str, Any], remote_root: Path) -> dict[str, Any]:
    """Replace absolute remote paths in artifacts with relative ones."""
    artifacts = result.get("artifacts", {})
    if isinstance(artifacts, dict):
        for k, v in list(artifacts.items()):
            if isinstance(v, str) and v.startswith(str(remote_root)):
                artifacts[k] = str(Path(v).relative_to(remote_root))
    return result


@app.function(image=image, gpu="A10G", timeout=3600)
def _run_pipeline_remote(
    example_key: str,
    method_key: str | None,
    config: dict[str, Any],
    obs_blob: bytes | None,
) -> dict[str, Any]:
    """Generic remote entry point. Runs any pipeline and returns artifacts inline."""
    import tempfile

    # Force local backend on the remote side to avoid recursive Modal dispatch.
    config = dict(config)
    config["compute"] = {"backend": "local", "device": "gpu"}

    obs_data = _deserialize_obs(obs_blob)

    # Late imports so they only happen inside the container.
    from run_manager import _PIPELINE_DISPATCH, _EXAMPLES_WITH_OBS  # type: ignore

    runner = _PIPELINE_DISPATCH.get((example_key, method_key))
    if runner is None:
        raise ValueError(
            f"No pipeline for example={example_key!r}, method={method_key!r}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        remote_root = Path(tmp) / "run"
        kwargs: dict[str, Any] = {
            "cfg": config,
            "output_root": remote_root,
            "device_preference": "gpu",
            "save_outputs": True,
        }
        if example_key in _EXAMPLES_WITH_OBS:
            kwargs["obs_csv_path"] = None
            kwargs["obs_data"] = obs_data

        result = runner(**kwargs)
        result = _strip_local_paths(result, remote_root)
        files = _pack_dir(remote_root)

    return {"result": result, "files": files}


class ModalBackend:
    """Client-side handle for submitting pipeline runs to Modal."""

    def run(
        self,
        example_key: str,
        method_key: str | None,
        config: dict[str, Any],
        obs_data: tuple[np.ndarray, np.ndarray] | None,
        local_output_root: Path,
    ) -> dict[str, Any]:
        obs_blob = _serialize_obs(obs_data)
        with app.run():
            payload = _run_pipeline_remote.remote(
                example_key, method_key, config, obs_blob
            )
        result = payload["result"]
        files = payload["files"]
        _unpack_dir(files, local_output_root)

        # Rewrite relative artifact paths back to absolute local paths.
        artifacts = result.get("artifacts", {})
        if isinstance(artifacts, dict):
            for k, v in list(artifacts.items()):
                if isinstance(v, str) and not Path(v).is_absolute():
                    artifacts[k] = str(local_output_root / v)
        return result
