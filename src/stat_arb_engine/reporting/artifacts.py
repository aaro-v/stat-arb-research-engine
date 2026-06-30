from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReportArtifact:
    path: str
    bytes: int
    sha256: str


def build_artifact_manifest(
    artifact_paths: list[Path],
    *,
    root: Path | None = None,
) -> list[ReportArtifact]:
    """Build deterministic size and checksum metadata for report artifacts."""

    if not artifact_paths:
        raise ValueError("artifact_paths must not be empty")

    manifest_root = Path.cwd() if root is None else root
    artifacts: list[ReportArtifact] = []
    for artifact_path in sorted(artifact_paths, key=lambda path: path.as_posix()):
        if not artifact_path.is_file():
            raise FileNotFoundError(f"report artifact does not exist: {artifact_path}")
        content = artifact_path.read_bytes()
        try:
            display_path = artifact_path.relative_to(manifest_root).as_posix()
        except ValueError:
            display_path = artifact_path.as_posix()
        artifacts.append(
            ReportArtifact(
                path=display_path,
                bytes=len(content),
                sha256=sha256(content).hexdigest(),
            )
        )
    return artifacts


def write_artifact_manifest(
    artifact_paths: list[Path],
    output_path: Path,
    *,
    root: Path | None = None,
) -> list[ReportArtifact]:
    """Write a CSV manifest and return the artifact rows that were recorded."""

    artifacts = build_artifact_manifest(artifact_paths, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "path": artifact.path,
                "bytes": artifact.bytes,
                "sha256": artifact.sha256,
            }
            for artifact in artifacts
        ]
    ).to_csv(output_path, index=False)
    return artifacts
