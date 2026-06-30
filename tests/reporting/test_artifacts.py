from __future__ import annotations

import re

import pandas as pd
import pytest

from stat_arb_engine.reporting import build_artifact_manifest, write_artifact_manifest


def test_build_artifact_manifest_records_sorted_relative_checksums(tmp_path) -> None:
    first = tmp_path / "z.csv"
    second = tmp_path / "a.csv"
    first.write_text("z\n", encoding="utf-8")
    second.write_text("alpha\n", encoding="utf-8")

    manifest = build_artifact_manifest([first, second], root=tmp_path)

    assert [artifact.path for artifact in manifest] == ["a.csv", "z.csv"]
    assert [artifact.bytes for artifact in manifest] == [6, 2]
    assert all(re.fullmatch(r"[0-9a-f]{64}", artifact.sha256) for artifact in manifest)


def test_build_artifact_manifest_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_artifact_manifest([])


def test_build_artifact_manifest_requires_existing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="report artifact does not exist"):
        build_artifact_manifest([tmp_path / "missing.csv"])


def test_write_artifact_manifest_writes_csv(tmp_path) -> None:
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    output = tmp_path / "manifest.csv"

    rows = write_artifact_manifest([artifact], output, root=tmp_path)

    frame = pd.read_csv(output)
    assert len(rows) == 1
    assert frame.to_dict("records") == [
        {
            "path": "artifact.csv",
            "bytes": 8,
            "sha256": rows[0].sha256,
        }
    ]
