import json

import pytest

from glys_rul import config
from glys_rul.errors import DataContractError
from glys_rul.io import verify_manifest, write_manifest


@pytest.mark.slow
def test_shipped_dataset_matches_manifest():
    verify_manifest(config.DATA_DIR, config.MANIFEST)


@pytest.mark.slow
def test_manifest_records_eleven_images_and_the_scale():
    recorded = json.loads(config.MANIFEST.read_text())
    images = [n for n in recorded if n.endswith(".jpeg")]
    assert len(images) == 11
    assert "temp.png" in recorded


def test_tampered_file_is_detected(tmp_path):
    (tmp_path / "a.txt").write_text("original")
    manifest = tmp_path / "MANIFEST.json"
    write_manifest(tmp_path, manifest)

    (tmp_path / "a.txt").write_text("tampered")

    with pytest.raises(DataContractError, match="content changed"):
        verify_manifest(tmp_path, manifest)


def test_missing_file_is_detected(tmp_path):
    (tmp_path / "a.txt").write_text("original")
    manifest = tmp_path / "MANIFEST.json"
    write_manifest(tmp_path, manifest)

    (tmp_path / "a.txt").unlink()

    with pytest.raises(DataContractError, match="missing"):
        verify_manifest(tmp_path, manifest)
