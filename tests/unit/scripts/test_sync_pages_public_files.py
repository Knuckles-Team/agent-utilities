"""The one-link Pages downloads must be the exact published source bytes."""

from pathlib import Path

from scripts.sync_pages_public_files import PUBLIC_FILES, sync


def test_pages_downloads_match_authorities(tmp_path: Path) -> None:
    for source_name, _ in PUBLIC_FILES:
        source = tmp_path / source_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"authority:{source_name}\n".encode())
    assert sync(tmp_path, write=True) == []
    assert sync(tmp_path, write=False) == []
    for source_name, public_name in PUBLIC_FILES:
        assert (tmp_path / public_name).read_bytes() == (tmp_path / source_name).read_bytes()


def test_pages_download_check_rejects_pointer_and_missing_file(tmp_path: Path) -> None:
    for source_name, public_name in PUBLIC_FILES:
        source = tmp_path / source_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"actual content\n")
        public = tmp_path / public_name
        public.parent.mkdir(parents=True, exist_ok=True)
        public.write_bytes(f"../{source_name}\n".encode())
    assert len(sync(tmp_path, write=False)) == len(PUBLIC_FILES)
    (tmp_path / PUBLIC_FILES[0][1]).unlink()
    assert len(sync(tmp_path, write=False)) == len(PUBLIC_FILES)
