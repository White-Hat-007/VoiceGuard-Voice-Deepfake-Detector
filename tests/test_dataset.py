import pytest

from download_dataset import load_excluded_test_hashes, validate_test_shards


def test_validate_test_shards_accepts_unique_official_indices() -> None:
    assert validate_test_shards([1, 7, 62]) == (1, 7, 62)


@pytest.mark.parametrize(
    "indices",
    [[], [0, 0], [-1], [63]],
)
def test_validate_test_shards_rejects_invalid_indices(
    indices: list[int],
) -> None:
    with pytest.raises(ValueError, match="unique values from 0 to 62"):
        validate_test_shards(indices)


def test_load_excluded_test_hashes_reads_prior_test_examples(tmp_path) -> None:
    manifest_path = tmp_path / "acquisition_manifest.json"
    manifest_path.write_text(
        (
            '{"splits":{"test":{"selected_files":{'
            '"fake":[{"sha256":"fake-hash"}],'
            '"real":[{"sha256":"real-hash"}]}}}}'
        ),
        encoding="utf-8",
    )

    assert load_excluded_test_hashes(manifest_path) == {
        "fake-hash",
        "real-hash",
    }


def test_load_excluded_test_hashes_rejects_unexpected_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "acquisition_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected structure"):
        load_excluded_test_hashes(manifest_path)
