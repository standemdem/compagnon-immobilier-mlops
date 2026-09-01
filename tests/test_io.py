from unittest.mock import Mock

import pytest

from compagnon_immo.data.io import save_parquet_gzip


def test_save_parquet_gzip_preserves_existing_file_on_error(
    tmp_path,
):
    output_path = tmp_path / "dataset.parquet.gz"
    temporary_path = tmp_path / "dataset.parquet.gz.part"

    original_content = b"existing-valid-file"
    output_path.write_bytes(original_content)

    dataframe = Mock()

    def failing_to_parquet(
        path,
        engine,
        compression,
        index,
    ):
        path.write_bytes(b"partial-new-file")
        raise RuntimeError("parquet write interrupted")

    dataframe.to_parquet.side_effect = failing_to_parquet

    with pytest.raises(
        RuntimeError,
        match="parquet write interrupted",
    ):
        save_parquet_gzip(
            dataframe,
            output_path,
            overwrite=True,
        )

    assert output_path.read_bytes() == original_content
    assert not temporary_path.exists()

def test_save_parquet_gzip_rejects_existing_file_without_overwrite(
    tmp_path,
):
    output_path = tmp_path / "dataset.parquet.gz"
    original_content = b"existing-valid-file"
    output_path.write_bytes(original_content)

    dataframe = Mock()

    with pytest.raises(FileExistsError):
        save_parquet_gzip(
            dataframe,
            output_path,
            overwrite=False,
        )

    assert output_path.read_bytes() == original_content
    dataframe.to_parquet.assert_not_called()