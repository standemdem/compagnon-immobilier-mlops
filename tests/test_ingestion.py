import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from compagnon_immo.data.ingestion import (
    download_file,
    convert_csv_gz_to_parquet,
    )


def test_download_file_uses_http_timeout(tmp_path):
    destination = tmp_path / "test.csv.gz"

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.raise_for_status = Mock()
    response.iter_content.return_value = []

    with patch(
        "compagnon_immo.data.ingestion.requests.get",
        return_value=response,
    ) as mock_get:
        download_file(
            "https://example.com/test.csv.gz",
            destination,
        )

    mock_get.assert_called_once_with(
        "https://example.com/test.csv.gz",
        stream=True,
        timeout=60,
    )

def test_download_file_removes_partial_file_on_error(tmp_path):
    destination = tmp_path / "test.csv.gz"
    temporary_path = tmp_path / "test.csv.gz.part"

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.raise_for_status = Mock()

    def failing_chunks(chunk_size):
        yield b"partial-data"
        raise RuntimeError("download interrupted")

    response.iter_content.side_effect = failing_chunks

    with patch(
        "compagnon_immo.data.ingestion.requests.get",
        return_value=response,
    ):
        with pytest.raises(RuntimeError, match="download interrupted"):
            download_file(
                "https://example.com/test.csv.gz",
                destination,
            )

    assert not destination.exists()
    assert not temporary_path.exists()

def test_convert_csv_gz_to_parquet_removes_partial_file_on_error(
    tmp_path,
):
    input_path = tmp_path / "input.csv.gz"
    output_path = tmp_path / "output.parquet"
    temporary_path = tmp_path / "output.parquet.part"

    input_path.touch()

    dataframe = Mock()
    dataframe.shape = (1, 1)

    def failing_to_parquet(path, engine, index):
        Path(path).write_bytes(b"partial-parquet")
        raise RuntimeError("parquet write interrupted")

    dataframe.to_parquet.side_effect = failing_to_parquet

    with patch(
        "compagnon_immo.data.ingestion.pd.read_csv",
        return_value=dataframe,
    ):
        with pytest.raises(
            RuntimeError,
            match="parquet write interrupted",
        ):
            convert_csv_gz_to_parquet(
                input_path,
                output_path,
            )

    assert not output_path.exists()
    assert not temporary_path.exists()