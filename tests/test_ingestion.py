from pathlib import Path
from unittest.mock import Mock, patch

from compagnon_immo.data.ingestion import download_file


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