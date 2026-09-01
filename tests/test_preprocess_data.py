import sys

import pytest

from scripts.preprocess_data import parse_args


def test_parse_args_rejects_unsupported_year(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_data.py",
            "--year",
            "2035",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()