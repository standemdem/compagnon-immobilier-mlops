from pathlib import Path

from compagnon_immo.data.ingestion import (
    convert_csv_gz_to_parquet,
)

RAW_DIR = Path("data/raw")
PARQUET_DIR = Path("data/parquet")

YEARS = [
    2020,
    2021,
    2022,
    2023,
    2024,
]


def main() -> None:
    for year in YEARS:
        input_path = RAW_DIR / f"full_{year}.csv.gz"
        output_path = PARQUET_DIR / f"full_{year}.parquet"

        print()
        print(f"=== Processing DVF {year} ===")

        convert_csv_gz_to_parquet(
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()