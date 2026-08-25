from compagnon_immo.data.ingestion import (
    convert_csv_gz_to_parquet,
    download_file,
)


URLS = {
    "https://files.data.gouv.fr/geo-dvf/latest/csv/2020/full.csv.gz": "full_2020.csv.gz",
    "https://files.data.gouv.fr/geo-dvf/latest/csv/2021/full.csv.gz": "full_2021.csv.gz",
    "https://files.data.gouv.fr/geo-dvf/latest/csv/2022/full.csv.gz": "full_2022.csv.gz",
    "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/full.csv.gz": "full_2023.csv.gz",
    "https://files.data.gouv.fr/geo-dvf/latest/csv/2024/full.csv.gz": "full_2024.csv.gz",
}

RAW_DIR = "data/raw"
PARQUET_DIR = "data/parquet"


def main() -> None:
    for url, filename in URLS.items():
        local_file = download_file(
            url=url,
            dest_folder=RAW_DIR,
            filename=filename,
        )

        convert_csv_gz_to_parquet(
            input_path=local_file,
            output_folder=PARQUET_DIR,
        )


if __name__ == "__main__":
    main()