from pathlib import Path

import pandas as pd

from compagnon_immo.data.build_dataset import (
    build_base_model_dataset,
    build_paris_model_dataset,
    concatenate_years,
)
from compagnon_immo.data.io import save_parquet_gzip


PROCESSED_DIR = Path("data/processed")
FRANCE_OUTPUT_PATH = Path(
    "data/prod/dvf_appartements_model_base_france_2020_2024.parquet.gz"
)

PARIS_OUTPUT_PATH = Path(
    "data/prod/dvf_appartements_model_base_paris_2020_2024.parquet.gz"
)

YEARS = [
    2020,
    2021,
    2022,
    2023,
    2024,
]


def main() -> None:
    france_datasets = []
    paris_datasets = []

    print("=== Construction du dataset ML multi-années ===")

    for year in YEARS:
        input_path = (
            PROCESSED_DIR
            / f"dvf_appartements_vente_{year}.parquet.gz"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Dataset processed introuvable : {input_path}"
            )

        print(f"\n=== Année {year} ===")
        print(f"Chargement : {input_path}")

        df = pd.read_parquet(input_path)

        print(
            f"Dataset processed : "
            f"{df.shape[0]} lignes / "
            f"{df.shape[1]} colonnes"
        )

        df_france = build_base_model_dataset(df)
        df_paris = build_paris_model_dataset(df)

        print(
            f"Dataset France annuel : "
            f"{df_france.shape[0]} lignes / "
            f"{df_france.shape[1]} colonnes"
        )

        print(
            f"Dataset Paris annuel : "
            f"{df_paris.shape[0]} lignes / "
            f"{df_paris.shape[1]} colonnes"
        )

        france_datasets.append(df_france)
        paris_datasets.append(df_paris)

    print("\n=== Concaténation France ===")

    df_france_all = concatenate_years(france_datasets)

    print(
        f"Dataset France consolidé : "
        f"{df_france_all.shape[0]} lignes / "
        f"{df_france_all.shape[1]} colonnes"
    )

    print("\nRépartition France par année :")
    print(
        df_france_all["annee_mutation"]
        .value_counts()
        .sort_index()
    )

    print("\n=== Concaténation Paris ===")

    df_paris_all = concatenate_years(paris_datasets)

    print(
        f"Dataset Paris consolidé : "
        f"{df_paris_all.shape[0]} lignes / "
        f"{df_paris_all.shape[1]} colonnes"
    )

    print("\nRépartition Paris par année :")
    print(
        df_paris_all["annee_mutation"]
        .value_counts()
        .sort_index()
    )

    print("\n=== Sauvegarde ===")

    save_parquet_gzip(
        df=df_france_all,
        output_path=str(FRANCE_OUTPUT_PATH),
        overwrite=True,
    )

    save_parquet_gzip(
        df=df_paris_all,
        output_path=str(PARIS_OUTPUT_PATH),
        overwrite=True,
    )


if __name__ == "__main__":
    main()