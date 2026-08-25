from pathlib import Path

import pandas as pd

from compagnon_immo.data.build_dataset import (build_base_model_dataset,concatenate_years,)
from compagnon_immo.data.io import save_parquet_gzip


PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path(
    "data/prod/dvf_appartements_model_base_2020_2024.parquet.gz"
)

YEARS = [
    2020,
    2021,
    2022,
    2023,
    2024,
]


def main() -> None:
    datasets = []

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

        df_model = build_base_model_dataset(df)

        print(
            f"Dataset ML annuel : "
            f"{df_model.shape[0]} lignes / "
            f"{df_model.shape[1]} colonnes"
        )

        datasets.append(df_model)

    print("\n=== Concaténation ===")

    df_all = concatenate_years(datasets)

    print(
        f"Dataset consolidé : "
        f"{df_all.shape[0]} lignes / "
        f"{df_all.shape[1]} colonnes"
    )

    print("\nRépartition par année :")
    print(
        df_all["annee_mutation"]
        .value_counts()
        .sort_index()
    )

    print("\n=== Sauvegarde ===")

    save_parquet_gzip(
        df=df_all,
        output_path=str(OUTPUT_PATH),
        overwrite=True,
    )


if __name__ == "__main__":
    main()