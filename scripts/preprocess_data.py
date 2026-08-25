import argparse
from pathlib import Path

import pandas as pd

from compagnon_immo.data.cleaning import clean_apartment_sales
from compagnon_immo.data.io import save_parquet_gzip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nettoyage métier des données DVF pour une année donnée."
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Année DVF à traiter, par exemple 2020.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    year = args.year

    input_path = Path(
        f"data/parquet/full_{year}.parquet"
    )

    output_path = Path(
        f"data/processed/dvf_appartements_vente_{year}.parquet.gz"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset source introuvable : {input_path}"
        )

    print(f"=== Préprocessing DVF {year} ===")

    print("\n=== Chargement du dataset ===")

    df = pd.read_parquet(input_path)

    print(
        f"Dataset source : "
        f"{df.shape[0]} lignes / {df.shape[1]} colonnes"
    )

    print("\n=== Nettoyage métier ===")

    df_clean = clean_apartment_sales(df)

    print(
        f"Dataset nettoyé : "
        f"{df_clean.shape[0]} lignes / "
        f"{df_clean.shape[1]} colonnes"
    )

    print("\n=== Sauvegarde ===")

    save_parquet_gzip(
        df=df_clean,
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()