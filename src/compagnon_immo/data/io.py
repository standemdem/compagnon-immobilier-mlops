from pathlib import Path
import pandas as pd


def save_parquet_gzip(df: pd.DataFrame,output_path: str,overwrite: bool = True) -> None:
    """
    Sauvegarde un DataFrame en Parquet compressé GZIP.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à sauvegarder
    output_path : str
        Chemin du fichier .parquet.gz
    overwrite : bool, default=True
        Autorise l'écrasement du fichier existant
    """

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} existe déjà.")

    # Création du dossier si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    try:
        df.to_parquet(
            temporary_path,
            engine="pyarrow",
            compression="gzip",
            index=False,
        )

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"✅ Dataset sauvegardé : {output_path}")
    print(f"   → lignes : {df.shape[0]}")
    print(f"   → colonnes : {df.shape[1]}")
