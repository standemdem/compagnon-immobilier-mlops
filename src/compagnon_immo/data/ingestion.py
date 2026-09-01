import requests
import pandas as pd
from pathlib import Path

def download_file(url: str, destination: Path) -> Path:
    """
    Télécharge un fichier depuis une URL vers un chemin local.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"✅ File already exists: {destination}")
        return destination
    
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    print(f"⬇️ Downloaded: {destination}")
    return destination

def convert_csv_gz_to_parquet(input_path:Path, output_path:Path) -> Path:
    """
    Convertit un fichier CSV compressé .csv.gz en Parquet.
    """
   
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Parquet already exists: {output_path}")
        return output_path

    print(f"Reading: {input_path}")

    df = pd.read_csv(
        input_path,
        compression="gzip",
        low_memory=False,
    )

    print(
        f"Loaded {df.shape[0]} rows "
        f"and {df.shape[1]} columns"
    )

    df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
    )

    print(f"Created: {output_path}")

    return output_path
