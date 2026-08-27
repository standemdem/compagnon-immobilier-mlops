import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status


load_dotenv()

API_KEY = os.getenv("API_KEY")


def verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> str:
    """
    Vérifie la clé API transmise dans le header x-api-key.
    """

    if API_KEY is None:
        raise RuntimeError(
            "La variable d'environnement API_KEY n'est pas définie."
        )

    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    return x_api_key