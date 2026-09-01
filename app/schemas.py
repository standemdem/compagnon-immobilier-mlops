import json
from pathlib import Path

from shapely.geometry import Point, shape
from pydantic import BaseModel, Field, model_validator
from typing import Literal

PARIS_ARRONDISSEMENTS = {
    "Paris 1er Arrondissement",
    "Paris 2e Arrondissement",
    "Paris 3e Arrondissement",
    "Paris 4e Arrondissement",
    "Paris 5e Arrondissement",
    "Paris 6e Arrondissement",
    "Paris 7e Arrondissement",
    "Paris 8e Arrondissement",
    "Paris 9e Arrondissement",
    "Paris 10e Arrondissement",
    "Paris 11e Arrondissement",
    "Paris 12e Arrondissement",
    "Paris 13e Arrondissement",
    "Paris 14e Arrondissement",
    "Paris 15e Arrondissement",
    "Paris 16e Arrondissement",
    "Paris 17e Arrondissement",
    "Paris 18e Arrondissement",
    "Paris 19e Arrondissement",
    "Paris 20e Arrondissement",
}
GEOJSON_PATH = Path(__file__).parent / "data" / "arrondissements.geojson"

with GEOJSON_PATH.open("r", encoding="utf-8") as file:
    PARIS_GEOJSON = json.load(file)


def find_paris_arrondissement(
    latitude: float,
    longitude: float,
) -> str | None:
    point = Point(longitude, latitude)

    for feature in PARIS_GEOJSON["features"]:
        polygon = shape(feature["geometry"])

        if polygon.covers(point):
            number = int(feature["properties"]["c_ar"])

            if number == 1:
                return "Paris 1er Arrondissement"

            return f"Paris {number}e Arrondissement"

    return None

class PredictionRequest(BaseModel):
    scope: Literal["france", "paris"] = Field(
    description="Périmètre du modèle utilisé pour la prédiction.",
    )
    
    surface_reelle_bati: float = Field(
    gt=0,
    description="Surface réelle bâtie en m².",
    )

    nombre_pieces_principales: int = Field(
        ge=1,
        description="Nombre de pièces principales.",
    )

    latitude: float = Field(
        ge=41.0,
        le=51.5,
        description="Latitude du bien.",
    )

    longitude: float = Field(
        ge=-5.5,
        le=10.0,
        description="Longitude du bien.",
    )

    has_dependance: bool = Field(
        description="Présence d'au moins une dépendance.",
    )

    nom_commune: str = Field(
        min_length=1,
        description="Nom de la commune.",
    )

    @model_validator(mode="after")
    def validate_scope_location(self):
        if self.scope != "paris":
            return self

        arrondissement = find_paris_arrondissement(
            self.latitude,
            self.longitude,
        )

        if arrondissement is None:
            raise ValueError(
                "Les coordonnées doivent être situées dans Paris."
            )

        if self.nom_commune != arrondissement:
            raise ValueError(
                "L'arrondissement ne correspond pas aux coordonnées fournies."
            )

        return self

class PredictionResponse(BaseModel):
    prix_m2: float
