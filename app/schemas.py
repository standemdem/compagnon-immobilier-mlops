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
        if (
            self.scope == "paris"
            and self.nom_commune not in PARIS_ARRONDISSEMENTS
        ):
            raise ValueError(
                "Le scope 'paris' nécessite un arrondissement "
                "parisien valide (ex: Paris 4e Arrondissement)."
            )

        return self

class PredictionResponse(BaseModel):
    prix_m2: float
