from pydantic import BaseModel, Field, model_validator
from typing import Literal

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
            and not self.nom_commune.startswith("Paris ")
        ):
            raise ValueError(
                "Le scope 'paris' nécessite un arrondissement parisien(ex: Paris 4e Arrondissement)."
            )

        return self

class PredictionResponse(BaseModel):
    prix_m2: float