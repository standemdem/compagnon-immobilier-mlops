import os

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

API_KEY = os.environ["API_KEY"]


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_without_api_key():
    response = client.post(
        "/predict",
        json={
            "surface_reelle_bati": 62,
            "nombre_pieces_principales": 3,
            "latitude": 46.20,
            "longitude": 5.22,
            "has_dependance": True,
            "nom_commune": "Bourg-en-Bresse",
        },
    )

    assert response.status_code == 401


def test_predict_with_invalid_api_key():
    response = client.post(
        "/predict",
        headers={"x-api-key": "mauvaise-cle"},
        json={
            "surface_reelle_bati": 62,
            "nombre_pieces_principales": 3,
            "latitude": 46.20,
            "longitude": 5.22,
            "has_dependance": True,
            "nom_commune": "Bourg-en-Bresse",
        },
    )

    assert response.status_code == 401

def test_predict_with_invalid_payload():
    response = client.post(
        "/predict",
        headers={"x-api-key": API_KEY},
        json={
            "surface_reelle_bati": -10,
            "nombre_pieces_principales": 3,
            "latitude": 46.20,
            "longitude": 5.22,
            "has_dependance": True,
            "nom_commune": "Bourg-en-Bresse",
        },
    )

    assert response.status_code == 422

def test_predict_with_valid_payload(monkeypatch):
    class FakeModel:
        def predict(self, input_df):
            return [3500.0]

    monkeypatch.setattr(
        "app.main.get_model",
        lambda: FakeModel(),
    )

    response = client.post(
        "/predict",
        headers={"x-api-key": API_KEY},
        json={
            "surface_reelle_bati": 62,
            "nombre_pieces_principales": 3,
            "latitude": 46.20,
            "longitude": 5.22,
            "has_dependance": True,
            "nom_commune": "Bourg-en-Bresse",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prix_m2" in data
    assert data["prix_m2"] == 3500.0