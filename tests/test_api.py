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
        lambda scope: FakeModel(),
    )

    response = client.post(
        "/predict",
        headers={"x-api-key": API_KEY},
        json={
            "scope": "france",
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

def test_predict_uses_requested_scope(monkeypatch):
    requested_scopes = []

    class FakeModel:
        def predict(self, input_df):
            return [10500.0]

    def fake_get_model(scope):
        requested_scopes.append(scope)
        return FakeModel()

    monkeypatch.setattr(
        "app.main.get_model",
        fake_get_model,
    )

    response = client.post(
        "/predict",
        headers={"x-api-key": API_KEY},
        json={
            "scope": "paris",
            "surface_reelle_bati": 50,
            "nombre_pieces_principales": 2,
            "latitude": 48.8566,
            "longitude": 2.3522,
            "has_dependance": False,
            "nom_commune": "Paris 4e Arrondissement",
        },
    )

    assert response.status_code == 200
    assert requested_scopes == ["paris"]

def test_model_info_returns_requested_scope(monkeypatch):
    def fake_get_model_info(scope):
        return {
            "scope": scope,
            "registered_model": (
                f"compagnon-immobilier-prix-m2-{scope}"
            ),
            "alias": "champion",
            "version": 1,
            "run_id": "fake-run-id",
        }

    monkeypatch.setattr(
        "app.main.get_model_info",
        fake_get_model_info,
    )

    response = client.get(
        "/model/info?scope=paris",
        headers={"x-api-key": API_KEY},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["scope"] == "paris"
    assert (
        body["registered_model"]
        == "compagnon-immobilier-prix-m2-paris"
    )
    assert body["alias"] == "champion"
    assert body["version"] == 1
    assert body["run_id"] == "fake-run-id"

def test_model_info_with_invalid_scope():
    response = client.get(
        "/model/info?scope=toulouse",
        headers={"x-api-key": API_KEY},
    )

    assert response.status_code == 422