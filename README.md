# 🏠 Compagnon Immobilier --- MLOps

Pipeline MLOps de prédiction du **prix au m² des appartements en France
métropolitaine** à partir des données DVF (*Demandes de valeurs
foncières*).

Ce dépôt transforme le projet **Compagnon Immobilier** en une
architecture reproductible et industrialisable :

-   préparation des données avec des scripts Python ;
-   pipeline reproductible avec DVC ;
-   entraînement et évaluation d'un modèle de Machine Learning ;
-   suivi des expérimentations avec MLflow ;
-   exposition du modèle via une API FastAPI sécurisée ;
-   conteneurisation avec Docker ;
-   reverse proxy Nginx ;
-   orchestration avec Airflow ;
-   supervision avec Prometheus et Grafana.

> **Statut : architecture MLOps fonctionnelle**
>
> Les briques principales sont intégrées et opérationnelles. La
> prochaine étape importante concerne la mise en place de la **CI/CD
> avec GitHub Actions**.

------------------------------------------------------------------------

## 📑 Sommaire

-   [Objectifs](#-objectifs)
-   [Données](#-données)
-   [Architecture](#️-architecture)
-   [Structure du projet](#-structure-du-projet)
-   [Pipeline DVC](#-pipeline-dvc)
-   [Modélisation](#-modélisation)
-   [DVC](#️-dvc)
-   [MLflow](#-mlflow)
-   [FastAPI](#-fastapi)
-   [Docker et Nginx](#-docker-et-nginx)
-   [Airflow](#️-airflow)
-   [Prometheus et Grafana](#-prometheus-et-grafana)
-   [Installation](#-installation)
-   [Services exposés](#-services-exposés)
-   [Roadmap](#-roadmap)

------------------------------------------------------------------------

# 🎯 Objectifs

L'objectif est de construire une chaîne Machine Learning reproductible
permettant de :

1.  récupérer et préparer les données DVF ;
2.  appliquer automatiquement les règles de nettoyage métier ;
3.  construire un dataset destiné au Machine Learning ;
4.  entraîner et évaluer un modèle de prédiction du prix au m² ;
5.  versionner les données et les artefacts du modèle ;
6.  suivre les expérimentations ;
7.  exposer le modèle via une API REST sécurisée ;
8.  conteneuriser les différents composants ;
9.  orchestrer le pipeline ;
10. superviser l'API.

------------------------------------------------------------------------

# 📊 Données

Le projet utilise les données ouvertes **DVF --- Demandes de valeurs
foncières**.

Les millésimes actuellement utilisés sont :

``` text
2020
2021
2022
2023
2024
```

> ⚠️ Le fichier 2020 utilisé dans le projet commence au **1er juillet
> 2020**.

Les fichiers volumineux ne sont pas stockés directement dans Git. Ils
sont versionnés avec **DVC** et stockés sur un remote **Google Drive**.

------------------------------------------------------------------------

# 🏗️ Architecture

``` text
Google Drive DVC
       │
    dvc pull
       ▼
   dvc-init
       │
       ├──────────────► FastAPI ──► Nginx
       │                   │
       │                /metrics
       │                   ▼
       │              Prometheus ──► Grafana
       │
       └──────────────► DVC runner ──► MLflow
                            ▲
                            │
                         Airflow
```

------------------------------------------------------------------------

# 📁 Structure du projet

``` text
compagnon-immobilier-mlops/
├── app/                    # API FastAPI
├── airflow/
│   ├── dags/
│   └── logs/
├── data/
│   ├── raw/
│   ├── parquet/
│   ├── processed/
│   ├── prod/
│   └── models/
├── grafana/
│   ├── dashboards/
│   └── provisioning/
├── nginx/
├── prometheus/
├── scripts/
├── src/
│   └── compagnon_immo/
│       ├── data/
│       └── models/
├── .dvc/
├── .env.example
├── data/raw.dvc
├── dvc.yaml
├── dvc.lock
├── Dockerfile
├── Dockerfile.airflow
├── Dockerfile.dvc-runner
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

------------------------------------------------------------------------

# 🔄 Pipeline DVC

Le pipeline est défini dans `dvc.yaml` :

``` text
DVF .csv.gz
    │
    ▼
prepare
    │
    ▼
full_YYYY.parquet
    │
    ├── preprocess_2020
    ├── preprocess_2021
    ├── preprocess_2022
    ├── preprocess_2023
    └── preprocess_2024
           │
           ▼
dvf_appartements_vente_YYYY.parquet.gz
           │
           ▼
build_dataset
           │
           ▼
dvf_appartements_model_base_2020_2024.parquet.gz
           │
           ▼
train
           │
           ├── prix_m2_pipeline_2020_2023.joblib
           └── prix_m2_pipeline_2020_2023.metadata.json
```

Reproduire le pipeline :

``` bash
docker exec compagnon_dvc dvc repro
```

------------------------------------------------------------------------

# 🤖 Modélisation

Le modèle est évalué avec un split temporel :

``` text
TRAIN : 2020 → 2023
TEST  : 2024
```

Les quantiles 1 % et 99 % de la cible sont calculés uniquement sur le
train afin d'éviter toute fuite d'information.

Features d'entrée :

``` text
surface_reelle_bati
nombre_pieces_principales
latitude
longitude
has_dependance
nom_commune
```

`CommuneSalesEncoder` crée la feature `nb_ventes_commune`.

Le modèle actuel est un `RandomForestRegressor` :

``` text
n_estimators     = 50
max_depth        = 20
min_samples_leaf = 2
random_state     = 42
n_jobs           = 2
```

## Performances sur 2024

  Métrique          Résultat
  ---------- ---------------
  MAE            722,43 €/m²
  RMSE         1 126,22 €/m²
  R²                  0,8017

Artefacts :

``` text
data/models/
├── prix_m2_pipeline_2020_2023.joblib
└── prix_m2_pipeline_2020_2023.metadata.json
```

------------------------------------------------------------------------

# 🗃️ DVC

DVC assure le versionnement des données et artefacts ML, la
reproductibilité du pipeline et la synchronisation avec Google Drive.

Versions validées avec Python 3.12 :

``` text
dvc             3.67.1
dvc-gdrive      3.0.1
PyDrive2        1.21.3
pyOpenSSL       24.2.1
cryptography    43.0.3
asyncssh        2.21.1
```

La configuration commune est dans `.dvc/config`.

Les credentials OAuth restent locaux dans :

``` text
.dvc/config.local
```

Le cache OAuth PyDrive2 est conservé dans :

``` text
~/.cache/pydrive2fs/
```

Commandes principales :

``` bash
docker exec compagnon_dvc dvc status
docker exec compagnon_dvc dvc status -c
docker exec compagnon_dvc dvc pull
docker exec compagnon_dvc dvc repro
docker exec compagnon_dvc dvc push
```

------------------------------------------------------------------------

# 🧪 MLflow

MLflow enregistre notamment :

-   paramètres du modèle ;
-   MAE, RMSE et R² ;
-   volumes train/test ;
-   bornes de filtrage ;
-   modèle et métadonnées.

Expérience :

``` text
compagnon-immobilier
```

Model Registry :

``` text
compagnon-immobilier-prix-m2
```

Alias de la dernière version entraînée :

``` text
champion
```

L'API utilise toutefois le modèle versionné avec DVC dans
`data/models/`.

Interface : `http://localhost:5000`

------------------------------------------------------------------------

# ⚡ FastAPI

Le modèle est chargé depuis :

``` text
data/models/prix_m2_pipeline_2020_2023.joblib
```

`data/models` est monté en lecture seule dans le conteneur API.

Endpoints :

``` text
GET  /health
GET  /model/info
POST /predict
GET  /metrics
```

`/predict` est protégé par une clé API via l'en-tête `x-api-key`.

Exemple :

``` bash
curl -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -H "x-api-key: VOTRE_CLE_API"   -d '{
    "surface_reelle_bati": 62,
    "nombre_pieces_principales": 3,
    "latitude": 46.20,
    "longitude": 5.22,
    "has_dependance": true,
    "nom_commune": "Bourg-en-Bresse"
  }'
```

Métriques Prometheus spécifiques :

``` text
compagnon_predictions_total
compagnon_prediction_errors_total
compagnon_prediction_duration_seconds
```

------------------------------------------------------------------------

# 🐳 Docker et Nginx

Les images sont séparées pour limiter les dépendances inutiles :

-   API : extra Python `.[api]` ;
-   DVC runner : extra `.[pipeline]` ;
-   Airflow : image officielle complétée par le SDK Docker.

Nginx sert de reverse proxy :

``` text
Client → localhost:8080 → Nginx → api:8000
```

------------------------------------------------------------------------

# 🌬️ Airflow

Le DAG `compagnon_immobilier_pipeline` orchestre :

``` text
dvc pull → dvc repro → dvc push
```

Les commandes sont exécutées dans `compagnon_dvc` via le socket Docker.

Interface : `http://localhost:8081`

> Le montage de `/var/run/docker.sock` convient au contexte
> local/démonstration du projet, mais nécessiterait un durcissement en
> production.

------------------------------------------------------------------------

# 📈 Prometheus et Grafana

Prometheus collecte les métriques de FastAPI toutes les 15 secondes.

Interface Prometheus : `http://localhost:9090`

Grafana utilise Prometheus comme datasource et affiche notamment :

-   total des prédictions ;
-   taux de prédictions ;
-   erreurs ;
-   latence moyenne ;
-   latence p95.

Interface Grafana : `http://localhost:3000`

Le provisioning est versionné dans :

``` text
grafana/provisioning/
grafana/dashboards/
```

La datasource Prometheus et le dashboard sont recréés automatiquement
sur un Grafana vierge.

------------------------------------------------------------------------

# 🚀 Installation

## Prérequis

-   Git
-   Docker
-   Docker Compose
-   accès au remote Google Drive DVC
-   Linux ou WSL2 recommandé

## 1. Cloner

``` bash
git clone https://github.com/standemdem/compagnon-immobilier-mlops.git
cd compagnon-immobilier-mlops
```

## 2. Configurer l'environnement

``` bash
cp .env.example .env
```

Le fichier contient notamment :

``` env
API_KEY=change_me
AIRFLOW_UID=1000
DOCKER_GID=999
```

Récupérer l'UID :

``` bash
id -u
```

Récupérer le GID du socket Docker :

``` bash
stat -c '%g' /var/run/docker.sock
```

Reporter les valeurs dans `.env`.

## 3. Configurer DVC / Google Drive

La configuration du remote est déjà versionnée dans `.dvc/config`.

Chaque utilisateur configure ses propres informations OAuth dans
`.dvc/config.local`.

Les secrets OAuth ne doivent jamais être ajoutés à Git.

## 4. Démarrer

``` bash
docker compose up -d --build
```

Au démarrage :

``` text
docker compose up
        │
        ▼
     dvc-init
        │
        ▼
     dvc pull
        │
        ▼
données + modèle disponibles
        │
        ▼
       API
```

`dvc-init` termine ensuite normalement en `Exited (0)`.

## 5. Vérifier

``` bash
docker compose ps -a
```

Puis :

``` bash
curl http://localhost:8080/health
```

Résultat attendu :

``` json
{
  "status": "ok"
}
```

## Arrêt

``` bash
docker compose down
```

------------------------------------------------------------------------

# 🔐 Configuration et secrets

Ne jamais versionner :

``` text
.env
.dvc/config.local
tokens OAuth Google
mlflow/
airflow/logs/
```

`.env.example` est versionné et sert de modèle de configuration.

------------------------------------------------------------------------

# 🌐 Services exposés

  Service       Adresse                        Rôle
  ------------- ------------------------------ ----------------------------
  API / Nginx   `http://localhost:8080`        API de prédiction
  Swagger       `http://localhost:8080/docs`   Documentation API
  Airflow       `http://localhost:8081`        Orchestration
  MLflow        `http://localhost:5000`        Tracking et Model Registry
  Prometheus    `http://localhost:9090`        Collecte des métriques
  Grafana       `http://localhost:3000`        Monitoring

------------------------------------------------------------------------

# 🔁 Workflow MLOps

``` text
GitHub
  │
  ▼
Code + métadonnées DVC
  │
  ▼
Google Drive DVC
  │
  ▼
dvc pull
  │
  ▼
Données
  │
  ▼
dvc repro
  │
  ▼
Préparation → entraînement
                 │
          ┌──────┴──────┐
          ▼             ▼
         DVC          MLflow
          │
          ▼
     data/models
          │
          ▼
       FastAPI
          │
          ▼
        Nginx

FastAPI /metrics → Prometheus → Grafana
```

------------------------------------------------------------------------

# 🚧 Roadmap

## Fonctionnel

-   [x] Scripts Python modulaires
-   [x] Préparation DVF
-   [x] Pipeline ML
-   [x] Split temporel
-   [x] Random Forest
-   [x] Sérialisation du pipeline
-   [x] DVC + Google Drive
-   [x] Pipeline `dvc.yaml`
-   [x] FastAPI
-   [x] Validation Pydantic
-   [x] API Key
-   [x] Docker
-   [x] Nginx
-   [x] MLflow Tracking
-   [x] MLflow Model Registry
-   [x] Airflow
-   [x] Prometheus
-   [x] Grafana
-   [x] Provisioning Grafana
-   [x] Initialisation automatique DVC avec `dvc-init`

## À faire

-   [ ] Tests automatisés
-   [ ] CI/CD avec GitHub Actions
-   [ ] Vérification automatisée des images Docker
-   [ ] Amélioration de la traçabilité dataset / version DVC dans MLflow
-   [ ] Durcissement des secrets et credentials pour la production

------------------------------------------------------------------------

# 🎓 Contexte

Ce dépôt constitue la version **MLOps / industrialisée** du projet
Compagnon Immobilier.

Le projet couvre désormais :

``` text
données
→ préparation
→ entraînement
→ versionnement
→ expérimentation
→ orchestration
→ serving
→ monitoring
```

L'objectif est de disposer d'un projet reproductible, compréhensible et
démontrable dans un contexte MLOps.
