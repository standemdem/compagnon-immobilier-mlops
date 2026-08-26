# 🏠 Compagnon Immobilier — MLOps

Pipeline MLOps de prédiction du **prix au m² des appartements en France métropolitaine** à partir des données DVF (*Demandes de valeurs foncières*).

Ce dépôt reprend le travail exploratoire du projet **Compagnon Immobilier** pour le transformer en une architecture industrialisable : préparation des données par scripts Python, entraînement reproductible, versionnement des données et modèles avec DVC, exposition du modèle via FastAPI et conteneurisation avec Docker/Nginx.

> **Statut : Work in Progress**
>
> Les briques **data processing**, **entraînement ML**, **DVC**, **FastAPI**, **sécurisation de l'API**, **Docker** et **Nginx** sont fonctionnelles.
>
> **Airflow**, **MLflow**, **Prometheus/Grafana** et la **CI/CD** restent à intégrer.

---

## 📑 Sommaire

- [🎯 Objectifs du projet](#-objectifs-du-projet)
- [📊 Données](#-données)
- [🏗️ Architecture actuelle](#️-architecture-actuelle)
- [🔄 Pipeline de données](#-pipeline-de-données)
- [🧹 Nettoyage métier](#-nettoyage-métier)
- [🧱 Dataset ML consolidé](#-dataset-ml-consolidé)
- [🤖 Modélisation](#-modélisation)
  - [Split temporel](#split-temporel)
  - [Valeurs extrêmes](#valeurs-extrêmes)
  - [Features](#features)
- [🌲 Modèle actuel](#-modèle-actuel)
  - [Performances sur 2024](#performances-sur-2024)
  - [Artefacts](#artefacts)
- [🚀 Installation locale](#-installation-locale)
- [🗃️ DVC — Versionnement des données et modèles](#️-dvc--versionnement-des-données-et-modèles)
  - [Compatibilité des dépendances Google Drive](#compatibilité-des-dépendances-google-drive)
  - [Remote Google Drive](#remote-google-drive)
  - [Récupérer les données](#récupérer-les-données)
  - [Publier une nouvelle version](#publier-une-nouvelle-version)
  - [Workflow équipe](#workflow-équipe)
- [⚡ FastAPI — API de prédiction](#-fastapi--api-de-prédiction)
  - [Lancement local](#lancement-local)
  - [Health check](#health-check)
  - [Informations modèle](#informations-modèle)
  - [Prédiction sécurisée](#prédiction-sécurisée)
- [🐳 Docker et Nginx](#-docker-et-nginx)
  - [Lancer la stack](#lancer-la-stack)
  - [Tester via Nginx](#tester-via-nginx)
- [▶️ Reproduire la chaîne complète](#️-reproduire-la-chaîne-complète)
- [🚧 Roadmap MLOps](#-roadmap-mlops)
- [🗺️ Architecture cible](#️-architecture-cible)
- [🎓 Contexte](#-contexte)

---

# 🎯 Objectifs du projet

L'objectif est de construire une chaîne ML reproductible permettant de :

1. préparer les données DVF ;
2. appliquer automatiquement les règles de nettoyage métier ;
3. construire un dataset destiné au Machine Learning ;
4. entraîner et évaluer un modèle de prédiction du prix au m² ;
5. versionner les données et les modèles ;
6. exposer le modèle via une API REST sécurisée ;
7. conteneuriser les différents composants ;
8. orchestrer le pipeline ;
9. suivre les expérimentations ML ;
10. superviser l'API et l'infrastructure.

Le projet vise donc un **cycle de vie ML reproductible et déployable**, et pas uniquement la performance d'un modèle.

---

# 📊 Données

Le projet utilise les données ouvertes **DVF — Demandes de valeurs foncières**, mises à disposition par l'administration française.

Elles contiennent notamment :

- la date de mutation ;
- la valeur foncière ;
- le type de bien ;
- la surface réelle bâtie ;
- le nombre de pièces ;
- la commune ;
- le département ;
- la latitude ;
- la longitude.

Les millésimes utilisés sont :

```text
2020
2021
2022
2023
2024
```

> ⚠️ Le fichier 2020 disponible dans le projet commence au **1er juillet 2020**. Les volumes annuels ne doivent donc pas être comparés naïvement entre années.

Les fichiers lourds ne sont pas stockés directement dans Git : les données sources et les artefacts du modèle sont versionnés avec **DVC** et stockés sur le remote **Google Drive** du projet.

---

# 🏗️ Architecture actuelle

```text
compagnon-immobilier-mlops/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── security.py
│
├── data/
│   ├── raw/                 # Sources DVF suivies par DVC
│   ├── parquet/             # Intermédiaires reconstruisibles
│   ├── processed/           # Données nettoyées
│   ├── prod/                # Dataset consolidé destiné au ML
│   └── models/              # Modèle + métadonnées suivis par DVC
│
├── nginx/
│   └── nginx.conf
│
├── scripts/
│   ├── prepare_raw_data.py
│   ├── preprocess_data.py
│   ├── build_model_dataset.py
│   └── train_model.py
│
├── src/
│   └── compagnon_immo/
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py
│       │   ├── cleaning.py
│       │   ├── build_dataset.py
│       │   └── io.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── evaluate.py
│           ├── train.py
│           └── transformers.py
│
├── .dvc/
├── .dvcignore
├── data/raw.dvc
├── data/models.dvc
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

La séparation est volontaire :

- `src/compagnon_immo/` contient la **logique Python réutilisable** ;
- `scripts/` contient les **points d'entrée exécutables** ;
- `app/` contient la **couche de serving FastAPI** ;
- `nginx/` contient la configuration du **reverse proxy**.

---

# 🔄 Pipeline de données

```text
DVF .csv.gz
     │
     ▼
prepare_raw_data.py
     │
     ▼
full_YYYY.parquet
     │
     ▼
preprocess_data.py --year YYYY
     │
     ▼
dvf_appartements_vente_YYYY.parquet.gz
     │
     ▼
build_model_dataset.py
     │
     ▼
dvf_appartements_model_base_2020_2024.parquet.gz
     │
     ▼
train_model.py
     │
     ▼
prix_m2_pipeline_2020_2023.joblib
```

---

# 🧹 Nettoyage métier

Le preprocessing applique notamment les règles suivantes :

- conservation des mutations de nature `Vente` ;
- conservation des mutations contenant un appartement ;
- exclusion des mutations mixtes contenant une maison ou un local industriel/commercial ;
- conservation des mutations contenant exactement un appartement ;
- agrégation des informations au niveau de la mutation ;
- création d'indicateurs comme `has_dependance` ;
- conservation d'une ligne par mutation/appartement.

Exemple sur 2020 :

```text
Dataset brut                  : 2 065 003 lignes
Ventes                        : 1 873 828 lignes
Mutations avec appartement    :   557 361 lignes
Après exclusion des mixtes    :   458 429 lignes
Après filtre 1 appartement    :   363 082 lignes
Après agrégation              :   203 167 mutations
```

Le même pipeline est appliqué aux millésimes 2021 à 2024.

---

# 🧱 Dataset ML consolidé

Le script :

```bash
python scripts/build_model_dataset.py
```

assemble les datasets annuels après création de `prix_m2`, validation des coordonnées et filtrage sur la France métropolitaine.

Le dataset de base contient actuellement :

```text
1 686 963 observations
9 colonnes
```

| Année | Observations |
|---:|---:|
| 2020 | 197 394 |
| 2021 | 421 016 |
| 2022 | 424 796 |
| 2023 | 341 114 |
| 2024 | 302 643 |

Colonnes :

```text
date_mutation
annee_mutation
surface_reelle_bati
nombre_pieces_principales
latitude
longitude
has_dependance
nom_commune
prix_m2
```

---

# 🤖 Modélisation

## Split temporel

Le modèle est évalué avec un split temporel :

```text
TRAIN : 2020 → 2023
TEST  : 2024
```

Avant filtrage de la cible :

```text
Train : 1 384 320 observations
Test  :   302 643 observations
```

Cette stratégie reproduit mieux un scénario de production qu'un split aléatoire : le modèle apprend sur le passé et est évalué sur une période future.

## Valeurs extrêmes

La cible est :

```text
prix_m2 = valeur_fonciere / surface_reelle_bati
```

Les bornes 1 % / 99 % sont calculées **uniquement sur le train**, afin d'éviter toute fuite d'information depuis 2024.

```text
q01 =   475,00 €/m²
q99 = 14 327,47 €/m²
```

Après filtrage :

```text
Train : 1 356 636 observations
Test  :   296 657 observations
```

## Features

Entrées du pipeline :

```text
surface_reelle_bati
nombre_pieces_principales
latitude
longitude
has_dependance
nom_commune
```

`CommuneSalesEncoder` apprend sur le train le nombre d'observations par commune et crée :

```text
nb_ventes_commune
```

Une commune inconnue lors d'une prédiction reçoit la médiane des volumes communaux du train.

Les features réellement transmises au modèle sont :

```text
surface_reelle_bati
nombre_pieces_principales
latitude
longitude
has_dependance
nb_ventes_commune
```

Les transformers custom sont définis dans :

```text
compagnon_immo.models.transformers
```

Cette organisation permet de sérialiser et recharger proprement le pipeline avec `joblib` sans dépendance à une classe définie dans un notebook ou dans `__main__`.

---

# 🌲 Modèle actuel

Modèle :

```text
RandomForestRegressor
```

Configuration actuelle :

```text
n_estimators     = 50
max_depth        = 20
min_samples_leaf = 2
random_state     = 42
n_jobs           = 2
```

## Performances sur 2024

| Métrique | Résultat |
|---|---:|
| MAE | 722,43 €/m² |
| RMSE | 1 126,22 €/m² |
| R² | 0,8017 |

Ces résultats constituent le modèle de référence actuel.

## Artefacts

```text
data/models/
├── prix_m2_pipeline_2020_2023.joblib
└── prix_m2_pipeline_2020_2023.metadata.json
```

Le `.joblib` contient :

```text
CommuneSalesEncoder
        ↓
FeatureSelector
        ↓
RandomForestRegressor
```

Le JSON contient notamment :

- années d'entraînement ;
- année de test ;
- cible et features ;
- bornes de filtrage ;
- paramètres du modèle ;
- volumes train/test ;
- MAE, RMSE et R².

---

# 🚀 Installation locale

## 1. Cloner le dépôt

```bash
git clone https://github.com/standemdem/compagnon-immobilier-mlops.git
cd compagnon-immobilier-mlops
```

## 2. Utiliser Python 3.12.2

Avec `pyenv` :

```bash
pyenv install 3.12.2
pyenv local 3.12.2
```

Vérifier :

```bash
python --version
```

## 3. Créer le virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
```

## 4. Installer le projet

```bash
python -m pip install --upgrade pip
pip install -e .
```

Les dépendances principales sont déclarées dans `pyproject.toml`, notamment :

```text
pandas
pyarrow
requests
scikit-learn
pydantic
fastapi
python-dotenv
uvicorn
dvc[gdrive]
```

---

# 🗃️ DVC — Versionnement des données et modèles

DVC est **fonctionnel** dans le projet.

Git versionne le code et les métadonnées DVC. Les fichiers lourds sont stockés dans un remote Google Drive.

```text
GitHub
├── code Python
├── .dvc/config
├── data/raw.dvc
└── data/models.dvc

Google Drive
├── données sources DVF
└── artefacts modèle
```

Les dossiers actuellement suivis par DVC sont :

```text
data/raw/
data/models/
```

Les intermédiaires :

```text
data/parquet/
data/processed/
data/prod/
```

ne sont pas suivis à ce stade, car ils sont reconstruisibles à partir des données sources et du code.

## Compatibilité des dépendances Google Drive

L'intégration DVC/Google Drive repose sur plusieurs dépendances dont certaines versions récentes sont incompatibles entre elles.

La combinaison validée sur Python 3.12.2 pour ce projet est :

```text
DVC            3.67.1
dvc-gdrive     3.0.1
PyDrive2       1.21.3
pyOpenSSL      24.2.1
cryptography   43.0.3
asyncssh       2.21.1
```

Après l'installation du projet :

```bash
pip install -e .
```

forcer les versions compatibles :

```bash
python -m pip install --force-reinstall \
  "PyDrive2==1.21.3" \
  "pyOpenSSL==24.2.1" \
  "cryptography==43.0.3" \
  "asyncssh==2.21.1"
```

Vérifier ensuite l'intégrité des dépendances :

```bash
python -m pip check
```

Résultat attendu :

```text
No broken requirements found.
```

Puis vérifier les versions :

```bash
python -m pip show \
  PyDrive2 \
  pyOpenSSL \
  cryptography \
  asyncssh \
  dvc-gdrive
```

> ⚠️ Éviter de mettre à jour isolément `pyOpenSSL`, `cryptography`, `PyDrive2` ou `asyncssh` sans vérifier les contraintes de l'ensemble de la pile.

## Remote Google Drive

Le remote est configuré dans :

```text
.dvc/config
```

Vérifier sa présence :

```bash
dvc remote list
```

Les identifiants OAuth ne doivent **jamais** être commités.

L'authentification utilise un client OAuth Google Cloud personnalisé.

Les identifiants sont configurés localement :

```bash
dvc remote modify --local gdrive \
  gdrive_client_id "CLIENT_ID"
```

Puis :

```bash
dvc remote modify --local gdrive \
  gdrive_client_secret "CLIENT_SECRET"
```

Ces valeurs sont enregistrées dans :

```text
.dvc/config.local
```

Ce fichier doit rester ignoré par Git.

### Configuration d'un nouveau contributeur

Pour accéder au remote DVC, chaque contributeur doit disposer :

1. d'un accès au repository GitHub ;
2. d'un accès au dossier Google Drive utilisé comme remote ;
3. d'un compte Google déclaré comme **utilisateur test** de l'application OAuth Google Cloud ;
4. de ses propres identifiants OAuth configurés localement.

Les secrets OAuth ne sont donc jamais partagés dans le repository.

## Récupérer les données

Après clonage du repository et configuration de l'authentification :

```bash
dvc pull
```

DVC restaure notamment :

```text
data/raw/
data/models/
```

## Publier une nouvelle version

Après modification d'un dossier suivi :

```bash
dvc add data/raw
```

ou :

```bash
dvc add data/models
```

Envoyer les fichiers vers le remote :

```bash
dvc push
```

Puis versionner les métadonnées DVC avec Git :

```bash
git add data/*.dvc
git commit -m "Update DVC tracked artifacts"
git push
```

## Workflow équipe

Le workflow standard est :

```text
git pull
   ↓
dvc pull
   ↓
travail / génération d'artefacts
   ↓
dvc add
   ↓
dvc push
   ↓
git add / commit / push
```

DVC conserve les différentes versions des fichiers par hash.

Pour restaurer une ancienne version :

```bash
git checkout <ancien_commit>
dvc pull
```

DVC restaure alors les fichiers correspondant à la version référencée par ce commit Git.

---

# ⚡ FastAPI — API de prédiction

La première version de l'API est **fonctionnelle**.

Elle expose :

```text
GET  /
GET  /health
GET  /model/info
POST /predict
GET  /docs
GET  /redoc
```

## Lancement local

Créer un fichier `.env` à la racine :

```env
API_KEY=une_cle_api
```

> ⚠️ `.env` est ignoré par Git et ne doit jamais être commité.

Lancer l'API :

```bash
uvicorn app.main:app --reload
```

Elle est alors disponible sur :

```text
http://127.0.0.1:8000
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

Documentation ReDoc :

```text
http://127.0.0.1:8000/redoc
```

## Health check

```bash
curl http://127.0.0.1:8000/health
```

Réponse :

```json
{
  "status": "ok"
}
```

## Informations modèle

```bash
curl http://127.0.0.1:8000/model/info
```

Cette route expose les métadonnées du modèle courant :

- nom et type du modèle ;
- années d'entraînement ;
- année de test ;
- features ;
- bornes de filtrage ;
- paramètres du Random Forest ;
- volumes train/test ;
- métriques d'évaluation.

## Prédiction sécurisée

La route :

```text
POST /predict
```

est protégée par une API key.

La requête doit contenir le header :

```text
x-api-key
```

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: une_cle_api" \
  -d '{
    "surface_reelle_bati": 62,
    "nombre_pieces_principales": 3,
    "latitude": 46.20,
    "longitude": 5.22,
    "has_dependance": true,
    "nom_commune": "Bourg-en-Bresse"
  }'
```

Réponse type :

```json
{
  "prix_m2": 2032.801978760189
}
```

Une mauvaise clé retourne :

```text
401 Unauthorized
```

Les données d'entrée sont validées par **Pydantic**.

Par exemple, une surface négative, un nombre de pièces inférieur à 1 ou des coordonnées invalides sont rejetés avant l'appel au modèle.

Une requête invalide retourne :

```text
422 Unprocessable Entity
```

---

# 🐳 Docker et Nginx

La conteneurisation de l'API est **fonctionnelle**.

L'architecture actuelle utilise deux conteneurs :

```text
Client
  │
  ▼
localhost:8080
  │
  ▼
┌──────────────────┐
│      Nginx       │
│ compagnon_nginx  │
│       :80        │
└────────┬─────────┘
         │
         │ réseau Docker
         ▼
┌──────────────────┐
│     FastAPI      │
│  compagnon_api   │
│      :8000       │
└────────┬─────────┘
         │
         ▼
 Pipeline sklearn
```

Cette séparation permet de conserver FastAPI comme service applicatif interne tandis que Nginx constitue le point d'entrée du système.

Le port FastAPI `8000` n'est pas directement exposé à la machine hôte.

Le point d'entrée est :

```text
localhost:8080
```

via Nginx.

## Lancer la stack

Créer le fichier `.env` :

```env
API_KEY=une_cle_api
```

Construire et lancer les conteneurs :

```bash
docker compose up --build
```

Vérifier leur état :

```bash
docker compose ps
```

Les conteneurs attendus sont :

```text
compagnon_api
compagnon_nginx
```

## Tester via Nginx

Health check :

```bash
curl http://127.0.0.1:8080/health
```

Prédiction :

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: une_cle_api" \
  -d '{
    "surface_reelle_bati": 62,
    "nombre_pieces_principales": 3,
    "latitude": 46.20,
    "longitude": 5.22,
    "has_dependance": true,
    "nom_commune": "Bourg-en-Bresse"
  }'
```

Arrêter les services :

```bash
docker compose down
```

---

# ▶️ Reproduire la chaîne complète

Voici le workflow permettant de reconstruire le projet sur une nouvelle machine.

## 1. Cloner le projet

```bash
git clone https://github.com/standemdem/compagnon-immobilier-mlops.git
cd compagnon-immobilier-mlops
```

## 2. Préparer Python

```bash
pyenv install 3.12.2
pyenv local 3.12.2

python -m venv .venv
source .venv/bin/activate
```

## 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -e .
```

Appliquer ensuite les versions compatibles DVC/Google Drive documentées dans la section DVC.

## 4. Configurer DVC

Configurer localement les identifiants OAuth Google Drive :

```bash
dvc remote modify --local gdrive \
  gdrive_client_id "CLIENT_ID"

dvc remote modify --local gdrive \
  gdrive_client_secret "CLIENT_SECRET"
```

Puis récupérer les données et modèles :

```bash
dvc pull
```

## 5. Reconstruire les données intermédiaires

```bash
python scripts/prepare_raw_data.py
```

Puis :

```bash
python scripts/preprocess_data.py --year 2020
python scripts/preprocess_data.py --year 2021
python scripts/preprocess_data.py --year 2022
python scripts/preprocess_data.py --year 2023
python scripts/preprocess_data.py --year 2024
```

Construire le dataset ML :

```bash
python scripts/build_model_dataset.py
```

## 6. Entraîner le modèle

```bash
python scripts/train_model.py
```

## 7. Publier le nouveau modèle avec DVC

```bash
dvc add data/models
dvc push
```

Puis :

```bash
git add data/models.dvc
git commit -m "Update trained model"
git push
```

## 8. Lancer l'API conteneurisée

Créer `.env` :

```env
API_KEY=une_cle_api
```

Puis :

```bash
docker compose up --build
```

Tester :

```bash
curl http://127.0.0.1:8080/health
```

---

# 🚧 Roadmap MLOps

| Composant | Statut |
|---|---|
| Ingestion / conversion DVF | ✅ Fonctionnel |
| Nettoyage métier | ✅ Fonctionnel |
| Dataset ML multi-années | ✅ Fonctionnel |
| Split temporel | ✅ Fonctionnel |
| Pipeline sklearn | ✅ Fonctionnel |
| Évaluation ML | ✅ Fonctionnel |
| Sérialisation Joblib | ✅ Fonctionnel |
| Métadonnées modèle | ✅ Fonctionnel |
| DVC | ✅ Fonctionnel |
| Remote Google Drive | ✅ Fonctionnel |
| FastAPI | ✅ Fonctionnel |
| Validation Pydantic | ✅ Fonctionnel |
| API Key | ✅ Fonctionnel |
| Docker | ✅ Fonctionnel |
| Nginx | ✅ Fonctionnel |
| `dvc.yaml` / `dvc repro` | 🚧 À intégrer |
| MLflow | 🚧 À intégrer |
| Airflow | 🚧 À intégrer |
| Prometheus / Grafana | 🚧 À intégrer |
| CI/CD GitHub Actions | 🚧 À intégrer |

---

# 🗺️ Architecture cible

```text
                    ┌──────────────────┐
                    │    Données DVF   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │       DVC        │
                    │  Google Drive    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Airflow      │
                    └────────┬─────────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
       Data processing                 Training ML
             │                               │
             └───────────────┬───────────────┘
                             ▼
                          MLflow
                             │
                             ▼
                       Modèle versionné
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      Nginx       │
                    └────────┬─────────┘
                             │
                             ▼
                         Utilisateur

                    Prometheus / Grafana
                    Monitoring & métriques
```

Cette architecture représente la **cible du projet**.

Les briques suivantes sont déjà opérationnelles :

```text
Data processing
      +
Training ML
      +
DVC / Google Drive
      +
FastAPI
      +
Docker / Nginx
```

Les prochaines briques permettront progressivement de compléter la chaîne :

```text
dvc.yaml / dvc repro
        ↓
MLflow
        ↓
Airflow
        ↓
Monitoring
        ↓
CI/CD
```

---

# 🎓 Contexte

Ce dépôt correspond à l'industrialisation d'un projet de Data Science consacré à l'immobilier français.

Une première phase du projet a permis :

- d'explorer les données DVF ;
- de comprendre leur structure ;
- de définir les règles de nettoyage ;
- de construire un dataset exploitable ;
- de développer et tester un premier modèle.

Ce dépôt vise à transformer cette phase exploratoire en un **projet MLOps structuré, reproductible et déployable**.

La logique de production est progressivement sortie des notebooks pour être organisée dans des modules Python réutilisables, des scripts d'exécution, une API et des services conteneurisés.