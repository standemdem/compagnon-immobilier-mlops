# 🏠 Compagnon Immobilier — MLOps

Pipeline MLOps de prédiction du **prix au m² des appartements en France métropolitaine** à partir des données DVF (*Demandes de valeurs foncières*).

Ce projet reprend le travail exploratoire réalisé autour de **Compagnon Immobilier** afin de le transformer en une architecture industrialisable : traitement des données par scripts Python, versionnement des datasets, entraînement reproductible, API de prédiction, conteneurisation, orchestration et monitoring.

> **Statut : Work in Progress**
>
> Le pipeline de préparation des données et d'entraînement du modèle est fonctionnel.  
> Les briques DVC, FastAPI, Docker, Airflow, monitoring et CI/CD sont en cours d'intégration.

---

## 🎯 Objectifs du projet

L'objectif est de construire une chaîne ML reproductible permettant de :

1. récupérer et préparer les données DVF ;
2. nettoyer les transactions immobilières selon des règles métier ;
3. construire automatiquement un dataset destiné au Machine Learning ;
4. entraîner et évaluer un modèle de prédiction du prix au m² ;
5. versionner les données et les modèles ;
6. exposer le modèle via une API sécurisée ;
7. conteneuriser les différents composants ;
8. orchestrer le pipeline ;
9. superviser le modèle et l'infrastructure.

L'objectif n'est donc pas uniquement d'obtenir un modèle performant, mais de construire un **cycle de vie ML complet et reproductible**.

---

# 📊 Données

Le projet utilise les données ouvertes **DVF — Demandes de valeurs foncières**.

Les données décrivent les mutations immobilières enregistrées en France et contiennent notamment :

- la date de mutation ;
- la valeur foncière ;
- le type de bien ;
- la surface réelle bâtie ;
- le nombre de pièces ;
- la commune ;
- le département ;
- la latitude ;
- la longitude.

Le projet utilise actuellement les millésimes :

```text
2020
2021
2022
2023
2024
```

> ⚠️ Le millésime 2020 disponible dans le projet commence au 1er juillet 2020. Les volumes annuels ne doivent donc pas être comparés naïvement entre années.

---

# 🏗️ Architecture du projet

```text
compagnon-immobilier-mlops/
│
├── data/
│   ├── raw/                 # Données DVF sources (.csv.gz)
│   ├── parquet/             # Conversion intermédiaire en Parquet
│   ├── processed/           # Données nettoyées selon les règles métier
│   ├── prod/                # Dataset consolidé destiné au ML
│   └── models/              # Modèles entraînés et métadonnées
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
├── pyproject.toml
├── .gitignore
└── README.md
```

La séparation entre `scripts/` et `src/` est volontaire :

- `src/compagnon_immo/` contient la **logique Python réutilisable** ;
- `scripts/` contient les **points d'entrée exécutables** qui orchestrent cette logique.

---

# 🔄 Pipeline de données

Le pipeline actuel suit les étapes suivantes :

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
preprocess_data.py
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
RandomForestRegressor
```

---

# 🧹 Nettoyage métier

Les données DVF brutes contiennent plusieurs lignes pour une même mutation et différents types de biens.

Le preprocessing applique notamment les règles suivantes :

- conservation des mutations de nature `Vente` ;
- conservation des mutations contenant un appartement ;
- exclusion des mutations mixtes contenant une maison ou un local industriel/commercial ;
- conservation des mutations avec exactement un appartement ;
- agrégation des différentes lignes d'une mutation ;
- création d'indicateurs comme `has_dependance` ;
- conservation d'une ligne par mutation.

Exemple sur 2020 :

```text
Dataset brut                  : 2 065 003 lignes
Ventes                        : 1 873 828 lignes
Mutations avec appartement    :   557 361 lignes
Après exclusion des mixtes    :   458 429 lignes
Après filtre 1 appartement    :   363 082 lignes
Après agrégation              :   203 167 mutations
```

Le même pipeline est ensuite appliqué aux autres millésimes.

---

# 🧱 Construction du dataset ML

Le script :

```bash
python scripts/build_model_dataset.py
```

charge les datasets annuels nettoyés, crée la cible `prix_m2`, vérifie les coordonnées géographiques, conserve la France métropolitaine et assemble les différents millésimes.

Le dataset consolidé actuel contient :

```text
1 686 963 observations
9 colonnes
```

Répartition :

| Année | Observations |
|---:|---:|
| 2020 | 197 394 |
| 2021 | 421 016 |
| 2022 | 424 796 |
| 2023 | 341 114 |
| 2024 | 302 643 |

Les colonnes conservées sont :

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

Le modèle est évalué avec un **split temporel**, afin de se rapprocher d'un véritable scénario de production.

```text
TRAIN : 2020 → 2023
TEST  : 2024
```

Volumes avant filtrage de la cible :

```text
Train : 1 384 320 observations
Test  :   302 643 observations
```

Cette approche évite d'entraîner et d'évaluer aléatoirement le modèle sur des transactions provenant des mêmes périodes.

---

## Traitement des valeurs extrêmes

La cible est :

```text
prix_m2 = valeur_fonciere / surface_reelle_bati
```

Les bornes sont calculées **uniquement sur le jeu d'entraînement** afin d'éviter une fuite d'information depuis le jeu de test.

Quantiles utilisés :

```text
q01 = 1 %
q99 = 99 %
```

Bornes obtenues sur le train :

```text
475,00 €/m²
14 327,47 €/m²
```

Après filtrage :

```text
Train : 1 356 636 observations
Test  :   296 657 observations
```

Les mêmes bornes apprises sur le train sont appliquées au jeu de test 2024.

---

# 🧠 Features

Les données fournies au pipeline sont :

```text
surface_reelle_bati
nombre_pieces_principales
latitude
longitude
has_dependance
nom_commune
```

Une transformation custom sklearn, `CommuneSalesEncoder`, apprend sur le train le nombre de transactions observées pour chaque commune.

Elle crée :

```text
nb_ventes_commune
```

Une commune inconnue lors de la prédiction reçoit la médiane du nombre de ventes par commune calculée lors de l'entraînement.

Les features réellement transmises au modèle sont donc :

```text
surface_reelle_bati
nombre_pieces_principales
latitude
longitude
has_dependance
nb_ventes_commune
```

Les transformations custom sont définies dans un véritable module Python :

```text
compagnon_immo.models.transformers
```

Cela permet notamment au pipeline sérialisé avec `joblib` d'être correctement rechargé dans un autre processus ou, à terme, par l'API.

---

# 🌲 Modèle actuel

Le premier modèle de référence est un :

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

Cette configuration a notamment été choisie pour permettre l'entraînement du dataset complet sur une machine disposant de ressources limitées.

---

# 📈 Performances actuelles

Évaluation sur les transactions **2024**, jamais utilisées pour entraîner le modèle :

| Métrique | Résultat |
|---|---:|
| MAE | 722,43 €/m² |
| RMSE | 1 126,22 €/m² |
| R² | 0,8017 |

Ces résultats constituent le **modèle de référence actuel** et pourront être comparés à de futurs modèles.

---

# 💾 Artefacts du modèle

Après entraînement :

```text
data/models/
├── prix_m2_pipeline_2020_2023.joblib
└── prix_m2_pipeline_2020_2023.metadata.json
```

Le fichier `.joblib` contient l'intégralité du pipeline sklearn :

```text
CommuneSalesEncoder
        ↓
FeatureSelector
        ↓
RandomForestRegressor
```

Le fichier JSON contient les métadonnées permettant de documenter l'entraînement :

- années d'entraînement ;
- année de test ;
- features ;
- cible ;
- bornes de filtrage ;
- paramètres du modèle ;
- nombre d'observations ;
- MAE ;
- RMSE ;
- R².

---

# 🚀 Installation

## 1. Cloner le dépôt

```bash
git clone <URL_DU_REPOSITORY>
cd compagnon-immobilier-mlops
```

---

## 2. Sélectionner Python

Le projet nécessite :

```text
Python >= 3.12
```

Avec `pyenv`, par exemple :

```bash
pyenv install 3.12.2
pyenv local 3.12.2
```

Vérifier :

```bash
python --version
```

---

## 3. Créer l'environnement virtuel

```bash
python -m venv .venv
```

Activation sous Linux / WSL :

```bash
source .venv/bin/activate
```

---

## 4. Installer le projet

```bash
pip install --upgrade pip
pip install -e .
```

Les dépendances principales sont déclarées dans `pyproject.toml`.

Actuellement :

```text
pandas
pyarrow
requests
scikit-learn
```

---

# ▶️ Reproduire le pipeline

## Étape 1 — Placer les données sources

Placer les fichiers DVF `.csv.gz` dans :

```text
data/raw/
```

Les données ne sont volontairement pas versionnées directement avec Git.

---

## Étape 2 — Conversion en Parquet

```bash
python scripts/prepare_raw_data.py
```

Les fichiers intermédiaires sont créés dans :

```text
data/parquet/
```

sous la forme :

```text
full_2020.parquet
full_2021.parquet
...
```

---

## Étape 3 — Preprocessing annuel

Exemple :

```bash
python scripts/preprocess_data.py --year 2020
```

Puis :

```bash
python scripts/preprocess_data.py --year 2021
python scripts/preprocess_data.py --year 2022
python scripts/preprocess_data.py --year 2023
python scripts/preprocess_data.py --year 2024
```

Les datasets nettoyés sont écrits dans :

```text
data/processed/
```

---

## Étape 4 — Construire le dataset ML

```bash
python scripts/build_model_dataset.py
```

Sortie :

```text
data/prod/dvf_appartements_model_base_2020_2024.parquet.gz
```

---

## Étape 5 — Entraîner le modèle

```bash
python scripts/train_model.py
```

Le script :

1. charge le dataset consolidé ;
2. réalise le split temporel ;
3. calcule les quantiles sur le train ;
4. filtre train et test ;
5. prépare les features ;
6. entraîne le pipeline ;
7. évalue le modèle sur 2024 ;
8. sauvegarde le modèle ;
9. sauvegarde ses métadonnées.

---

# 🔁 Reproductibilité

Le projet cherche à séparer clairement :

```text
code
données
configuration
modèles
métriques
```

Git est utilisé pour versionner le code.

Les datasets et modèles volumineux ne doivent pas être stockés directement dans Git.

L'intégration de **DVC** est prévue afin de permettre à un autre développeur de reproduire le pipeline avec les mêmes versions de données.

---

# 🚧 Work in Progress — Roadmap MLOps

Le pipeline data et l'entraînement constituent actuellement la partie la plus avancée du projet.

Les briques suivantes sont prévues.

## 🗃️ DVC — Data Version Control

Objectifs :

- versionner les datasets ;
- versionner les artefacts ML volumineux ;
- stocker les données sur un remote externe ;
- permettre la récupération des données avec `dvc pull`.

Remote envisagé :

```text
Google Drive / OneDrive
```

Statut :

```text
🚧 À intégrer
```

---

## ⚡ FastAPI

Une API REST exposera le modèle.

Endpoints envisagés :

```text
GET  /health
GET  /model/info
POST /predict
```

`POST /predict` recevra notamment :

```json
{
    "surface_reelle_bati": 62,
    "nombre_pieces_principales": 3,
    "latitude": 46.20,
    "longitude": 5.22,
    "has_dependance": 1,
    "nom_commune": "Bourg-en-Bresse"
}
```

L'API chargera le pipeline `.joblib` et retournera le prix au m² prédit.

Une authentification par **API Key** est prévue.

Statut :

```text
🚧 À intégrer
```

---

## 🐳 Docker

La conteneurisation doit permettre d'exécuter l'application indépendamment de la machine hôte.

Architecture envisagée :

```text
Client
  │
  ▼
Nginx
  │
  ▼
FastAPI
  │
  ▼
Pipeline ML
```

Statut :

```text
🚧 À intégrer
```

---

## 🌬️ Airflow

Apache Airflow sera utilisé pour orchestrer le pipeline.

DAG envisagé :

```text
ingestion
   ↓
conversion parquet
   ↓
preprocessing
   ↓
construction dataset ML
   ↓
entraînement
   ↓
évaluation
   ↓
publication du modèle
```

Objectifs :

- automatiser les traitements ;
- gérer les dépendances entre étapes ;
- tracer les exécutions ;
- faciliter les futurs réentraînements.

Statut :

```text
🚧 À intégrer
```

---

## 📊 Monitoring — Grafana

Une couche de monitoring est prévue autour de Grafana.

Les métriques envisagées concernent notamment :

- disponibilité de l'API ;
- nombre de requêtes ;
- latence ;
- erreurs HTTP ;
- prédictions produites ;
- métriques techniques de l'application.

Selon l'architecture retenue, une collecte de métriques via Prometheus pourra être ajoutée.

Statut :

```text
🚧 À intégrer
```

---

## 🧪 Tracking des expérimentations

Une solution de tracking ML pourra être ajoutée afin de conserver :

- paramètres des modèles ;
- métriques ;
- versions ;
- artefacts ;
- comparaison entre expériences.

**MLflow** est envisagé pour cette partie.

Statut :

```text
🚧 À intégrer
```

---

## ⚙️ CI/CD

Une pipeline CI/CD est également prévue afin d'automatiser notamment :

```text
lint
↓
tests
↓
build Docker
↓
validation
↓
déploiement
```

La solution envisagée est GitHub Actions.

Statut :

```text
🚧 À intégrer
```

---

# 🗺️ Architecture cible

À terme, l'architecture recherchée est :

```text
                    ┌──────────────────┐
                    │    Données DVF   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │       DVC        │
                    │  Remote storage  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Airflow      │
                    └────────┬─────────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
      Data processing                  Training ML
             │                               │
             └───────────────┬───────────────┘
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

             Monitoring / métriques
                       │
                       ▼
              Prometheus / Grafana
```

Cette architecture constitue la **cible du projet** et non son état actuel.

---

# 📌 État d'avancement

| Composant | Statut |
|---|---|
| Ingestion DVF | ✅ Fonctionnel |
| Conversion CSV → Parquet | ✅ Fonctionnel |
| Nettoyage métier | ✅ Fonctionnel |
| Pipeline multi-années | ✅ Fonctionnel |
| Split temporel | ✅ Fonctionnel |
| Pipeline sklearn | ✅ Fonctionnel |
| Évaluation ML | ✅ Fonctionnel |
| Sérialisation Joblib | ✅ Fonctionnel |
| Métadonnées modèle | ✅ Fonctionnel |
| DVC | 🚧 Work in Progress |
| FastAPI | 🚧 Work in Progress |
| Sécurisation API | 🚧 Work in Progress |
| Docker | 🚧 Work in Progress |
| Airflow | 🚧 Work in Progress |
| MLflow | 🚧 Work in Progress |
| Prometheus / Grafana | 🚧 Work in Progress |
| CI/CD | 🚧 Work in Progress |

---

# 🎓 Contexte

Ce dépôt correspond à l'industrialisation d'un projet de Data Science consacré à l'immobilier français.

Une première phase du projet a permis d'explorer les données DVF, de définir les règles de nettoyage et de développer un premier modèle.

Ce nouveau dépôt vise à transformer ce travail en **projet MLOps structuré, reproductible et déployable**, en sortant progressivement la logique des notebooks pour la placer dans des modules et scripts Python dédiés.

---

# 📄 Licence et données

Les données immobilières utilisées proviennent des données ouvertes DVF mises à disposition par l'administration française.

Les données sources ne sont pas incluses directement dans le dépôt Git.