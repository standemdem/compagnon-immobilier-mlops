# 🏠 Compagnon Immobilier --- MLOps

Pipeline MLOps de prédiction du **prix au m² des appartements en France
métropolitaine** à partir des données DVF (*Demandes de valeurs
foncières*).

Ce dépôt transforme le projet **Compagnon Immobilier** en une
architecture reproductible et industrialisable :

-   préparation des données avec des scripts Python ;
-   pipeline reproductible avec DVC ;
-   entraînement et évaluation de modèles de Machine Learning ;
-   suivi des expérimentations avec MLflow ;
-   exposition des modèles champions via une API FastAPI sécurisée ;
-   conteneurisation avec Docker ;
-   reverse proxy Nginx ;
-   orchestration avec Airflow ;
-   supervision avec Prometheus et Grafana.

> **Statut : architecture MLOps fonctionnelle**
>
> Le projet dispose d'une chaîne MLOps complète intégrant le versionnement
> des données, l'expérimentation, le Model Registry, le serving multi-modèles,
> l'orchestration, la supervision et une CI GitHub Actions.
>
> Deux périmètres de prédiction sont actuellement disponibles :
>
> - **France métropolitaine**
> - **Paris**

------------------------------------------------------------------------

## Sommaire

- [Objectifs](#-objectifs)
- [Données](#-données)
- [Architecture](#️-architecture)
- [Structure du projet](#-structure-du-projet)
- [Pipeline DVC](#-pipeline-dvc)
- [Modélisation](#-modélisation)
- [DVC](#️-dvc)
- [MLflow](#-mlflow)
- [Workflow MLOps](#-workflow-mlops)
- [FastAPI](#-fastapi)
- [Docker et Nginx](#-docker-et-nginx)
- [Airflow](#️-airflow)
- [Prometheus et Grafana](#-prometheus-et-grafana)
- [Services exposés](#-services-exposés)
- [Installation et configuration](#️-installation-et-configuration)
- [Intégration continue](#-intégration-continue)
- [Roadmap](#️-roadmap)

------------------------------------------------------------------------

# 🎯 Objectifs

L'objectif est de construire une chaîne Machine Learning reproductible
permettant de :

1.  récupérer et préparer les données DVF ;
2.  appliquer automatiquement les règles de nettoyage métier ;
3.  construire un dataset destiné au Machine Learning ;
4.  entraîner et évaluer des modèles de prédiction du prix au m² ;
5.  versionner les données et les artefacts du modèle ;
6.  suivre les expérimentations ;
7.  exposer les modèles champions via une API REST sécurisée ;
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

L'architecture sépare le cycle de vie des données, l'entraînement des modèles,
leur mise en production et la supervision de l'API.

```text
                         ┌──────────────────────┐
                         │       Airflow        │
                         │   Orchestration ML   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      DVC runner      │
                         │                      │
                         │ pull / exp run / push│
                         └───────┬──────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
      ┌──────────────────┐             ┌──────────────────────┐
      │ Google Drive DVC │             │        MLflow        │
      │                  │             │ Tracking + Registry  │
      │ données / cache  │             │                      │
      └──────────────────┘             │ France @champion     │
                                       │ Paris  @champion     │
                                       └──────────┬───────────┘
                                                  │
                                                  ▼
                                       ┌──────────────────────┐
                                       │       FastAPI        │
                                       │                      │
                                       │ scope france / paris │
                                       └───────┬───────┬──────┘
                                               │       │
                                      prediction│       │metrics
                                               │       │
                                               ▼       ▼
                                          ┌───────┐ ┌────────────┐
                                          │ Nginx │ │ Prometheus │
                                          └───┬───┘ └─────┬──────┘
                                              │           │
                                              │           ▼
                                              │       ┌─────────┐
                                              │       │ Grafana │
                                              │       └─────────┘
                                              │
                                              ▼
                                            Client
```

### Responsabilités

- **DVC** : versionnement des données, reproductibilité du pipeline et gestion des expérimentations.
- **Google Drive** : stockage distant des données et artefacts suivis par DVC.
- **Airflow** : orchestration des expérimentations DVC.
- **MLflow** : suivi des runs, métriques, artefacts et gestion des versions de modèles.
- **MLflow Model Registry** : stockage des modèles candidats et gestion des alias `champion`.
- **FastAPI** : chargement et exposition des champions France et Paris.
- **Prometheus** : collecte des métriques de l'API.
- **Grafana** : visualisation des métriques.
- **Nginx** : point d'entrée unique vers l'API et les interfaces d'administration.

------------------------------------------------------------------------

# 📁 Structure du projet

Le dépôt est organisé par responsabilité afin de séparer le serving,
le pipeline Machine Learning, l'orchestration et la supervision.

```text
compagnon-immobilier-mlops/
│
├── app/                       # API FastAPI
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── security.py
│
├── src/compagnon_immo/        # Code métier Python
│   ├── data/                  # Préparation et transformation des données
│   └── models/                # Entraînement et composants ML
│
├── scripts/                   # Entrées du pipeline DVC
│   ├── download_data.py
│   ├── prepare_raw_data.py
│   ├── preprocess_data.py
│   ├── build_model_dataset.py
│   └── train_model.py
│
├── tests/                     # Tests automatisés
│
├── airflow/
│   └── dags/                  # DAG d'orchestration
│
├── prometheus/                # Configuration Prometheus
│
├── grafana/
│   ├── provisioning/          # Provisioning automatique
│   └── dashboards/            # Dashboards versionnés
│
├── nginx/
│   ├── nginx.conf             # Reverse proxy
│   ├── control/               # Portail d'administration
│   └── auth/                  # Credentials Basic Auth locaux
│
├── data/
│   ├── raw/                   # Données sources
│   ├── parquet/               # Données intermédiaires
│   ├── processed/             # Données nettoyées
│   ├── prod/                  # Datasets destinés au modèle
│   └── models/                # Artefacts modèles suivis par DVC
│
├── mlflow/                    # Backend et artefacts MLflow locaux
│
├── .github/
│   └── workflows/             # GitHub Actions
│
├── Dockerfile                 # Image FastAPI
├── Dockerfile.dvc-runner      # Image pipeline DVC / ML
├── Dockerfile.airflow         # Image Airflow
├── Dockerfile.test            # Image de tests
├── docker-compose.yml         # Orchestration des conteneurs
│
├── dvc.yaml                   # Définition du pipeline DVC
├── dvc.lock                   # État reproductible du pipeline
├── params.yaml                # Paramètres ML / DVC Experiments
├── pyproject.toml             # Configuration du projet Python
└── README.md
```

## Séparation des responsabilités

```text
app/
└── Serving et exposition des modèles

src/compagnon_immo/
└── Logique métier réutilisable

scripts/
└── Points d'entrée du pipeline DVC

airflow/
└── Orchestration des expérimentations

mlflow/
└── Tracking, Registry et artefacts ML

prometheus/ + grafana/
└── Monitoring et visualisation

nginx/
└── Point d'entrée unique et sécurisation des interfaces

tests/
└── Validation automatisée
```

Les données volumineuses et les modèles reproductibles ne sont pas stockés
directement dans Git : leur cycle de vie est géré par DVC.

------------------------------------------------------------------------

# 🔄 Pipeline DVC

Le pipeline de données et d'entraînement est défini dans `dvc.yaml`.

```text
Données DVF brutes
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
          ┌──────┴──────┐
          ▼             ▼
       France          Paris
          │             │
          ▼             ▼
dvf_appartements_   dvf_appartements_
model_base_france_  model_base_paris_
2020_2024           2020_2024
.parquet.gz          .parquet.gz
          └──────┬──────┘
                 │
                 ▼
        sélection par scope
                 │
                 ▼
               train
                 │
                 ▼
      modèle + métadonnées
                 │
          ┌──────┴──────┐
          ▼             ▼
         DVC           MLflow
                    Tracking + Registry
```

Le scope d'entraînement est défini dans `params.yaml` :

```yaml
training:
  test_year: 2024
  scope: france
```

Les valeurs supportées sont :

```text
france
paris
```

Le pipeline peut être reproduit depuis le conteneur DVC :

```bash
docker exec compagnon_dvc dvc repro
```

Pour reproduire uniquement l'étape d'entraînement :

```bash
docker exec compagnon_dvc dvc repro --single-item train
```

DVC Experiments permet de tester des configurations sans modifier
définitivement les paramètres du workspace :

```bash
docker exec compagnon_dvc dvc exp run \
  -S training.scope=paris
```

Dans le fonctionnement normal du projet, ces expériences peuvent également
être déclenchées depuis Airflow.

------------------------------------------------------------------------

# 🤖 Modélisation

## Split temporel

Les modèles sont évalués avec un split temporel afin de reproduire un cas
d'utilisation réaliste :

```text
TRAIN : 2020 → 2023
TEST  : 2024
```

Les quantiles 1 % et 99 % de la cible `prix_m2` sont calculés uniquement
sur le jeu d'entraînement, puis appliqués au train et au test afin d'éviter
toute fuite d'information.

## Features d'entrée

Les deux scopes utilisent les mêmes données d'entrée :

```text
surface_reelle_bati
nombre_pieces_principales
latitude
longitude
has_dependance
nom_commune
```

Le traitement de `nom_commune` diffère ensuite selon le périmètre.

### Scope France

Pour le modèle France, `CommuneSalesEncoder` transforme `nom_commune` en
`nb_ventes_commune`.

Les features finales sont :

```text
surface_reelle_bati
nombre_pieces_principales
latitude
longitude
has_dependance
nb_ventes_commune
```

Le champion France est un `RandomForestRegressor` configuré avec :

```text
n_estimators     = 50
max_depth        = 20
min_samples_leaf = 2
random_state     = 42
n_jobs           = 2
```

Performances sur le jeu de test 2024 :

| Métrique | Résultat |
|---|---:|
| MAE | 722,43 €/m² |
| RMSE | 1 126,22 €/m² |
| R² | 0,8017 |

### Scope Paris

Un dataset spécifique est construit en filtrant les transactions des
20 arrondissements parisiens.

Pour ce périmètre, `nom_commune` correspond à l'arrondissement, par exemple :

```text
Paris 1er Arrondissement
Paris 10e Arrondissement
Paris 20e Arrondissement
```

Cette variable est encodée avec un `OneHotEncoder`, tandis que les variables
numériques sont transmises au modèle via un `ColumnTransformer`.

Le pipeline Paris permet notamment d'expérimenter avec :

```text
RandomForestRegressor
HistGradientBoostingRegressor
```

Le champion Paris actuel est issu d'une expérimentation Random Forest.

Performances sur le jeu de test 2024 :

| Métrique | Résultat |
|---|---:|
| MAE | 2 050,94 €/m² |
| RMSE | 2 966,13 €/m² |
| R² | 0,1407 |

Ces performances, sensiblement inférieures à celles du modèle France,
montrent que le périmètre parisien constitue un problème de prédiction
plus difficile avec les features actuellement disponibles. Elles justifient
également le suivi séparé des expérimentations et des performances pour
chaque périmètre géographique.

## Artefacts DVC

Chaque entraînement génère localement :

```text
data/models/
├── prix_m2_pipeline_2020_2023.joblib
└── prix_m2_pipeline_2020_2023.metadata.json
```

Ces chemins restent génériques : ils correspondent à l'état courant du
workspace DVC.

Les modèles destinés au serving sont, eux, versionnés séparément dans
le MLflow Model Registry selon leur scope.

------------------------------------------------------------------------

# 🗃️ DVC

DVC assure :

- le versionnement des données ;
- la reproductibilité du pipeline ;
- la synchronisation avec le remote Google Drive ;
- la gestion des expérimentations avec DVC Experiments.

Le pipeline génère également un modèle local et ses métadonnées dans
`data/models/`. Ces artefacts permettent de conserver un pipeline DVC
reproductible, mais le modèle servi par FastAPI est sélectionné depuis
le MLflow Model Registry.

Versions validées avec Python 3.12 :

``` text
dvc             3.67.1
dvc-gdrive      3.0.1
PyDrive2        1.21.3
pyOpenSSL       24.2.1
cryptography    43.0.3
asyncssh        2.21.1
```

La configuration commune du remote DVC est versionnée dans :

```text
.dvc/config
```

Elle définit le remote Google Drive utilisé par le projet.

L'authentification Google Drive est réalisée localement via PyDrive2.
Le cache d'authentification est conservé sur la machine hôte dans :

```text
~/.cache/pydrive2fs/
```

Ce répertoire est monté dans les conteneurs DVC afin de leur permettre
d'accéder au remote sans versionner les informations d'authentification.

Commandes principales :

``` bash
docker exec compagnon_dvc dvc status
docker exec compagnon_dvc dvc status -c
docker exec compagnon_dvc dvc pull
docker exec compagnon_dvc dvc repro
docker exec compagnon_dvc dvc repro --single-item train
docker exec compagnon_dvc dvc push
```

------------------------------------------------------------------------

# 🧪 MLflow

MLflow assure le suivi des expérimentations et la gestion des modèles
destinés au serving.

## Tracking des expérimentations

Chaque entraînement crée un run MLflow contenant notamment :

- le scope d'entraînement (`france` ou `paris`) ;
- le type de modèle ;
- les hyperparamètres ;
- le nombre d'observations train/test ;
- les bornes de filtrage de la cible ;
- les métriques MAE, RMSE et R² ;
- les métadonnées du modèle ;
- le modèle entraîné.

L'expérience utilisée est :

```text
compagnon-immobilier-v2
```

Les entraînements déclenchés depuis Airflow reçoivent un nom de run
permettant d'identifier plus facilement l'expérimentation réalisée.

## Model Registry

Les modèles sont séparés selon leur périmètre géographique dans deux
Registered Models :

```text
compagnon-immobilier-prix-m2-france
compagnon-immobilier-prix-m2-paris
```

Chaque nouvel entraînement peut créer une nouvelle version dans le
Registered Model correspondant à son scope.

## Promotion des modèles

L'enregistrement d'une nouvelle version ne provoque pas automatiquement
sa mise en production.

L'alias MLflow :

```text
champion
```

désigne explicitement la version validée pour chaque scope.

On dispose donc de deux références de production indépendantes :

```text
compagnon-immobilier-prix-m2-france@champion
compagnon-immobilier-prix-m2-paris@champion
```

Cette séparation permet par exemple d'expérimenter avec une nouvelle
configuration sans remplacer automatiquement le modèle actuellement servi.

La promotion d'une version vers l'alias `champion` est volontaire et peut
être réalisée après comparaison de ses métriques avec celles du champion
existant.

## Serving

FastAPI ne charge plus directement le fichier `.joblib` généré dans
`data/models/`.

Au démarrage de l'API, les deux champions sont récupérés depuis le
MLflow Model Registry :

```text
MLflow Registry
      │
      ├── France @champion
      │
      └── Paris  @champion
              │
              ▼
           FastAPI
```

Les modèles sont ensuite conservés en mémoire par l'API afin de servir
les prédictions.

Le premier démarrage peut prendre plusieurs minutes, notamment en raison
du chargement du modèle France.

## Interface

MLflow n'est pas directement exposé sur son port `5000`.

L'interface est accessible via le portail d'administration protégé par
Nginx :

```text
http://localhost:8080/control/mlflow/
```

------------------------------------------------------------------------

# ⚡ FastAPI

FastAPI expose les modèles champions enregistrés dans le MLflow Model Registry.

## Chargement des modèles

Au démarrage, l'API charge les deux modèles correspondant aux alias :

```text
compagnon-immobilier-prix-m2-france@champion
compagnon-immobilier-prix-m2-paris@champion
```

Les modèles sont chargés depuis MLflow puis conservés en mémoire afin d'éviter
un nouveau chargement à chaque prédiction.

L'API ne dépend donc plus d'un montage direct de `data/models/` ou du stockage
local `mlflow/`.

## Endpoints

```text
GET  /health
GET  /model/info
POST /predict
GET  /metrics
```

### Health check

```bash
curl http://localhost:8080/health
```

Résultat attendu :

```json
{
  "status": "ok"
}
```

### Prédiction

L'endpoint `/predict` permet de sélectionner le périmètre du modèle avec
le champ `scope`.

Valeurs supportées :

```text
france
paris
```

Exemple pour le modèle France :

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: VOTRE_CLE_API" \
  -d '{
    "scope": "france",
    "surface_reelle_bati": 50,
    "nombre_pieces_principales": 2,
    "latitude": 48.8566,
    "longitude": 2.3522,
    "has_dependance": 0,
    "nom_commune": "Paris 1er Arrondissement"
  }'
```

Exemple pour le modèle Paris :

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: VOTRE_CLE_API" \
  -d '{
    "scope": "paris",
    "surface_reelle_bati": 50,
    "nombre_pieces_principales": 2,
    "latitude": 48.8608,
    "longitude": 2.3411,
    "has_dependance": 0,
    "nom_commune": "Paris 1er Arrondissement"
  }'
```

Pour le scope Paris, `nom_commune` correspond à l'un des
20 arrondissements présents dans les données DVF.

## Informations sur le modèle

L'endpoint `/model/info` interroge le MLflow Model Registry afin de retourner
les informations relatives au champion du scope demandé.

Le scope France est utilisé par défaut :

```bash
curl http://localhost:8080/model/info
```

Il peut également être indiqué explicitement :

```bash
curl "http://localhost:8080/model/info?scope=france"
```

ou :

```bash
curl "http://localhost:8080/model/info?scope=paris"
```

Les valeurs supportées sont :

```text
france
paris
```

Cet endpoint permet notamment de vérifier quelle version du modèle possède
actuellement l'alias MLflow `champion` pour chaque périmètre.

## Sécurité

Les endpoints protégés utilisent une clé API transmise dans l'en-tête :

```text
x-api-key
```

La clé est définie via la variable d'environnement :

```text
API_KEY
```

Elle est stockée dans le fichier `.env`, qui n'est pas versionné dans Git.

## Métriques Prometheus

FastAPI expose également les métriques nécessaires à la supervision :

```text
compagnon_predictions_total
compagnon_prediction_errors_total
compagnon_prediction_duration_seconds
```

Elles sont collectées automatiquement par Prometheus puis visualisées
dans Grafana.

------------------------------------------------------------------------

# 🐳 Docker et Nginx

L'ensemble de la stack MLOps est conteneurisé avec Docker Compose.

## Services

La stack comprend les services suivants :

```text
api
nginx
dvc-init
dvc
mlflow
airflow
prometheus
grafana
```

Un profil Docker supplémentaire permet d'exécuter les tests :

```text
tests
```

Les images Python sont séparées afin de limiter les dépendances propres
à chaque composant :

- **API** : FastAPI et dépendances nécessaires au serving ;
- **DVC runner** : DVC, accès Google Drive et dépendances du pipeline ML ;
- **Airflow** : Airflow complété par les dépendances nécessaires au pilotage
  du conteneur DVC.

## Point d'entrée unique

Nginx constitue le seul service exposé directement sur l'hôte :

```text
Client
  │
  ▼
localhost:8080
  │
  ▼
Nginx
  │
  ├──► FastAPI
  │
  └──► /control/
          ├── Airflow
          ├── MLflow
          ├── Prometheus
          └── Grafana
```

Les autres services communiquent uniquement via le réseau Docker interne.

## Portail d'administration

Un portail centralise l'accès aux interfaces MLOps :

```text
http://localhost:8080/control/
```

Il donne accès à :

```text
/control/airflow/
/control/mlflow/
/control/prometheus/
/control/grafana/
```

L'ensemble de l'espace `/control/` est protégé par une authentification
HTTP Basic gérée par Nginx.

Airflow et Grafana conservent en complément leur propre mécanisme
d'authentification.

Le fichier contenant les credentials Basic Auth est local :

```text
nginx/auth/.htpasswd
```

et n'est pas versionné dans Git.

> En environnement distant ou en production, cette authentification devrait
> être utilisée derrière HTTPS afin de protéger les credentials pendant
> leur transport.

## Persistance

Les composants nécessitant de conserver un état utilisent des stockages
persistants.

### MLflow

```text
./mlflow:/mlflow
```

permet de conserver notamment la base SQLite MLflow et les artefacts.

### Grafana

```text
grafana_data:/var/lib/grafana
```

conserve l'état de Grafana.

### Prometheus

```text
prometheus_data:/prometheus
```

conserve les séries temporelles collectées par Prometheus.

Les fichiers de provisioning Grafana et la configuration Prometheus restent
par ailleurs versionnés dans le dépôt.

## Initialisation DVC

Le service `dvc-init` est un conteneur one-shot exécuté lors du démarrage
de la stack.

Il effectue :

```text
dvc pull
```

avant le démarrage de l'API.

Une exécution normale se termine avec :

```text
Exited (0)
```

Le service `dvc` reste ensuite disponible comme runner pour les commandes
DVC et les expérimentations orchestrées par Airflow.

## Cold start

La stack complète a été conçue pour pouvoir être redémarrée avec :

```bash
docker compose down
docker compose up -d
```

sans intervention manuelle sur les différents services.

Le chargement initial des champions MLflow par FastAPI peut prendre plusieurs
minutes avant que l'API ne soit disponible.

------------------------------------------------------------------------

# 🌬️ Airflow

Airflow orchestre les expérimentations Machine Learning exécutées par DVC.

Le DAG principal est :

```text
compagnon_immobilier_pipeline
```

Il est déclenché manuellement depuis l'interface Airflow afin de pouvoir
choisir la configuration de l'expérience.

## Enchaînement du DAG

```text
dvc_pull
    │
    ▼
dvc_experiment
    │
    ▼
dvc_push
    │
    ▼
restore_workspace
```

### 1. `dvc_pull`

La première tâche exécute :

```bash
dvc pull
```

dans le conteneur `compagnon_dvc`.

Elle synchronise le workspace avec le remote DVC avant le lancement
de l'expérience.

### 2. `dvc_experiment`

Airflow construit dynamiquement une commande :

```bash
dvc exp run
```

à partir des paramètres choisis dans le formulaire de déclenchement du DAG.

Les paramètres disponibles comprennent notamment :

- le périmètre géographique :
  - `france`
  - `paris`
- le type de modèle :
  - `random_forest`
  - `hist_gradient_boosting`
- le nom de l'expérience ;
- les hyperparamètres propres au modèle sélectionné.

Pour Random Forest, le formulaire permet notamment de configurer :

```text
n_estimators
max_depth
min_samples_leaf
```

Pour HistGradientBoosting :

```text
max_iter
learning_rate
max_depth
min_samples_leaf
```

Le nom de l'expérience DVC est également transmis à l'entraînement via :

```text
MLFLOW_RUN_NAME
```

Le run créé dans MLflow porte ainsi le même nom que l'expérience choisie
dans Airflow, ce qui facilite le suivi entre les différents outils.

### 3. `dvc_push`

Lorsque l'expérience s'est terminée correctement, Airflow exécute :

```bash
dvc push
```

afin de synchroniser les artefacts suivis par DVC avec le remote Google Drive.

### 4. `restore_workspace`

La dernière tâche restaure le workspace de travail après l'expérience.

Elle remet d'abord :

```text
params.yaml
dvc.lock
```

dans leur état enregistré dans le `HEAD` Git courant, puis exécute :

```bash
dvc checkout train --force
```

afin de restaurer les outputs de l'étape `train` correspondant à cet état.

Cette étape évite qu'une expérimentation Airflow laisse le workspace DVC
dans la configuration temporaire utilisée pendant le run.

`restore_workspace` utilise la règle :

```text
all_done
```

afin d'être exécuté même si une tâche précédente échoue.

> La restauration du workspace ne promeut pas le modèle entraîné et ne
> modifie pas l'alias MLflow `champion`. La promotion d'un modèle reste
> une opération distincte dans le MLflow Model Registry.

## Exécution dans le conteneur DVC

Airflow n'exécute pas directement le pipeline ML dans son propre conteneur.

Il communique avec Docker via :

```text
/var/run/docker.sock
```

et exécute les commandes dans :

```text
compagnon_dvc
```

Cette séparation permet de conserver l'environnement DVC et Machine Learning
dans un conteneur dédié.

Le montage du socket Docker convient au contexte local et démonstratif du
projet. Une architecture de production nécessiterait un mécanisme
d'exécution plus fortement isolé.

## Concurrence

Le DAG est configuré avec :

```text
max_active_runs = 1
```

Une seule exécution du DAG peut donc être active à la fois.

Cette contrainte est volontaire : les expériences utilisent le même workspace
DVC et les mêmes outputs locaux. Des entraînements concurrents pourraient
entrer en conflit sur `params.yaml`, `dvc.lock` ou les artefacts générés.

## Interface

Airflow n'est pas directement exposé sur un port de l'hôte.

Il est accessible via Nginx :

```text
http://localhost:8080/control/airflow/
```

L'accès est protégé par l'authentification Basic du portail MLOps ainsi que
par le mécanisme d'authentification propre à Airflow.

------------------------------------------------------------------------

# 📈 Prometheus et Grafana

La supervision de l'API repose sur Prometheus pour la collecte des métriques
et Grafana pour leur visualisation.

```text
FastAPI
   │
   │ /metrics
   ▼
Prometheus
   │
   ▼
Grafana
```

## Prometheus

FastAPI expose un endpoint :

```text
/metrics
```

Prometheus interroge cet endpoint toutes les 15 secondes.

Les principales métriques applicatives sont :

```text
compagnon_predictions_total
compagnon_prediction_errors_total
compagnon_prediction_duration_seconds
```

Elles permettent notamment de suivre :

- le nombre de prédictions réalisées ;
- le rythme des prédictions ;
- les erreurs rencontrées ;
- la durée des prédictions ;
- les distributions de latence.

La configuration Prometheus est versionnée dans :

```text
prometheus/prometheus.yml
```

Les séries temporelles sont stockées dans un volume Docker persistant :

```text
prometheus_data:/prometheus
```

Elles sont ainsi conservées lorsque le conteneur Prometheus est recréé.

## Grafana

Grafana utilise Prometheus comme datasource.

Le dashboard du projet permet notamment de visualiser :

- le nombre total de prédictions ;
- le taux de prédictions ;
- les erreurs ;
- la latence moyenne ;
- la latence p95.

Le provisioning est versionné dans :

```text
grafana/provisioning/
grafana/dashboards/
```

La datasource Prometheus et le dashboard peuvent ainsi être recréés
automatiquement lors du démarrage d'un nouvel environnement Grafana.

L'état propre à Grafana est conservé dans le volume :

```text
grafana_data:/var/lib/grafana
```

## Réseau Docker

Grafana communique directement avec Prometheus sur le réseau Docker interne :

```text
http://prometheus:9090
```

Ce port n'est pas publié sur l'hôte.

De la même manière, le port `3000` de Grafana n'est pas directement exposé.

## Interfaces d'administration

Les interfaces sont accessibles uniquement via Nginx :

```text
Prometheus
http://localhost:8080/control/prometheus/

Grafana
http://localhost:8080/control/grafana/
```

Les deux interfaces sont protégées par l'authentification Basic du portail
d'administration.

Grafana conserve en complément son propre compte administrateur.

## Persistance et redémarrage

Prometheus et Grafana utilisent tous deux des volumes Docker persistants :

```text
prometheus_data
grafana_data
```

Un redémarrage de la stack :

```bash
docker compose down
docker compose up -d
```

ne supprime donc pas leurs données persistantes.

La suppression explicite des volumes, par exemple avec :

```bash
docker compose down -v
```

supprimerait en revanche ces données.

------------------------------------------------------------------------

# ⚙️ Installation et configuration

## Prérequis

Le projet nécessite :

```text
Docker
Docker Compose
Git
```

L'accès au remote DVC Google Drive doit également être configuré sur la
machine utilisée pour exécuter le projet.

## Cloner le dépôt

```bash
git clone https://github.com/standemdem/compagnon-immobilier-mlops.git
cd compagnon-immobilier-mlops
```

## Configuration de l'environnement

Créer le fichier `.env` à partir du modèle :

```bash
cp .env.example .env
```

Puis adapter les variables locales.

Exemple :

```env
API_KEY=change_me

DVC_UID=1000
DVC_GID=1000

AIRFLOW_UID=1000
DOCKER_GID=999

GIT_USER_NAME=Votre Nom
GIT_USER_EMAIL=votre-email@example.com

GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change_me
```

Les identifiants Linux peuvent être récupérés avec :

```bash
id -u
id -g
```

Le groupe associé au socket Docker peut être obtenu avec :

```bash
stat -c '%g' /var/run/docker.sock
```

Le fichier `.env` contient des informations locales et potentiellement
sensibles. Il ne doit pas être versionné dans Git.

## Authentification du portail Nginx

Le portail d'administration `/control/` utilise une authentification HTTP Basic.

Les credentials sont stockés localement dans :

```text
nginx/auth/.htpasswd
```

Ce fichier n'est pas versionné.

Il doit être créé avant le démarrage de la stack.

Exemple avec `htpasswd` :

```bash
mkdir -p nginx/auth
htpasswd -c nginx/auth/.htpasswd admin
```

La commande demande ensuite le mot de passe à utiliser.

## Configuration DVC

Le projet utilise Google Drive comme remote DVC.

Les credentials OAuth sont conservés localement dans le cache utilisé par
PyDrive2 et montés dans les conteneurs DVC :

```text
${HOME}/.cache/pydrive2fs
```

Les données et artefacts suivis par DVC peuvent ensuite être récupérés avec :

```bash
docker compose run --rm dvc-init
```

Lors d'un démarrage normal de la stack, cette opération est automatiquement
effectuée par le service `dvc-init`.

## Démarrage de la stack

Construire et démarrer les services :

```bash
docker compose up -d --build
```

Vérifier leur état :

```bash
docker compose ps -a
```

Le service `dvc-init` doit normalement apparaître avec :

```text
Exited (0)
```

Les autres services doivent être actifs.

Le chargement des modèles champions MLflow par FastAPI peut prendre plusieurs
minutes lors du premier démarrage.

## Accès aux services

Tous les accès passent par Nginx sur :

```text
http://localhost:8080
```

Les principaux points d'entrée sont :

```text
API
http://localhost:8080

Documentation Swagger
http://localhost:8080/docs

Portail MLOps
http://localhost:8080/control/

Airflow
http://localhost:8080/control/airflow/

MLflow
http://localhost:8080/control/mlflow/

Prometheus
http://localhost:8080/control/prometheus/

Grafana
http://localhost:8080/control/grafana/
```

## Arrêt

Pour arrêter les conteneurs :

```bash
docker compose down
```

Cette commande conserve les volumes persistants Grafana et Prometheus.

Pour supprimer également les volumes :

```bash
docker compose down -v
```

Cette deuxième commande supprime les données persistantes associées aux volumes
Docker et doit donc être utilisée avec précaution.

## Secrets et fichiers locaux

Les secrets et fichiers d'authentification ne sont pas destinés à être
versionnés.

Cela concerne notamment :

```text
.env
nginx/auth/.htpasswd
airflow/data/
```

Les fichiers versionnés comme `.env.example` contiennent uniquement des valeurs
d'exemple.

------------------------------------------------------------------------

# 🌐 Services exposés

Nginx constitue le **point d'entrée unique** de la stack.

Seul le port `8080` est publié sur l'hôte.

| Service | Adresse | Protection |
|---|---|---|
| API | `http://localhost:8080` | API Key sur les endpoints protégés |
| Swagger | `http://localhost:8080/docs` | — |
| Portail MLOps | `http://localhost:8080/control/` | Basic Auth |
| Airflow | `http://localhost:8080/control/airflow/` | Basic Auth + authentification Airflow |
| MLflow | `http://localhost:8080/control/mlflow/` | Basic Auth |
| Prometheus | `http://localhost:8080/control/prometheus/` | Basic Auth |
| Grafana | `http://localhost:8080/control/grafana/` | Basic Auth + authentification Grafana |

Les ports internes des différents services ne sont pas publiés directement
sur l'hôte :

```text
FastAPI     8000
Airflow     8080
MLflow      5000
Prometheus  9090
Grafana     3000
```

Les communications entre services utilisent le réseau Docker interne.

Cette organisation évite d'exposer directement les composants
d'administration et centralise les accès via Nginx.

------------------------------------------------------------------------

# 🔁 Workflow MLOps

Le projet sépare le cycle d'expérimentation du modèle utilisé en production.

```text
                     ┌───────────────┐
                     │    Airflow    │
                     └───────┬───────┘
                             │
                             │ paramètres
                             ▼
                     ┌───────────────┐
                     │ DVC Experiment│
                     └───────┬───────┘
                             │
                             ▼
                  Préparation / entraînement
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
                   DVC              MLflow
              données / état     run + métriques
                                      │
                                      ▼
                              MLflow Model Registry
                                      │
                              validation manuelle
                                      │
                                      ▼
                                  @champion
                               ┌──────┴──────┐
                               │             │
                               ▼             ▼
                             France         Paris
                               │             │
                               └──────┬──────┘
                                      │
                                      ▼
                                   FastAPI
                                      │
                                      ▼
                                    Nginx
                                      │
                                      ▼
                                    Client
```

## Expérimentation

Airflow permet de déclencher un entraînement en choisissant notamment :

- le périmètre `france` ou `paris` ;
- le type de modèle ;
- les hyperparamètres associés.

Le DAG pilote le conteneur DVC, qui exécute l'expérience et enregistre les
résultats dans MLflow.

Chaque entraînement produit ainsi un **run MLflow identifiable**, avec ses
paramètres, métriques et son modèle.

## Promotion

L'entraînement d'un nouveau modèle ne remplace pas automatiquement le modèle
servi par l'API.

Les modèles validés sont enregistrés dans deux Registered Models :

```text
compagnon-immobilier-prix-m2-france
compagnon-immobilier-prix-m2-paris
```

L'alias MLflow :

```text
champion
```

désigne, pour chaque périmètre, la version actuellement utilisée en production.

La promotion d'un nouveau champion reste volontaire afin qu'une nouvelle
expérience ne remplace pas automatiquement un modèle déjà validé.

## Serving

Au démarrage, FastAPI charge directement :

```text
compagnon-immobilier-prix-m2-france@champion
compagnon-immobilier-prix-m2-paris@champion
```

depuis le **MLflow Model Registry**.

Le modèle local généré par le pipeline DVC dans `data/models/` reste un artefact
reproductible du pipeline, mais il n'est plus utilisé directement par l'API.

------------------------------------------------------------------------

# 🔄 Intégration continue

Le projet utilise GitHub Actions pour exécuter automatiquement les contrôles
principaux à chaque évolution du code.

Le workflow est défini dans :

```text
.github/workflows/ci.yml
```

## Déclenchement

La CI est exécutée automatiquement :

```text
push sur master
pull request vers master
```

## Pipeline CI

Le workflow s'exécute sur un runner Ubuntu fourni par GitHub Actions.

```text
Checkout du dépôt
        │
        ▼
Création de l'environnement de test
        │
        ▼
Construction de l'image Docker de tests
        │
        ▼
Exécution des tests
        │
        ▼
Construction de l'image Docker API
```

### Tests

La CI construit l'image dédiée aux tests :

```bash
docker compose --profile test build tests
```

puis exécute la suite de tests dans un conteneur isolé :

```bash
docker compose --profile test run --rm tests
```

Les tests couvrent notamment l'API et plusieurs composants du pipeline
Machine Learning.

### Validation de l'image API

Une fois les tests validés, GitHub Actions vérifie également que l'image
Docker de l'API peut être construite :

```bash
docker compose build api
```

Cette étape permet de détecter des problèmes de dépendances ou de
construction de l'image avant utilisation de la nouvelle version du code.

## Gestion des secrets dans la CI

La CI utilise une clé API dédiée aux tests et génère un fichier `.env`
temporaire dans le runner GitHub Actions.

Les credentials utilisés dans l'environnement local de développement ne
sont donc pas nécessaires à l'exécution des tests.

## Périmètre

Le workflow actuel met en œuvre une chaîne de **Continuous Integration (CI)** :

```text
Code
  │
  ▼
Tests automatisés
  │
  ▼
Validation du build Docker
```

Il ne réalise pas de déploiement automatique.

La partie **Continuous Deployment / Delivery (CD)** reste donc distincte
de la CI actuellement mise en place.

------------------------------------------------------------------------

# 🗺️ Roadmap

## Fonctionnalités implémentées

- [x] Pipeline de préparation des données DVF avec DVC
- [x] Versionnement des données et artefacts avec DVC et Google Drive
- [x] Pipeline reproductible avec `dvc.yaml` et `dvc.lock`
- [x] Gestion des paramètres d'entraînement avec `params.yaml`
- [x] Expérimentation avec DVC Experiments
- [x] Entraînement configurable Random Forest / HistGradientBoosting
- [x] Modèles distincts France métropolitaine / Paris
- [x] Tracking des expérimentations avec MLflow
- [x] Versionnement des modèles avec MLflow Model Registry
- [x] Gestion indépendante des champions France et Paris
- [x] Serving des modèles `@champion` avec FastAPI
- [x] Sécurisation de l'API par clé API
- [x] Conteneurisation de la stack avec Docker Compose
- [x] Reverse proxy Nginx et point d'entrée unique
- [x] Portail d'administration protégé par Basic Auth
- [x] Orchestration des expérimentations avec Airflow
- [x] Configuration des expériences depuis l'interface Airflow
- [x] Monitoring de l'API avec Prometheus
- [x] Dashboard de supervision avec Grafana
- [x] Persistance des données Prometheus et Grafana
- [x] Tests automatisés et conteneurisés
- [x] Continuous Integration avec GitHub Actions

## Améliorations possibles

Les évolutions suivantes ne sont pas nécessaires au fonctionnement actuel
du projet mais constituent des pistes d'amélioration pour une utilisation
plus proche d'un environnement de production :

- [ ] Ajouter HTTPS pour un déploiement distant
- [ ] Externaliser le stockage MLflow pour faciliter le déploiement sur plusieurs machines
- [ ] Automatiser éventuellement la promotion d'un modèle vers l'alias `champion`
- [ ] Ajouter une stratégie de Continuous Delivery / Deployment
- [ ] Ajouter des mécanismes avancés de monitoring du modèle et de détection de dérive
- [ ] Permettre l'exécution parallèle d'expériences dans des workspaces isolés

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
