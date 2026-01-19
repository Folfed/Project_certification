# 🌽 Prédiction du Rendement du Maïs en Afrique

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description

Application de **Machine Learning** pour prédire le **rendement agricole du maïs** (yield en tonnes/hectare) dans les pays africains. Ce projet couvre l'ensemble du cycle de vie d'un projet Data Science, de l'exploration des données au déploiement en production.

### 🎯 Problématique Métier

L'agriculture est le pilier économique de nombreux pays africains. La prédiction précise des rendements agricoles permet:
- **Aux agriculteurs**: Mieux planifier leurs cultures et ressources
- **Aux gouvernements**: Anticiper les besoins alimentaires et gérer les stocks
- **Aux organisations**: Optimiser la distribution des aides agricoles

### 📊 Source des Données

- **Dataset**: HarvestStat Africa (hvstat_africa_data_v1.0.csv)
- **Taille**: ~17,000 observations de rendements de maïs
- **Couverture**: 54 pays africains
- **Période**: 1996-2023

## 🏗️ Structure du Projet

```
project/
├── 📁 data/                          # Données
│   ├── hvstat_africa_data_v1.0.csv   # Dataset original
│   └── df_clean_maize.csv            # Données nettoyées
│
├── 📁 docs/                           # Documentation
│   └── rapport.pdf                   # Rapport de projet
│
├── 📁 ml_models_pkg/                  # Modèles entraînés
│   ├── final_model.pkl               # Modèle final déployé
│   ├── model_metadata.pkl            # Métadonnées du modèle
│   ├── feature_names.pkl             # Noms des features
│   ├── ridge_regression_model.pkl    # Modèle Ridge (.pkl)
│   ├── ridge_regression_model.joblib # Modèle Ridge (.joblib)
│   ├── random_forest_model.pkl       # Modèle Random Forest
│   ├── gb_model.pkl                  # Modèle Gradient Boosting
│   ├── gb_feature_names.pkl          # Features Gradient Boosting
│   ├── scaler.pkl                    # Scaler (.pkl)
│   ├── scaler.joblib                 # Scaler (.joblib)
│   ├── model_comparison.png          # Graphique comparaison modèles
│   ├── predictions_comparison.png    # Graphique prédictions
│   └── error_distribution.png        # Distribution des erreurs
│
├── 📁 notebooks/                      # Jupyter Notebooks
│   ├── EDA.ipynb                     # Analyse exploratoire
│   ├── linear_regression.ipynb       # Modèle Ridge
│   ├── random_forest.ipynb           # Modèle Random Forest
│   ├── Gradient_Boost.ipynb          # Modèle Gradient Boosting
│   └── model_selection.ipynb         # Comparaison & sélection finale
│
├── 📁 templates/                      # Templates HTML
│   └── index.html                    # Interface web de prédiction
│
├── 📄 app.py                          # API FastAPI
├── 📄 retrain_model.py               # Script de réentraînement
├── 📄 scheduler.py                   # Planificateur automatique
├── 📄 requirements.txt               # Dépendances Python
├── 📄 Dockerfile                     # Image Docker API
├── 📄 Dockerfile.retrainer           # Image Docker Retrainer
├── 📄 docker-compose.yml             # Orchestration Docker
├── 📄 Procfile                       # Configuration Heroku/Render
├── 📄 runtime.txt                    # Version Python pour déploiement
├── 📄 .python-version                # Version Python (Render)
├── 📄 DEPLOYMENT_GUIDE.md            # Guide de déploiement
└── 📄 README.md                      # Ce fichier
```

## 🚀 Installation & Démarrage

### Option 1: Installation Locale

```bash
# 1. Cloner le repository
git clone https://github.com/Folfed/Project_certification.git

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'API
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker (Recommandé)

```bash
# Avec Docker Compose (API + Retrainer)
docker-compose up --build

# Ou juste l'API
docker build -t maize-api .
docker run -p 8000:8000 maize-api
```

## 📡 Utilisation de l'API

### Accès à l'API

Une fois démarrée, l'API est accessible sur:
- **Page d'accueil**: http://localhost:8000
- **Documentation Swagger**: http://localhost:8000/docs
- **Documentation ReDoc**: http://localhost:8000/redoc

### Exemple de Prédiction

```python
import requests

# Prédiction pour une parcelle au Kenya
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "country_code": "KE",
        "season_name": "Main",
        "planting_month": 3,
        "harvest_month": 8,
        "area": 500,
        "production_system": "rainfed"
    }
)

print(response.json())
# {
#     "predicted_yield": 1.85,
#     "predicted_production": 925.0,
#     "confidence": "Élevé",
#     "unit": "tonnes/hectare",
#     "model_used": "Random Forest"
# }
```

### Endpoints Disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil |
| GET | `/health` | État de santé de l'API |
| GET | `/model/info` | Informations sur le modèle |
| POST | `/predict` | Prédiction unique |
| POST | `/predict/batch` | Prédictions par lot |
| GET | `/countries` | Liste des pays supportés |
| GET | `/systems` | Systèmes de production |

## 🔄 Réentraînement Automatique

Le système inclut un mécanisme de réentraînement automatique:

### Exécution Manuelle

```bash
# Réentraîner le modèle
python retrain_model.py --data-path ./data/hvstat_africa_data_v1.0.csv

# Forcer le réentraînement
python retrain_model.py --force
```

### Planification Automatique

```bash
# Démarrer le scheduler (réentraînement toutes les 24h)
python scheduler.py --interval 24

# Avec Docker
docker-compose up retrainer
```

### Configuration Windows Task Scheduler

```powershell
# Créer une tâche planifiée Windows
schtasks /create /tn "MaizeModelRetrain" /tr "python C:\path\to\retrain_model.py --force" /sc daily /st 02:00
```

### Configuration Cron (Linux)

```bash
# Éditer crontab
crontab -e

# Ajouter cette ligne (exécution quotidienne à 2h)
0 2 * * * /usr/bin/python3 /path/to/retrain_model.py --force >> /path/to/logs/cron.log 2>&1
```

## 📊 Performance des Modèles

| Modèle | R² Score | MAE (t/ha) | RMSE (t/ha) |
|--------|----------|------------|-------------|
| Ridge Regression | ~0.35 | ~0.48 | ~0.70 |
| Gradient Boosting | ~0.42 | ~0.43 | ~0.65 |
| **Random Forest** | **0.4425** | **0.4215** | **0.6344** |

> ⭐ Le modèle **Random Forest** a été sélectionné comme modèle final.

## 📈 Variables Utilisées

### Features d'entrée
- `country_code`: Code ISO du pays (54 pays africains)
- `season_name`: Saison de culture (Main, Secondary, etc.)
- `planting_month`: Mois de plantation (1-12)
- `harvest_month`: Mois de récolte (1-12)
- `area`: Surface cultivée (hectares)
- `production_system`: Système de production (irrigated, rainfed, etc.)

### Target
- `yield`: Rendement en tonnes par hectare

## 🧪 Tests

```bash
# Exécuter les tests
pytest tests/ -v

# Test de l'endpoint de prédiction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"country_code":"KE","season_name":"Main","planting_month":3,"harvest_month":8,"area":500,"production_system":"rainfed"}'
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Auteurs

- **Équipe Projet Data Science** - Travail pratique

## 📚 Références

- [HarvestStat Africa Dataset](https://harveststat.org)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Scikit-learn Documentation](https://scikit-learn.org)

## 📄 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

<p align="center">
  🌽 <strong>Maize Yield Prediction API</strong> - Projet Data Science 2026
</p>
