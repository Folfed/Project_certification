"""
🌽 API de Prédiction du Rendement du Maïs en Afrique
====================================================
Application FastAPI pour prédire le rendement agricole (yield) du maïs
basé sur les données HarvestStat Africa.

Auteurs: Groupe Projet Data Science
Date: 2026
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION DE L'APPLICATION
# ============================================================================

app = FastAPI(
    title="🌽 Maize Yield Prediction API",
    description="""
    ## API de Prédiction du Rendement du Maïs en Afrique
    
    Cette API permet de prédire le rendement (yield) du maïs en tonnes par hectare 
    basé sur plusieurs facteurs agronomiques.
    
    ### Fonctionnalités
    - **Prédiction unique**: Prédire le rendement pour une parcelle
    - **Prédiction par lot**: Prédire pour plusieurs parcelles simultanément
    - **Informations modèle**: Obtenir les détails du modèle utilisé
    
    ### Variables d'entrée
    - Code pays (ISO 2 lettres)
    - Saison de culture (Main, Secondary, etc.)
    - Mois de plantation et de récolte
    - Surface cultivée (hectares)
    - Système de production
    """,
    version="1.0.0",
    contact={
        "name": "Équipe Data Science",
        "email": "contact@maize-prediction.africa"
    }
)

# Configuration CORS pour permettre les requêtes depuis n'importe quelle origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================

# Chemins des fichiers modèle
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml_models_pkg")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

def load_model_components():
    """Charge le modèle et ses composants."""
    try:
        # Essayer de charger le modèle final
        model_path = os.path.join(MODEL_DIR, "final_model.pkl")
        metadata_path = os.path.join(MODEL_DIR, "model_metadata.pkl")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
        else:
            # Fallback: charger le modèle Gradient Boosting (.pkl ou .joblib)
            gb_path_pkl = os.path.join(MODEL_DIR, "gb_model.pkl")
            gb_path_joblib = os.path.join(MODEL_DIR, "gradient_boosting_model.joblib")
            
            if os.path.exists(gb_path_pkl):
                model = joblib.load(gb_path_pkl)
            elif os.path.exists(gb_path_joblib):
                model = joblib.load(gb_path_joblib)
            else:
                # Dernier fallback: Ridge Regression
                ridge_path = os.path.join(MODEL_DIR, "ridge_regression_model.joblib")
                model = joblib.load(ridge_path)
            
            # Charger les noms de features
            feature_names_paths = [
                os.path.join(MODEL_DIR, "gb_feature_names.pkl"),
                os.path.join(MODEL_DIR, "feature_names.pkl")
            ]
            feature_names = []
            for fpath in feature_names_paths:
                if os.path.exists(fpath):
                    feature_names = joblib.load(fpath)
                    break
            
            metadata = {
                'model_type': 'gradient_boosting',
                'model_name': 'Gradient Boosting Regressor',
                'features': feature_names
            }
        
        # Charger le scaler si nécessaire (pour Ridge)
        scaler_paths = [
            os.path.join(MODEL_DIR, "final_scaler.pkl"),
            os.path.join(MODEL_DIR, "scaler.pkl"),
            os.path.join(MODEL_DIR, "scaler.joblib")
        ]
        scaler = None
        for spath in scaler_paths:
            if os.path.exists(spath):
                scaler = joblib.load(spath)
                break
        
        return model, metadata, scaler
    
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, {}, None

# Chargement initial
MODEL, MODEL_METADATA, SCALER = load_model_components()

# ============================================================================
# MODÈLES DE DONNÉES (Pydantic)
# ============================================================================

class PredictionInput(BaseModel):
    """Schéma d'entrée pour une prédiction."""
    country_code: str = Field(
        ..., 
        description="Code ISO du pays (ex: KE, ZA, NG)",
        example="KE",
        min_length=2,
        max_length=3
    )
    season_name: str = Field(
        ..., 
        description="Nom de la saison de culture",
        example="Main"
    )
    planting_month: int = Field(
        ..., 
        description="Mois de plantation (1-12)",
        example=3,
        ge=1,
        le=12
    )
    harvest_month: int = Field(
        ..., 
        description="Mois de récolte (1-12)",
        example=8,
        ge=1,
        le=12
    )
    area: float = Field(
        ..., 
        description="Surface cultivée en hectares",
        example=500.0,
        gt=0
    )
    production_system: str = Field(
        default="general_unknown",
        description="Système de production (irrigated, rainfed, commercial_mechanized, traditional_small_scale, general_unknown)",
        example="rainfed"
    )

class PredictionOutput(BaseModel):
    """Schéma de sortie pour une prédiction."""
    predicted_yield: float = Field(..., description="Rendement prédit en tonnes/hectare")
    predicted_production: float = Field(..., description="Production estimée en tonnes")
    confidence: str = Field(..., description="Niveau de confiance de la prédiction")
    unit: str = Field(default="tonnes/hectare", description="Unité de mesure")
    model_used: str = Field(..., description="Modèle utilisé pour la prédiction")
    timestamp: str = Field(..., description="Horodatage de la prédiction")

class BatchPredictionInput(BaseModel):
    """Schéma pour prédictions par lot."""
    predictions: List[PredictionInput]

class ModelInfo(BaseModel):
    """Informations sur le modèle."""
    model_name: str
    model_type: str
    r2_score: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    features_count: int
    supported_countries: List[str]
    supported_systems: List[str]
    last_updated: str

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

# Liste des pays supportés (codes ISO)
SUPPORTED_COUNTRIES = [
    'AO', 'BF', 'BI', 'BJ', 'BW', 'CD', 'CF', 'CG', 'CI', 'CM', 'CV', 'DJ',
    'DZ', 'EG', 'ER', 'ET', 'GA', 'GH', 'GM', 'GN', 'GQ', 'GW', 'KE', 'KM',
    'LR', 'LS', 'LY', 'MA', 'MG', 'ML', 'MR', 'MU', 'MW', 'MZ', 'NA', 'NE',
    'NG', 'RW', 'SC', 'SD', 'SL', 'SN', 'SO', 'SS', 'ST', 'SZ', 'TD', 'TG',
    'TN', 'TZ', 'UG', 'ZA', 'ZM', 'ZW'
]

SUPPORTED_SYSTEMS = [
    'irrigated', 'rainfed', 'commercial_mechanized', 
    'traditional_small_scale', 'general_unknown', 'other'
]

SUPPORTED_SEASONS = ['Main', 'Secondary', 'Minor', 'Off-season']

def prepare_features(input_data: PredictionInput) -> pd.DataFrame:
    """Prépare les features pour la prédiction."""
    
    # Créer un DataFrame avec les données d'entrée
    data = {
        'country_code': [input_data.country_code.upper()],
        'season_name': [input_data.season_name],
        'planting_month': [input_data.planting_month],
        'harvest_month': [input_data.harvest_month],
        'area': [input_data.area],
        'system_simplified': [input_data.production_system]
    }
    
    df = pd.DataFrame(data)
    
    # Encodage One-Hot
    df_encoded = pd.get_dummies(
        df, 
        columns=['country_code', 'season_name', 'system_simplified'], 
        drop_first=True
    )
    
    # Aligner avec les features du modèle
    if MODEL_METADATA and 'features' in MODEL_METADATA:
        expected_features = MODEL_METADATA['features']
        
        # Ajouter les colonnes manquantes avec 0
        for col in expected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Réordonner les colonnes
        df_encoded = df_encoded[expected_features]
    
    return df_encoded

def get_confidence_level(area: float, country_code: str) -> str:
    """Détermine le niveau de confiance basé sur les caractéristiques."""
    if country_code.upper() in ['KE', 'ZA', 'NG', 'ET', 'TZ', 'GH', 'ZM', 'MW']:
        if 100 <= area <= 10000:
            return "Élevé"
        elif 50 <= area <= 20000:
            return "Moyen"
    return "Modéré"

# ============================================================================
# ENDPOINTS DE L'API
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil avec l'interface de prédiction."""
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    
    # Fallback si le template n'existe pas
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🌽 Maize Yield Prediction API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a5f2a 0%, #2d8a3e 50%, #f4d03f 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                max-width: 800px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #1a5f2a;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.2em;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #2d8a3e;
            }
            .feature-card h3 {
                color: #1a5f2a;
                margin-bottom: 10px;
            }
            .btn {
                display: inline-block;
                background: #2d8a3e;
                color: white;
                padding: 15px 30px;
                border-radius: 30px;
                text-decoration: none;
                margin: 10px 10px 10px 0;
                transition: transform 0.3s, box-shadow 0.3s;
            }
            .btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            }
            .stats {
                background: #1a5f2a;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌽 Maize Yield Prediction API</h1>
            <p class="subtitle">Prédiction du rendement du maïs en Afrique</p>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>🎯 Prédiction</h3>
                    <p>Prédisez le rendement agricole en tonnes/hectare</p>
                </div>
                <div class="feature-card">
                    <h3>🌍 54 Pays</h3>
                    <p>Couverture de tous les pays africains</p>
                </div>
                <div class="feature-card">
                    <h3>🤖 ML Model</h3>
                    <p>Modèle entraîné sur 16,000+ observations</p>
                </div>
                <div class="feature-card">
                    <h3>⚡ API REST</h3>
                    <p>Interface simple et rapide</p>
                </div>
            </div>
            
            <a href="/docs" class="btn">📚 Documentation Swagger</a>
            <a href="/redoc" class="btn">📖 Documentation ReDoc</a>
            <a href="/health" class="btn">❤️ Health Check</a>
            
            <div class="stats">
                <strong>Projet Data Science - Prédiction des rendements agricoles</strong><br>
                Source: HarvestStat Africa Dataset
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Vérifie l'état de santé de l'API."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Retourne les informations sur le modèle utilisé."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return ModelInfo(
        model_name=MODEL_METADATA.get('model_name', 'Unknown'),
        model_type=MODEL_METADATA.get('model_type', 'unknown'),
        r2_score=MODEL_METADATA.get('r2_score'),
        mae=MODEL_METADATA.get('mae'),
        rmse=MODEL_METADATA.get('rmse'),
        features_count=len(MODEL_METADATA.get('features', [])),
        supported_countries=SUPPORTED_COUNTRIES,
        supported_systems=SUPPORTED_SYSTEMS,
        last_updated=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict_yield(input_data: PredictionInput):
    """
    Prédit le rendement du maïs pour une parcelle donnée.
    
    **Paramètres:**
    - country_code: Code ISO du pays (ex: KE pour Kenya)
    - season_name: Saison de culture (Main, Secondary, etc.)
    - planting_month: Mois de plantation (1-12)
    - harvest_month: Mois de récolte (1-12)
    - area: Surface en hectares
    - production_system: Type de système de production
    
    **Retourne:**
    - predicted_yield: Rendement prédit en tonnes/hectare
    - predicted_production: Production totale estimée
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503, 
            detail="Le modèle n'est pas disponible. Veuillez réessayer plus tard."
        )
    
    try:
        # Préparer les features
        X = prepare_features(input_data)
        
        # Appliquer le scaling si nécessaire (pour Ridge)
        if SCALER is not None and MODEL_METADATA.get('model_type') == 'ridge':
            X = SCALER.transform(X)
        
        # Faire la prédiction
        prediction = MODEL.predict(X)[0]
        
        # S'assurer que la prédiction est dans une plage raisonnable
        prediction = max(0.1, min(prediction, 10.0))
        
        # Calculer la production estimée
        estimated_production = prediction * input_data.area
        
        return PredictionOutput(
            predicted_yield=round(prediction, 4),
            predicted_production=round(estimated_production, 2),
            confidence=get_confidence_level(input_data.area, input_data.country_code),
            unit="tonnes/hectare",
            model_used=MODEL_METADATA.get('model_name', 'ML Model'),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Prédit le rendement pour plusieurs parcelles.
    
    Permet d'envoyer jusqu'à 100 prédictions en une seule requête.
    """
    if len(batch_input.predictions) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 prédictions par requête"
        )
    
    results = []
    for i, input_data in enumerate(batch_input.predictions):
        try:
            result = await predict_yield(input_data)
            results.append({
                "index": i,
                "status": "success",
                "prediction": result.dict()
            })
        except HTTPException as e:
            results.append({
                "index": i,
                "status": "error",
                "error": e.detail
            })
    
    return {
        "total": len(batch_input.predictions),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results
    }

@app.get("/countries")
async def list_countries():
    """Liste tous les pays supportés avec leurs codes ISO."""
    country_names = {
        'AO': 'Angola', 'BF': 'Burkina Faso', 'BI': 'Burundi', 'BJ': 'Bénin',
        'BW': 'Botswana', 'CD': 'RD Congo', 'CF': 'Centrafrique', 'CG': 'Congo',
        'CI': "Côte d'Ivoire", 'CM': 'Cameroun', 'CV': 'Cap-Vert', 'DJ': 'Djibouti',
        'DZ': 'Algérie', 'EG': 'Égypte', 'ER': 'Érythrée', 'ET': 'Éthiopie',
        'GA': 'Gabon', 'GH': 'Ghana', 'GM': 'Gambie', 'GN': 'Guinée',
        'GQ': 'Guinée équatoriale', 'GW': 'Guinée-Bissau', 'KE': 'Kenya',
        'KM': 'Comores', 'LR': 'Liberia', 'LS': 'Lesotho', 'LY': 'Libye',
        'MA': 'Maroc', 'MG': 'Madagascar', 'ML': 'Mali', 'MR': 'Mauritanie',
        'MU': 'Maurice', 'MW': 'Malawi', 'MZ': 'Mozambique', 'NA': 'Namibie',
        'NE': 'Niger', 'NG': 'Nigeria', 'RW': 'Rwanda', 'SC': 'Seychelles',
        'SD': 'Soudan', 'SL': 'Sierra Leone', 'SN': 'Sénégal', 'SO': 'Somalie',
        'SS': 'Soudan du Sud', 'ST': 'São Tomé-et-Príncipe', 'SZ': 'Eswatini',
        'TD': 'Tchad', 'TG': 'Togo', 'TN': 'Tunisie', 'TZ': 'Tanzanie',
        'UG': 'Ouganda', 'ZA': 'Afrique du Sud', 'ZM': 'Zambie', 'ZW': 'Zimbabwe'
    }
    
    return {
        "count": len(SUPPORTED_COUNTRIES),
        "countries": [
            {"code": code, "name": country_names.get(code, code)} 
            for code in sorted(SUPPORTED_COUNTRIES)
        ]
    }

@app.get("/systems")
async def list_production_systems():
    """Liste les systèmes de production supportés."""
    system_descriptions = {
        'irrigated': "Culture irriguée - Utilisation de systèmes d'irrigation",
        'rainfed': "Culture pluviale - Dépend des précipitations naturelles",
        'commercial_mechanized': "Agriculture commerciale mécanisée",
        'traditional_small_scale': "Agriculture traditionnelle à petite échelle",
        'general_unknown': "Système non spécifié ou mixte",
        'other': "Autre système de production"
    }
    
    return {
        "count": len(SUPPORTED_SYSTEMS),
        "systems": [
            {"code": system, "description": system_descriptions.get(system, "")}
            for system in SUPPORTED_SYSTEMS
        ]
    }

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
