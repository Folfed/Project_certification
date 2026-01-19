"""
🔄 Script de Réentraînement Automatique du Modèle
=================================================
Ce script permet de réentraîner le modèle de prédiction du rendement du maïs
avec de nouvelles données ou de manière planifiée.

Fonctionnalités:
- Chargement et nettoyage des données
- Entraînement de plusieurs modèles (Ridge, Random Forest, Gradient Boosting)
- Sélection automatique du meilleur modèle
- Sauvegarde du modèle et des métadonnées
- Logging des performances

Usage:
    python retrain_model.py --data-path ./data/hvstat_africa_data_v1.0.csv
    python retrain_model.py --force  # Force le réentraînement même si les données n'ont pas changé
"""

import os
import sys
import argparse
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================

# Répertoires
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ml_models_pkg"
LOGS_DIR = BASE_DIR / "logs"

# Créer les répertoires s'ils n'existent pas
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'retrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FONCTIONS DE PRÉTRAITEMENT
# ============================================================================

def calculate_data_hash(df: pd.DataFrame) -> str:
    """Calcule un hash des données pour détecter les changements."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """Charge et nettoie les données."""
    logger.info(f"Chargement des données depuis: {data_path}")
    
    df = pd.read_csv(data_path)
    original_size = len(df)
    
    # Filtrer le maïs
    df_maize = df[df['product'].str.contains('maize|corn|maïs', case=False, na=False)].copy()
    logger.info(f"Filtrage maïs: {len(df_maize)} observations")
    
    # Nettoyage des outliers
    mask = (
        df_maize['area'].between(0.1, 50000) &
        df_maize['production'].between(1, 50000) &
        df_maize['yield'].between(0.1, 8)
    )
    df_clean = df_maize[mask].copy()
    df_clean = df_clean.dropna(subset=['yield', 'area', 'production'])
    
    # Standardisation du système de production
    df_clean['system_simplified'] = 'other'
    sys_lower = df_clean['crop_production_system'].str.lower()
    
    df_clean.loc[sys_lower.str.contains('irrigated|water|dam|riverine', na=False), 'system_simplified'] = 'irrigated'
    df_clean.loc[sys_lower.str.contains('rainfed|dieri|recessional', na=False), 'system_simplified'] = 'rainfed'
    df_clean.loc[sys_lower.str.contains('commercial|mechanized|large_scale|lscf', na=False), 'system_simplified'] = 'commercial_mechanized'
    df_clean.loc[sys_lower.str.contains('traditional|communal|small|pastoral|sscf|a1|a2', na=False), 'system_simplified'] = 'traditional_small_scale'
    df_clean.loc[sys_lower.str.contains('all|none|or \\(ps\\)', na=False), 'system_simplified'] = 'general_unknown'
    
    logger.info(f"Données nettoyées: {len(df_clean)} observations (supprimé {original_size - len(df_clean)})")
    
    return df_clean

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prépare les features et la target."""
    features_cols = ['country_code', 'season_name', 'planting_month', 
                     'harvest_month', 'area', 'system_simplified']
    
    X = df[features_cols]
    y = df['yield']
    
    # Encodage One-Hot
    X_encoded = pd.get_dummies(X, columns=['country_code', 'season_name', 'system_simplified'], drop_first=True)
    
    return X_encoded, y

# ============================================================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================================

def train_ridge_regression(X_train, X_test, y_train, y_test, scaler):
    """Entraîne un modèle Ridge Regression."""
    logger.info("Entraînement Ridge Regression...")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    logger.info(f"Ridge - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")
    
    return best_model, metrics, scaler

def train_random_forest(X_train, X_test, y_train, y_test):
    """Entraîne un modèle Random Forest."""
    logger.info("Entraînement Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 15, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    logger.info(f"Random Forest - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")
    
    return best_model, metrics

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Entraîne un modèle Gradient Boosting."""
    logger.info("Entraînement Gradient Boosting...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    logger.info(f"Gradient Boosting - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")
    
    return best_model, metrics

# ============================================================================
# FONCTION PRINCIPALE DE RÉENTRAÎNEMENT
# ============================================================================

def retrain_model(data_path: str, force: bool = False) -> dict:
    """
    Fonction principale de réentraînement.
    
    Args:
        data_path: Chemin vers le fichier de données
        force: Force le réentraînement même si les données n'ont pas changé
    
    Returns:
        dict: Résultats du réentraînement
    """
    logger.info("=" * 60)
    logger.info("🔄 DÉMARRAGE DU RÉENTRAÎNEMENT")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    # 1. Charger et nettoyer les données
    df_clean = load_and_clean_data(data_path)
    
    # 2. Vérifier si les données ont changé
    current_hash = calculate_data_hash(df_clean)
    hash_file = MODEL_DIR / "data_hash.txt"
    
    if not force and hash_file.exists():
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        
        if stored_hash == current_hash:
            logger.info("✅ Les données n'ont pas changé. Pas besoin de réentraîner.")
            logger.info("   Utilisez --force pour forcer le réentraînement.")
            return {"status": "skipped", "reason": "data_unchanged"}
    
    # 3. Préparer les features
    X, y = prepare_features(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # 4. Entraîner les modèles
    scaler = StandardScaler()
    
    models_results = {}
    
    # Ridge Regression
    ridge_model, ridge_metrics, fitted_scaler = train_ridge_regression(
        X_train, X_test, y_train, y_test, scaler
    )
    models_results['ridge'] = {
        'model': ridge_model,
        'metrics': ridge_metrics,
        'scaler': fitted_scaler,
        'name': 'Ridge Regression'
    }
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    models_results['random_forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'scaler': None,
        'name': 'Random Forest'
    }
    
    # Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting(X_train, X_test, y_train, y_test)
    models_results['gradient_boosting'] = {
        'model': gb_model,
        'metrics': gb_metrics,
        'scaler': None,
        'name': 'Gradient Boosting'
    }
    
    # 5. Sélectionner le meilleur modèle
    best_model_key = max(models_results.keys(), 
                         key=lambda k: models_results[k]['metrics']['r2'])
    best_result = models_results[best_model_key]
    
    logger.info("=" * 60)
    logger.info(f"🏆 MEILLEUR MODÈLE: {best_result['name']}")
    logger.info(f"   R²:   {best_result['metrics']['r2']:.4f}")
    logger.info(f"   MAE:  {best_result['metrics']['mae']:.4f}")
    logger.info(f"   RMSE: {best_result['metrics']['rmse']:.4f}")
    logger.info("=" * 60)
    
    # 6. Sauvegarder le modèle final
    final_model_path = MODEL_DIR / "final_model.pkl"
    joblib.dump(best_result['model'], final_model_path)
    logger.info(f"✅ Modèle sauvegardé: {final_model_path}")
    
    # Sauvegarder le scaler si nécessaire
    if best_result['scaler'] is not None:
        scaler_path = MODEL_DIR / "final_scaler.pkl"
        joblib.dump(best_result['scaler'], scaler_path)
        logger.info(f"✅ Scaler sauvegardé: {scaler_path}")
    
    # 7. Sauvegarder les métadonnées
    metadata = {
        'model_type': best_model_key,
        'model_name': best_result['name'],
        'r2_score': best_result['metrics']['r2'],
        'mae': best_result['metrics']['mae'],
        'rmse': best_result['metrics']['rmse'],
        'features': X.columns.tolist(),
        'target': 'yield',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'training_date': datetime.now().isoformat(),
        'data_hash': current_hash
    }
    
    metadata_path = MODEL_DIR / "model_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    logger.info(f"✅ Métadonnées sauvegardées: {metadata_path}")
    
    # 8. Sauvegarder le hash des données
    with open(hash_file, 'w') as f:
        f.write(current_hash)
    
    # 9. Sauvegarder tous les modèles individuels
    for key, result in models_results.items():
        model_path = MODEL_DIR / f"{key}_model.pkl"
        joblib.dump(result['model'], model_path)
    
    # Sauvegarder les noms des features
    joblib.dump(X.columns.tolist(), MODEL_DIR / "feature_names.pkl")
    
    # Calculer le temps d'exécution
    duration = datetime.now() - start_time
    
    logger.info("=" * 60)
    logger.info("✅ RÉENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    logger.info(f"   Durée: {duration}")
    logger.info("=" * 60)
    
    return {
        "status": "success",
        "best_model": best_result['name'],
        "metrics": best_result['metrics'],
        "training_samples": len(X_train),
        "duration_seconds": duration.total_seconds(),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="🔄 Script de réentraînement du modèle de prédiction du rendement du maïs"
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=str(DATA_DIR / "hvstat_africa_data_v1.0.csv"),
        help="Chemin vers le fichier de données CSV"
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force le réentraînement même si les données n'ont pas changé"
    )
    
    args = parser.parse_args()
    
    try:
        result = retrain_model(args.data_path, args.force)
        
        if result['status'] == 'success':
            print("\n✅ Réentraînement terminé avec succès!")
            print(f"   Meilleur modèle: {result['best_model']}")
            print(f"   R² Score: {result['metrics']['r2']:.4f}")
            sys.exit(0)
        else:
            print(f"\n⏭️ Réentraînement ignoré: {result.get('reason', 'unknown')}")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du réentraînement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
