"""
⏰ Scheduler pour le réentraînement automatique
===============================================
Ce module configure un scheduler pour réentraîner le modèle automatiquement.

Options:
1. Schedule - Planification Python native
2. Windows Task Scheduler - Configuration via script
3. Cron (Linux) - Configuration crontab
"""

import schedule
import time
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

def run_retraining():
    """Exécute le script de réentraînement."""
    logger.info("🔄 Démarrage du réentraînement planifié...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "retrain_model.py"), "--force"],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR)
        )
        
        if result.returncode == 0:
            logger.info("✅ Réentraînement terminé avec succès")
            logger.info(result.stdout)
        else:
            logger.error(f"❌ Erreur lors du réentraînement: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ Exception lors du réentraînement: {e}")

def start_scheduler(interval_hours: int = 24):
    """
    Démarre le scheduler pour le réentraînement périodique.
    
    Args:
        interval_hours: Intervalle en heures entre chaque réentraînement
    """
    logger.info(f"📅 Scheduler démarré - Réentraînement toutes les {interval_hours} heures")
    
    # Planifier le réentraînement
    schedule.every(interval_hours).hours.do(run_retraining)
    
    # Optionnel: planifier aussi à une heure fixe (ex: tous les jours à 2h du matin)
    # schedule.every().day.at("02:00").do(run_retraining)
    
    # Premier réentraînement immédiat (optionnel)
    # run_retraining()
    
    # Boucle principale
    while True:
        schedule.run_pending()
        time.sleep(60)  # Vérifie toutes les minutes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scheduler de réentraînement")
    parser.add_argument('--interval', type=int, default=24, help="Intervalle en heures")
    parser.add_argument('--run-now', action='store_true', help="Exécuter immédiatement")
    
    args = parser.parse_args()
    
    if args.run_now:
        run_retraining()
    else:
        start_scheduler(args.interval)
