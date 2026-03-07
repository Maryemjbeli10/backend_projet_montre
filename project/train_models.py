#!/usr/bin/env python3
"""
Script d'initialisation pour entraîner et loguer les modèles dans MLflow
Exécuté au démarrage du conteneur mlflow-init
"""
import sys
import os
import time

# Attendre que MLflow soit prêt
print("⏳ Attente de MLflow server...")
time.sleep(10)

try:
    print("🚀 Démarrage de l'entraînement...")
    
    # Import et exécution du modeling
    print("\n" + "="*60)
    print("04_MODELING - Régression Prix")
    print("="*60)
    from project import _04_modeling
    _04_modeling.compare_models(['random_forest', 'gradient_boosting', 'xgboost'])
    
    # Import et exécution de la classification
    print("\n" + "="*60)
    print("05_CLASSIFICATION - Classification Investissement")
    print("="*60)
    from project import _05_classification
    _05_classification.compare_classifiers(['random_forest', 'gradient_boosting', 'logistic_regression', 'xgboost'])
    
    print("\n✅ Tous les modèles ont été entraînés et logués avec succès!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)