"""
API FastAPI - Système d'Investissement Montres de Luxe
VERSION STRUCTURÉE PAR ONGLETS - Correspondance exacte avec le Frontend
Endpoints: /general, /materials, /seller, /analyze (tout combiné)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Literal
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from enum import Enum

# Imports XAI
try:
    import shap
    SHAP_OK = True
except:
    SHAP_OK = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.preprocessing import LabelEncoder
    LIME_OK = True
except:
    LIME_OK = False

from setup import PROCESSED_DATA_DIR, CLEANED_CSV

# ============================================================
# CONFIGURATION
# ============================================================

class InvestmentThreshold:
    ROI_EXCELLENT = 25.0
    ROI_GOOD = 15.0
    ROI_ACCEPTABLE = 5.0

app = FastAPI(
    title="API Investissement Montres de Luxe - Par Onglets",
    description="API organisée selon les 3 onglets du frontend: Général, Matériaux, Vendeur",
    version="3.0-structured"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ENUMS - Listes de valeurs pour les dropdowns
# ============================================================

class BrandEnum(str, Enum):
    ROLEX = "Rolex"
    PATEK_PHILIPPE = "Patek Philippe"
    AUDEMARS_PIGUET = "Audemars Piguet"
    OMEGA = "Omega"
    CARTIER = "Cartier"
    TUDOR = "Tudor"
    TAG_HEUER = "TAG Heuer"
    BREITLING = "Breitling"
    IWC = "IWC"
    JAEGER_LECOULTRE = "Jaeger-LeCoultre"
    PANERAI = "Panerai"
    HUBLOT = "Hublot"
    ZENITH = "Zenith"
    SEIKO = "Seiko"
    CASIO = "Casio"
    CITIZEN = "Citizen"
    TISSOT = "Tissot"
    LONGINES = "Longines"
    RADO = "Rado"
    HAMILTON = "Hamilton"
    CYMA = "Cyma"
    CONCORD = "Concord"
    SQUALE = "Squale"
    GUCCI = "Gucci"
    CUERVO_Y_SOBRINOS = "Cuervo y Sobrinos"
    NORQAIN = "Norqain"
    DAVOSA = "Davosa"
    POLJOT = "Poljot"
    REVUE_THOMMEN = "Revue Thommen"
    CWC = "CWC"
    RICHARD_MILLE = "Richard Mille"
    GREUBEL_FORSEY = "Greubel Forsey"
    GRAF = "Graf"
    LAURENT_FERRIER = "Laurent Ferrier"
    DOLCE_GABBANA = "Dolce & Gabbana"
    MERCURE = "Mercure"


class ConditionEnum(str, Enum):
    NEW = "New"
    UNWORN = "Unworn"
    LIKE_NEW = "Like new & unworn"
    USED_MINT = "Used (Mint)"
    USED_VERY_GOOD = "Used (Very good)"
    USED_GOOD = "Used (Good)"
    USED_FAIR = "Used (Fair)"
    USED_POOR = "Used (Poor)"

class MovementEnum(str, Enum):
    AUTOMATIC = "Automatic"
    MANUAL = "Manual winding"
    QUARTZ = "Quartz"

class GenderEnum(str, Enum):
    MEN = "Men's watch"
    WOMEN = "Women's watch"
    UNISEX = "Unisex"

class ShapeEnum(str, Enum):
    ROUND = "Round"
    SQUARE = "Square"
    RECTANGULAR = "Rectangular"
    TONNEAU = "Tonneau"
    CUSHION = "Cushion"
    OCTAGONAL = "Octagonal"

class MaterialEnum(str, Enum):
    STEEL = "Steel"
    GOLD = "Gold"
    ROSE_GOLD = "Rose gold"
    WHITE_GOLD = "White gold"
    PLATINUM = "Platinum"
    TITANIUM = "Titanium"
    CERAMIC = "Ceramic"
    CARBON = "Carbon"
    YELLOW_GOLD = "Yellow gold"
    PLASTIC = "Plastic"
    RED_GOLD = "Red gold"
    BRONZE = "Bronze"
    ALUMINUM = "Aluminum"
    SAPPIRE_CRYSTAL = "Sapphire crystal"

class CrystalEnum(str, Enum):
    SAPPHIRE = "Sapphire crystal"
    MINERAL = "Mineral crystal"
    PLEXIGLASS = "Plexiglass"
    GLASS = "Glass"
    PLASTIC = "Plastic"

class ColorEnum(str, Enum):
    BLACK = "Black"
    SILVER = "Silver"
    WHITE = "White"
    BLUE = "Blue"
    GREEN = "Green"
    BROWN = "Brown"
    CHAMPAGNE = "Champagne"
    GREY = "Grey"
    STEEL = "Steel"
    ORANGE = "Orange"
    PINK = "Pink"
    PURPLE = "Purple"
    BRONZE = "Bronze"
    RED = "Red"

class AvailabilityEnum(str, Enum):
    IN_STOCK = "Item is in stock"
    PARTNER = "Item is in stock at a partner boutique"
    SERVICING = "Item is being serviced"
    HOLD = "Item is on hold"

class DeliveryEnum(str, Enum):
    BOX_PAPERS = "Original box, original papers"
    BOX_ONLY = "Original box, no original papers"
    PAPERS_ONLY = "Original papers, no original box"
    NOTHING = "No original box, no original papers"

# ============================================================
# MODÈLES PAR ONGLET
# ============================================================

class GeneralTab(BaseModel):
    """Onglet GÉNÉRAL - 8 champs"""
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    prix_achat: float = Field(
        default=10000,
        alias="Prix d'achat ($)",
        description="Prix d'achat en dollars",
        ge=1000,
        le=10000000
    )
    marque: BrandEnum = Field(
        default=BrandEnum.ROLEX,
        alias="Marque"
    )
    etat: ConditionEnum = Field(
        default=ConditionEnum.LIKE_NEW,
        alias="État"
    )
    mouvement: MovementEnum = Field(
        default=MovementEnum.AUTOMATIC,
        alias="Mouvement"
    )
    genre: GenderEnum = Field(
        default=GenderEnum.MEN,
        alias="Genre"
    )
    annee_production: int = Field(
        default=2020,
        alias="Année de production",
        ge=1900,
        le=2030
    )
    forme: ShapeEnum = Field(
        default=ShapeEnum.ROUND,
        alias="Forme"
    )
    surface_cadran: float = Field(
        default=40.0,
        alias="Surface du cadran (mm²)",
        gt=0
    )
    contenu_livraison: DeliveryEnum = Field(
        default=DeliveryEnum.BOX_PAPERS,
        alias="Contenu de la livraison"
    )

    horizon_annees: int = Field(default=3, ge=1, le=10)

class MaterialsTab(BaseModel):
    """Onglet MATÉRIAUX - 6 champs"""
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    materiau_boitier: MaterialEnum = Field(
        default=MaterialEnum.STEEL,
        alias="Matériau du boîtier"
    )
    materiau_bracelet: MaterialEnum = Field(
        default=MaterialEnum.STEEL,
        alias="Matériau du bracelet"
    )
    verre: CrystalEnum = Field(
        default=CrystalEnum.SAPPHIRE,
        alias="Verre"
    )
    cadran: ColorEnum = Field(
        default=ColorEnum.BLACK,
        alias="Cadran"
    )
    couleur_bracelet: ColorEnum = Field(
        default=ColorEnum.SILVER,
        alias="Couleur du bracelet"
    )
    disponibilite: AvailabilityEnum = Field(
        default=AvailabilityEnum.IN_STOCK,
        alias="Disponibilité"
    )

class SellerTab(BaseModel):
    """Onglet VENDEUR - 6 champs (3 numériques + 3 toggles)"""
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    montres_vendues: int = Field(
        default=50,
        alias="Montres vendues par le vendeur",
        ge=0
    )
    annonces_actives: int = Field(
        default=10,
        alias="Annonces actives du vendeur",
        ge=0
    )
    avis_vendeur: int = Field(
        default=100,
        alias="Avis du vendeur",
        ge=0
    )
    expedition_rapide: int = Field(
        default=1,
        alias="Expédition rapide",
        description="Le vendeur expédie rapidement (toggle)",
        ge=0,
        le=1
    )
    vendeur_confiance: int = Field(
        default=1,
        alias="Vendeur de confiance",
        description="Le vendeur est vérifié (toggle)",
        ge=0,
        le=1
    )
    ponctualite: int = Field(
        default=1,
        alias="Ponctualité",
        description="Le vendeur est ponctuel (toggle)",
        ge=0,
        le=1
    )

class CompleteWatchInput(BaseModel):
    """Tous les onglets combinés"""
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    # Général
    prix_achat: float = Field(default=10000, alias="Prix d'achat ($)", gt=0 , ge=1000)
    marque: BrandEnum = Field(default=BrandEnum.ROLEX, alias="Marque")
    etat: ConditionEnum = Field(default=ConditionEnum.LIKE_NEW, alias="État")
    mouvement: MovementEnum = Field(default=MovementEnum.AUTOMATIC, alias="Mouvement")
    genre: GenderEnum = Field(default=GenderEnum.MEN, alias="Genre")
    annee_production: int = Field(default=2020, alias="Année de production", ge=1900, le=2030)
    forme: ShapeEnum = Field(default=ShapeEnum.ROUND, alias="Forme")
    surface_cadran: float = Field(default=40.0, alias="Surface du cadran (mm²)", gt=0)
    contenu_livraison: DeliveryEnum = Field(default=DeliveryEnum.BOX_PAPERS, alias="Contenu de la livraison")

    # Matériaux
    materiau_boitier: MaterialEnum = Field(default=MaterialEnum.STEEL, alias="Matériau du boîtier")
    materiau_bracelet: MaterialEnum = Field(default=MaterialEnum.STEEL, alias="Matériau du bracelet")
    verre: CrystalEnum = Field(default=CrystalEnum.SAPPHIRE, alias="Verre")
    cadran: ColorEnum = Field(default=ColorEnum.BLACK, alias="Cadran")
    couleur_bracelet: ColorEnum = Field(default=ColorEnum.SILVER, alias="Couleur du bracelet")
    disponibilite: AvailabilityEnum = Field(default=AvailabilityEnum.IN_STOCK, alias="Disponibilité")

    # Vendeur
    montres_vendues: int = Field(default=50, alias="Montres vendues par le vendeur", ge=0)
    annonces_actives: int = Field(default=10, alias="Annonces actives du vendeur", ge=0)
    avis_vendeur: int = Field(default=100, alias="Avis du vendeur", ge=0)
    expedition_rapide: int = Field(default=1, alias="Expédition rapide", ge=0, le=1)
    vendeur_confiance: int = Field(default=1, alias="Vendeur de confiance", ge=0, le=1)
    ponctualite: int = Field(default=1, alias="Ponctualité", ge=0, le=1)

    horizon_annees: int = Field(default=3, ge=1, le=10)

class InvestmentResult(BaseModel):
    """Résultat d'évaluation"""
    prix_achat: float
    prix_futur_estime: float
    plus_value: float
    roi_percent: float
    roi_annualise: float
    horizon_annees: int
    recommandation: str
    confiance: str
    risque: str
    evaluation_simple: Literal["Bon", "Moyen", "Risqué"]
    details: Dict[str, Any]

class TabValidationResponse(BaseModel):
    """Réponse de validation d'un onglet"""
    onglet: str
    valide: bool
    erreurs: List[str]
    donnees_recues: Dict[str, Any]

# ============================================================
# CHARGEMENT MODÈLES
# ============================================================

print("📦 Chargement des modèles...")

try:
    price_model = joblib.load(PROCESSED_DATA_DIR / "best_price_model.pkl")
    print("✅ Modèle prix chargé")
except Exception as e:
    price_model = None
    print(f"❌ Erreur modèle prix: {e}")

try:
    clf_model = joblib.load(PROCESSED_DATA_DIR / "best_classifier.pkl")
    print("✅ Classifieur chargé")
except Exception as e:
    clf_model = None
    print(f"❌ Erreur classifieur: {e}")

try:
    with open(CLEANED_CSV, 'r', encoding='utf-8') as f:
        sep = ';' if ';' in f.readline() else ','
    data_sample = pd.read_csv(CLEANED_CSV, sep=sep, nrows=500)
    print(f"✅ Données chargées: {len(data_sample)} lignes")
except Exception as e:
    data_sample = None
    print(f"❌ Erreur données: {e}")

MODELS_READY = price_model is not None and clf_model is not None

FEATURES = [
    "Year of production", "age", "Face Area",
    "Watches Sold by the Seller", "Active listing of the seller",
    "Seller Reviews", "seller_reputation_score", "scope_score",
    "Fast Shipper", "Trusted Seller", "Punctuality",
    "is_modern", "seller_consistent", "price_anomaly_low", "price_anomaly_high",
    "current_price_estimate",
    "Brand", "Movement", "Case material", "Bracelet material",
    "Condition", "Scope of delivery", "Gender", "Availability",
    "Shape", "Crystal", "Dial", "Bracelet color"
]

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def combine_tabs(general: GeneralTab, materials: MaterialsTab, seller: SellerTab) -> CompleteWatchInput:
    """Combine les 3 onglets en un seul objet"""
    return CompleteWatchInput(
        prix_achat=general.prix_achat,
        marque=general.marque,
        etat=general.etat,
        mouvement=general.mouvement,
        genre=general.genre,
        annee_production=general.annee_production,
        forme=general.forme,
        surface_cadran=general.surface_cadran,
        contenu_livraison=general.contenu_livraison,
        materiau_boitier=materials.materiau_boitier,
        materiau_bracelet=materials.materiau_bracelet,
        verre=materials.verre,
        cadran=materials.cadran,
        couleur_bracelet=materials.couleur_bracelet,
        disponibilite=materials.disponibilite,
        montres_vendues=seller.montres_vendues,
        annonces_actives=seller.annonces_actives,
        avis_vendeur=seller.avis_vendeur,
        expedition_rapide=seller.expedition_rapide,
        vendeur_confiance=seller.vendeur_confiance,
        ponctualite=seller.ponctualite,
        horizon_annees=general.horizon_annees
    )

def prepare_data(watch: CompleteWatchInput) -> pd.DataFrame:
    """Prépare les features pour le modèle"""
    current_year = 2026
    age = current_year - watch.annee_production

    # Mapping des valeurs
    data = {
        "Year of production": watch.annee_production,
        "age": age,
        "Face Area": watch.surface_cadran,
        "Watches Sold by the Seller": watch.montres_vendues,
        "Active listing of the seller": watch.annonces_actives,
        "Seller Reviews": watch.avis_vendeur,
        "Fast Shipper": watch.expedition_rapide,
        "Trusted Seller": watch.vendeur_confiance,
        "Punctuality": watch.ponctualite,
        "is_modern": 1 if watch.annee_production >= 2000 else 0,
        "seller_reputation_score": np.mean([
            watch.expedition_rapide, 
            watch.vendeur_confiance, 
            watch.ponctualite
        ]),
        "scope_score": {
            'Original box, original papers': 3,
            'Original box, no original papers': 2,
            'Original papers, no original box': 1,
            'No original box, no original papers': 0
        }.get(watch.contenu_livraison.value, 0),
        "seller_consistent": 1 if watch.montres_vendues >= watch.annonces_actives else 0,
        "price_anomaly_low": 0,
        "price_anomaly_high": 0,
        "current_price_estimate": watch.prix_achat ,
        "Brand": watch.marque.value,
        "Movement": watch.mouvement.value,
        "Case material": watch.materiau_boitier.value,
        "Bracelet material": watch.materiau_bracelet.value,
        "Condition": watch.etat.value,
        "Scope of delivery": watch.contenu_livraison.value,
        "Gender": watch.genre.value,
        "Availability": watch.disponibilite.value,
        "Shape": watch.forme.value,
        "Crystal": watch.verre.value,
        "Dial": watch.cadran.value,
        "Bracelet color": watch.couleur_bracelet.value
    }

    df = pd.DataFrame([data])
    return df

def estimate_market_price(brand: str, age: int, condition: str) -> float:
    """Estime le prix du marché actuel"""
    base_prices = {
        "Rolex": 12000, "Patek Philippe": 40000, "Audemars Piguet": 35000,
        "Omega": 6000, "Cartier": 8000, "Tudor": 4000, "TAG Heuer": 3000,
        "Breitling": 5000, "IWC": 7000, "Jaeger-LeCoultre": 8000,
        "Panerai": 7000, "Hublot": 15000, "Zenith": 6000,
        "Seiko": 500, "Casio": 200, "Citizen": 300, "Tissot": 400,
        "Longines": 1500, "Rado": 2000, "Hamilton": 800
    }
    
    base = base_prices.get(brand, 1000)
    
    if age <= 1:
        age_factor = 0.85
    elif age <= 5:
        age_factor = 0.75 - (age - 1) * 0.03
    elif age <= 15:
        age_factor = 0.60 - (age - 5) * 0.02
    elif age <= 30:
        age_factor = 0.40 - (age - 15) * 0.01
    else:
        age_factor = 0.25
    
    condition_factors = {
        "New": 1.0, "Unworn": 0.95, "Like new & unworn": 0.92,
        "Used (Mint)": 0.88, "Used (Very good)": 0.80,
        "Used (Good)": 0.65, "Used (Fair)": 0.50, "Used (Poor)": 0.35
    }
    condition_factor = condition_factors.get(condition, 0.70)
    
    brand_premium = 1.2 if brand in ["Rolex", "Patek Philippe", "Audemars Piguet"] else \
                    0.7 if brand in ["Seiko", "Casio", "Citizen"] else 1.0
    
    return max(100, base * age_factor * condition_factor * brand_premium)

def evaluate_investment(watch: CompleteWatchInput) -> InvestmentResult:
    """Évalue l'investissement"""
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")

    if watch.prix_achat < 100:
        raise HTTPException(status_code=400, detail="Prix d'achat minimum: 100$")

    X = prepare_data(watch)

    # Prédiction prix
    price_log = price_model.predict(X)[0]
    prix_futur = float(np.expm1(price_log))

    # Calculs ROI
    plus_value = prix_futur - watch.prix_achat
    roi_total = (plus_value / watch.prix_achat) * 100
    roi_annualise = roi_total / watch.horizon_annees if watch.horizon_annees > 0 else 0

    # Classification - déterminer la classe prédite
    proba = clf_model.predict_proba(X)[0]
    class_names = ["Risqué", "Moyen", "Bon investissement"]
    predicted_class_idx = np.argmax(proba)
    predicted_class = class_names[predicted_class_idx]
    predicted_proba = proba[predicted_class_idx]

    # Recommandation basée sur la classe prédite ET le ROI
    # 🔒 Règle absolue : ROI négatif = mauvais investissement
    if roi_total < 0:
        recommandation = "MAUVAIS INVESTISSEMENT - ÉVITER"
        confiance = "Élevée"
        risque = "Très Élevé"
        evaluation_simple = "Risqué"

    # 🟠 ROI faible mais positif
    elif roi_annualise < InvestmentThreshold.ROI_ACCEPTABLE:
        recommandation = "INVESTISSEMENT RISQUÉ"
        confiance = "Faible"
        risque = "Élevé"
        evaluation_simple = "Risqué"

    # 🟡 ROI correct
    elif roi_annualise < InvestmentThreshold.ROI_GOOD:
        recommandation = "INVESTISSEMENT ACCEPTABLE"
        confiance = "Moyenne"
        risque = "Moyen"
        evaluation_simple = "Moyen"

    # 🟢 Bon investissement
    elif roi_annualise < InvestmentThreshold.ROI_EXCELLENT:
        recommandation = "BON INVESTISSEMENT"
        confiance = "Moyenne à Élevée"
        risque = "Modéré"
        evaluation_simple = "Bon"

    # 🟢🟢 Excellent investissement
    else:
        recommandation = "EXCELLENT INVESTISSEMENT"
        confiance = "Élevée"
        risque = "Faible"
        evaluation_simple = "Bon"


    # Estimation prix marché
    age = int(X["age"].iloc[0])
    prix_marche = estimate_market_price(watch.marque.value, age, watch.etat.value)
    
    if watch.prix_achat < prix_marche * 0.9:
        deal_quality = "Excellente affaire (sous le marché)"
    elif watch.prix_achat < prix_marche * 1.05:
        deal_quality = "Prix du marché"
    else:
        deal_quality = "Cher (au-dessus du marché)"

    return InvestmentResult(
        prix_achat=watch.prix_achat,
        prix_futur_estime=round(prix_futur, 2),
        plus_value=round(plus_value, 2),
        roi_percent=round(roi_total, 2),
        roi_annualise=round(roi_annualise, 2),
        horizon_annees=watch.horizon_annees,
        recommandation=recommandation,
        confiance=confiance,
        risque=risque,
        evaluation_simple=evaluation_simple,
        details={
            "prix_marche_actuel": round(prix_marche, 2),
            "qualite_affaire": deal_quality,
            "difference_avec_marche": round(watch.prix_achat - prix_marche, 2)
            # SUPPRESSION: plus de "probabilites" ici
        }
    )
# ============================================================
# MODÈLES DE RÉPONSE XAI
# ============================================================

class SHAPFeature(BaseModel):
    name: str
    value: float  # Valeur SHAP
    impact: str  # "positive" ou "negative"
    description: str

class LIMEContribution(BaseModel):
    feature: str
    value: str
    contribution: float
    impact: str  # "favorable" ou "defavorable"
    description: str

class XAIResponse(BaseModel):
    shap: Dict[str, Any]
    lime: Dict[str, Any]
    top_features: List[Dict[str, Any]]

# ============================================================
# FONCTIONS XAI
# ============================================================

def generate_shap_explanation(X: pd.DataFrame, model) -> Dict[str, Any]:
    """Génère l'explication SHAP au format de vos captures"""
    if not SHAP_OK:
        return {"available": False, "error": "SHAP non disponible"}
    
    try:
        if model is None:
            return {"available": False, "error": "Modèle non chargé"}
        
        # Mapping noms français
        feature_names_fr = {
            "Year of production": "Année",
            "age": "Âge", 
            "Face Area": "Surface",
            "Watches Sold by the Seller": "Ventes",
            "Active listing of the seller": "Annonces",
            "Seller Reviews": "Avis",
            "seller_reputation_score": "Réputation",
            "scope_score": "Livraison",
            "Fast Shipper": "Expédition",
            "Trusted Seller": "Vendeur de confiance",
            "Punctuality": "Ponctualité",
            "is_modern": "Moderne",
            "seller_consistent": "Cohérence",
            "price_anomaly_low": "Anomalie -",
            "price_anomaly_high": "Anomalie +",
            "current_price_estimate": "Prix actuel",
            "Brand": "Marque",
            "Movement": "Mouvement",
            "Case material": "Matériau",
            "Bracelet material": "Bracelet",
            "Condition": "État",
            "Scope of delivery": "Box/Papiers",
            "Gender": "Genre",
            "Availability": "Stock",
            "Shape": "Forme",
            "Crystal": "Verre",
            "Dial": "Cadran",
            "Bracelet color": "Couleur"
        }
        
        # Extraire le modèle final
        actual_model = model
        if hasattr(model, 'named_steps'):
            last_step_name = list(model.named_steps.keys())[-1]
            actual_model = model.named_steps[last_step_name]
        
        # Préprocessing si nécessaire
        X_processed = X.copy()
        if hasattr(model, 'named_steps') and 'preprocessing' in model.named_steps:
            X_processed = model.named_steps['preprocessing'].transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
        
        # TreeExplainer
        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X_processed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
            
        # Créer les données du graphique
        features_data = []
        for i, col in enumerate(X.columns):
            if i < len(shap_values):
                val = float(shap_values[i])
                features_data.append({
                    "name": feature_names_fr.get(col, col),
                    "value": val,
                    "impact": "positive" if val > 0 else "negative"
                })
        
        # Trier par valeur absolue
        features_data.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        # Format pour le graphique (comme votre capture)
        chart_data = []
        for f in features_data[:10]:  # Top 10
            chart_data.append({
                "feature": f["name"],
                "value": round(f["value"], 3),
                "color": "#22c55e" if f["impact"] == "positive" else "#ef4444"  # Vert/Rouge
            })
        
        # Top 5 pour les cartes en dessous
        top_features = [
            {"rank": i+1, "name": f["name"]} 
            for i, f in enumerate(features_data[:5])
        ]
        
        return {
            "available": True,
            "type": "global",
            "title": "Importance des caractéristiques (SHAP)",
            "description": "Les valeurs SHAP montrent l'impact de chaque caractéristique sur la prédiction finale. Les valeurs positives (vert) augmentent le prix prédit, les négatives (rouge) le diminuent.",
            "chart_data": chart_data,
            "top_features": top_features,
            "legend": {
                "positive": "Augmente le prix",
                "negative": "Diminue le prix"
            }
        }
            
    except Exception as e:
        print(f"SHAP Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"available": False, "error": str(e)}

def generate_lime_explanation(X: pd.DataFrame, model, feature_names: List[str]) -> Dict[str, Any]:
    """Génère l'explication LIME au format de vos captures"""
    if not LIME_OK:
        return {"available": False, "error": "LIME non disponible"}
    
    try:
        if model is None or not hasattr(model, 'predict_proba'):
            return {"available": False, "error": "Modèle incompatible"}
        
        # Encoder les données
        X_numeric = X.copy()
        label_encoders = {}
        categorical_features = []
        
        for i, col in enumerate(X_numeric.columns):
            if X_numeric[col].dtype == 'object':
                categorical_features.append(i)
                le = LabelEncoder()
                X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
                label_encoders[col] = le
        
        # Créer explainer
        explainer = LimeTabularExplainer(
            training_data=X_numeric.values,
            feature_names=list(X.columns),
            class_names=['Risqué', 'Moyen', 'Bon investissement'],
            mode='classification',
            discretize_continuous=False,
            sample_around_instance=True,
            random_state=42
        )
        
        # Fonction de prédiction wrapper
        def predict_fn(x):
            if isinstance(x, list):
                x = np.array(x)
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            
            # Reconstruire DataFrame avec bon types
            X_df = pd.DataFrame(x, columns=X.columns)
            for col, le in label_encoders.items():
                if col in X_df.columns:
                    vals = X_df[col].astype(int)
                    decoded = []
                    for v in vals:
                        if 0 <= v < len(le.classes_):
                            decoded.append(le.classes_[v])
                        else:
                            decoded.append(le.classes_[0])
                    X_df[col] = decoded
            
            return model.predict_proba(X_df)
        
        # Générer explication
        exp = explainer.explain_instance(
            X_numeric.values[0],
            predict_fn,
            num_features=6,
            top_labels=1
        )
        
        label = exp.available_labels()[0]
        explanations = exp.as_list(label=label)
        
        # Mapping pour les noms lisibles
        feature_map = {
            "Brand": "Marque",
            "Condition": "État", 
            "current_price_estimate": "Prix",
            "Trusted Seller": "Vendeur de confiance",
            "Seller Reviews": "Avis",
            "Movement": "Mouvement",
            "Case material": "Matériau",
            "Year of production": "Année"
        }
        
        # Parser les explications LIME
        contributions = []
        chart_data = []
        
        for feature_str, weight in explanations:
            weight_float = float(weight)
            is_positive = weight_float > 0
            
            # Parser le feature string (ex: "Brand=Rolex")
            if '=' in feature_str:
                feat_name_raw, feat_value = feature_str.split('=', 1)
                feat_name = feature_map.get(feat_name_raw.strip(), feat_name_raw.strip())
                display_name = f"{feat_name} {feat_value.strip()}"
            else:
                feat_name = feature_map.get(feature_str.strip(), feature_str.strip())
                display_name = feat_name
                feat_value = ""
            
            # Déterminer l'impact textuel
            impact_text = "favorable" if is_positive else "defavorable"
            
            # Descriptions contextuelles
            descriptions = {
                "Marque": "La marque a une excellente réputation sur le marché." if is_positive else "Marque moins reconnue sur le marché.",
                "État": "L'état augmente la valeur perçue." if is_positive else "L'état diminue la valeur perçue.",
                "Prix": "Le prix d'achat initial affecte le potentiel de ROI." if not is_positive else "Prix attractif pour l'investissement.",
                "Vendeur de confiance": "La confiance dans le vendeur réduit le risque." if is_positive else "Manque de confiance dans le vendeur.",
                "Année": "L'année de production influence la rareté et la collectibilité." if not is_positive else "Année recherchée par les collectionneurs.",
                "Avis": "Les avis positifs renforcent la crédibilité." if is_positive else "Peu d'avis disponibles."
            }
            
            contrib_data = {
                "feature": feat_name,
                "value": feat_value,
                "contribution": round(abs(weight_float), 4),
                "raw_value": round(weight_float, 4),
                "impact": impact_text,
                "description": descriptions.get(feat_name, "Impact sur la prédiction")
            }
            
            contributions.append(contrib_data)
            
            # Données pour le graphique
            chart_data.append({
                "feature": display_name,
                "value": round(weight_float, 4),
                "color": "#22c55e" if is_positive else "#ef4444"
            })
        
        # Trier contributions par importance absolue
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        chart_data.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "available": True,
            "type": "local",
            "title": "Explication locale (LIME)",
            "description": "LIME montre quelles caractéristiques ont contribué positivement ou négativement à la prédiction pour cette montre spécifique.",
            "chart_data": chart_data,
            "contributions": contributions,
            "legend": {
                "positive": "Favorise l'investissement",
                "negative": "Défavorise l'investissement"
            }
        }
            
    except Exception as e:
        print(f"LIME Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"available": False, "error": str(e)}
# ============================================================
# ENDPOINTS PAR ONGLET
# ============================================================

@app.get("/health")
def health():
    """Vérifie l'état de l'API"""
    return {
        "status": "ok" if MODELS_READY else "error",
        "version": "3.0-structured",
        "models": {
            "price": price_model is not None, 
            "classifier": clf_model is not None
        },
        "xai": {
            "shap": {
                "available": SHAP_OK,
                "version": shap.__version__ if SHAP_OK else None
            },
            "lime": {
                "available": LIME_OK,
                "version": None  # LIME n'a pas souvent d'attribut __version__ direct
            }
        },
        "endpoints": ["/general", "/materials", "/seller", "/analyze", "/explain"]
    }

@app.get("/options")
def get_form_options():
    """
    Retourne toutes les options pour les dropdowns du frontend
    """
    return {
        "marques": [b.value for b in BrandEnum],
        "etats": [c.value for c in ConditionEnum],
        "mouvements": [m.value for m in MovementEnum],
        "genres": [g.value for g in GenderEnum],
        "formes": [s.value for s in ShapeEnum],
        "materiaux": [m.value for m in MaterialEnum],
        "verres": [c.value for c in CrystalEnum],
        "couleurs": [c.value for c in ColorEnum],
        "disponibilites": [a.value for a in AvailabilityEnum],
        "box": [d.value for d in DeliveryEnum]
    }
# ============================================================
# ENDPOINTS INDÉPENDANTS AVEC PRÉDICTION
# ============================================================

@app.post("/general")
def analyze_general(data: GeneralTab):
    """
    Étape 1: Analyse basée UNIQUEMENT sur l'onglet Général
    Matériaux et vendeur = valeurs par défaut standards
    """
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    # Crée objet complet avec défauts pour matériaux et vendeur
    complete = CompleteWatchInput(
        # Général (depuis input)
        prix_achat=data.prix_achat,
        marque=data.marque,
        etat=data.etat,
        mouvement=data.mouvement,
        genre=data.genre,
        annee_production=data.annee_production,
        forme=data.forme,
        surface_cadran=data.surface_cadran,
        contenu_livraison=data.contenu_livraison,
        horizon_annees=data.horizon_annees,
        # Matériaux (défauts neutres)
        materiau_boitier=MaterialEnum.STEEL,
        materiau_bracelet=MaterialEnum.STEEL,
        verre=CrystalEnum.SAPPHIRE,
        cadran=ColorEnum.BLACK,
        couleur_bracelet=ColorEnum.SILVER,
        disponibilite=AvailabilityEnum.IN_STOCK,
        # Vendeur (défauts moyens)
        montres_vendues=50,
        annonces_actives=10,
        avis_vendeur=100,
        expedition_rapide=1,
        vendeur_confiance=1,
        ponctualite=1
    )
    
    result = evaluate_investment(complete)
    
    return {
        "onglet": "general",
        "analyse_basee_sur": "Caractéristiques générales uniquement",
        "precision": "Estimation grossière",
        "avertissement": "Matériaux et vendeur utilisent des valeurs par défaut",
        "prediction": {
            "prix_achat": result.prix_achat,
            "prix_futur_estime": result.prix_futur_estime,
            "plus_value": result.plus_value,
            "roi_percent": result.roi_percent,
            "roi_annualise": result.roi_annualise,
            "evaluation": result.evaluation_simple,
            "recommandation": result.recommandation,
            "confiance": "Faible - complétez les onglets Matériaux et Vendeur"
        },
        "hypotheses_utilisees": {
            "materiaux": {
                "boitier": "Steel (défaut)",
                "bracelet": "Steel (défaut)",
                "verre": "Sapphire crystal (défaut)"
            },
            "vendeur": {
                "reputation": "Moyenne (défaut)",
                "avis": 100,
                "confiance": "Standard"
            }
        },
        "prochaine_etape": "Rendez-vous dans l'onglet Matériaux pour affiner la prédiction"
    }

@app.post("/materials")
def analyze_materials(data: MaterialsTab):
    """
    Étape 2: Analyse basée UNIQUEMENT sur l'onglet Matériaux
    Général et vendeur = valeurs par défaut
    """
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    # Valeurs par défaut pour Général (montre "standard" milieu de gamme)
    complete = CompleteWatchInput(
        # Général (défauts)
        prix_achat=15000,  # Prix moyen par défaut
        marque=BrandEnum.OMEGA,  # Marque mid-tier par défaut
        etat=ConditionEnum.LIKE_NEW,
        mouvement=MovementEnum.AUTOMATIC,
        genre=GenderEnum.MEN,
        annee_production=2020,
        forme=ShapeEnum.ROUND,
        surface_cadran=40.0,
        contenu_livraison=DeliveryEnum.BOX_PAPERS,
        horizon_annees=3,
        # Matériaux (depuis input)
        materiau_boitier=data.materiau_boitier,
        materiau_bracelet=data.materiau_bracelet,
        verre=data.verre,
        cadran=data.cadran,
        couleur_bracelet=data.couleur_bracelet,
        disponibilite=data.disponibilite,
        # Vendeur (défauts)
        montres_vendues=50,
        annonces_actives=10,
        avis_vendeur=100,
        expedition_rapide=1,
        vendeur_confiance=1,
        ponctualite=1
    )
    
    result = evaluate_investment(complete)
    
    # Analyse spécifique matériaux
    premium_materials = []
    if data.materiau_boitier in [MaterialEnum.GOLD, MaterialEnum.PLATINUM, MaterialEnum.ROSE_GOLD, MaterialEnum.WHITE_GOLD]:
        premium_materials.append(f"Boîtier {data.materiau_boitier.value} (+valeur)")
    if data.materiau_bracelet in [MaterialEnum.GOLD, MaterialEnum.PLATINUM, MaterialEnum.ROSE_GOLD, MaterialEnum.WHITE_GOLD]:
        premium_materials.append(f"Bracelet {data.materiau_bracelet.value} (+valeur)")
    if data.verre == CrystalEnum.SAPPHIRE:
        premium_materials.append("Verre saphir (standard qualité)")
    
    return {
        "onglet": "materials",
        "analyse_basee_sur": "Matériaux uniquement",
        "precision": "Estimation matériaux isolés",
        "avertissement": "Caractéristiques générales et vendeur utilisent des valeurs par défaut",
        "prediction": {
            "prix_achat": result.prix_achat,
            "prix_futur_estime": result.prix_futur_estime,
            "plus_value": result.plus_value,
            "roi_percent": result.roi_percent,
            "roi_annualise": result.roi_annualise,
            "evaluation": result.evaluation_simple,
            "recommandation": result.recommandation,
            "confiance": "Moyenne - complétez l'onglet Vendeur pour plus de précision"
        },
        "analyse_materiaux": {
            "composition": {
                "boitier": data.materiau_boitier.value,
                "bracelet": data.materiau_bracelet.value,
                "verre": data.verre.value,
                "cadran": data.cadran.value
            },
            "qualite_percue": "Premium" if len(premium_materials) >= 2 else "Standard",
            "elements_premium": premium_materials if premium_materials else ["Acier standard"],
            "impact_estime": "Élevé" if len(premium_materials) >= 2 else "Modéré"
        },
        "hypotheses_utilisees": {
            "general": {
                "marque": "Omega (défaut)",
                "prix_achat": "15 000$ (défaut)",
                "annee": 2020
            },
            "vendeur": "Standard (défaut)"
        },
        "prochaine_etape": "Rendez-vous dans l'onglet Vendeur pour la prédiction finale"
    }

@app.post("/seller")
def analyze_seller(data: SellerTab):
    """
    Étape 3: Analyse basée UNIQUEMENT sur l'onglet Vendeur
    Général et matériaux = valeurs par défaut
    """
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    # Valeurs par défaut pour Général et Matériaux
    complete = CompleteWatchInput(
        # Général (défauts)
        prix_achat=15000,
        marque=BrandEnum.OMEGA,
        etat=ConditionEnum.LIKE_NEW,
        mouvement=MovementEnum.AUTOMATIC,
        genre=GenderEnum.MEN,
        annee_production=2020,
        forme=ShapeEnum.ROUND,
        surface_cadran=40.0,
        contenu_livraison=DeliveryEnum.BOX_PAPERS,
        horizon_annees=3,
        # Matériaux (défauts)
        materiau_boitier=MaterialEnum.STEEL,
        materiau_bracelet=MaterialEnum.STEEL,
        verre=CrystalEnum.SAPPHIRE,
        cadran=ColorEnum.BLACK,
        couleur_bracelet=ColorEnum.SILVER,
        disponibilite=AvailabilityEnum.IN_STOCK,
        # Vendeur (depuis input)
        montres_vendues=data.montres_vendues,
        annonces_actives=data.annonces_actives,
        avis_vendeur=data.avis_vendeur,
        expedition_rapide=data.expedition_rapide,
        vendeur_confiance=data.vendeur_confiance,
        ponctualite=data.ponctualite
    )
    
    result = evaluate_investment(complete)
    
    # Analyse spécifique vendeur
    reputation_score = (data.expedition_rapide + data.vendeur_confiance + data.ponctualite) / 3
    activite_ratio = data.montres_vendues / max(data.annonces_actives, 1)
    
    qualite_vendeur = "Excellente" if reputation_score >= 0.9 and data.avis_vendeur > 200 else \
                     "Bonne" if reputation_score >= 0.7 and data.avis_vendeur > 50 else \
                     "Moyenne" if reputation_score >= 0.5 else "À vérifier"
    
    return {
        "onglet": "seller",
        "analyse_basee_sur": "Profil vendeur uniquement",
        "precision": "Analyse risque transactionnel",
        "avertissement": "Caractéristiques montre et matériaux utilisent des valeurs par défaut",
        "prediction": {
            "prix_achat": result.prix_achat,
            "prix_futur_estime": result.prix_futur_estime,
            "plus_value": result.plus_value,
            "roi_percent": result.roi_percent,
            "roi_annualise": result.roi_annualise,
            "evaluation": result.evaluation_simple,
            "recommandation": result.recommandation,
            "confiance": "Variable selon qualité vendeur"
        },
        "analyse_vendeur": {
            "profil": {
                "qualite": qualite_vendeur,
                "reputation_score": round(reputation_score * 100, 1),
                "nb_avis": data.avis_vendeur,
                "experience": "Confirmé" if data.montres_vendues > 100 else "Débutant"
            },
            "fiabilite": {
                "expedition_rapide": bool(data.expedition_rapide),
                "vendeur_verifie": bool(data.vendeur_confiance),
                "ponctuel": bool(data.ponctualite)
            },
            "activite": {
                "montres_vendues": data.montres_vendues,
                "annonces_actives": data.annonces_actives,
                "ratio_rotation": round(activite_ratio, 2)
            },
            "impact_transaction": "Faible risque" if qualite_vendeur == "Excellente" else "Risque modéré" if qualite_vendeur == "Bonne" else "Attention requise"
        },
        "hypotheses_utilisees": {
            "general": "Montre Omega 15k$ (défaut)",
            "materiaux": "Acier standard (défaut)"
        },
        "note_finale": "Pour une prédiction précise, utilisez /analyze avec toutes les données réelles"
    }

@app.post("/analyze", response_model=InvestmentResult)
def analyze_complete(data: CompleteWatchInput):
    """
    Analyse COMPLÈTE avec TOUS les onglets
    Prédiction finale précise - utilisez celui-ci pour le résultat définitif
    """
    try:
        result = evaluate_investment(data)
        return result
    except Exception as e:
        raise HTTPException(500, f"Erreur analyse: {str(e)}")
    

@app.post("/explain")
def explain_prediction(data: CompleteWatchInput):
    """
    Explications XAI complètes (SHAP + LIME) pour la prédiction
    """
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    # Préparer les données
    X = prepare_data(data)
    
    # Convertir les types si nécessaire
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str)
    
    result = {
        "evaluation": {},
        "shap": {},
        "lime": {}
    }
    
    # Évaluation
    try:
        eval_result = evaluate_investment(data)
        result["evaluation"] = {
            "prix_achat": eval_result.prix_achat,
            "prix_futur": eval_result.prix_futur_estime,
            "roi": eval_result.roi_annualise,
            "recommandation": eval_result.recommandation,
            "evaluation_simple": eval_result.evaluation_simple
        }
    except Exception as e:
        result["evaluation_error"] = str(e)
    
    # SHAP avec le modèle de prix
    result["shap"] = generate_shap_explanation(X, price_model)
    
    # LIME avec le classifieur
    result["lime"] = generate_lime_explanation(X, clf_model, list(X.columns))
    
    return result
# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 API Montres de Luxe - Structure par Onglets v3.0")
    print("="*60)
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔍 Health Check:  http://localhost:8000/health")
    print("\nEndpoints:")
    print("   POST /general    → Analyse onglet Général")
    print("   POST /materials  → Analyse onglet Matériaux (nécessite GeneralTab)")
    print("   POST /seller     → Analyse onglet Vendeur (nécessite GeneralTab)")
    print("   POST /analyze    → Analyse complète (tous onglets)")
    print("   GET  /options    → Options dropdowns")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)