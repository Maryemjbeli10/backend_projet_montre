import React, { useState, useEffect } from 'react';
import { 
  Watch, 
  Settings, 
  Package, 
  User, 
  DollarSign, 
  Calendar, 
  Maximize, 
  Gem, 
  Store, 
  Activity,
  Clock,
  Shield,
  TrendingUp,
  TrendingDown,
  Globe,
  Lightbulb,
  BarChart3,
  RefreshCw,
  ArrowRight,
  ChevronDown,
  Loader2,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  ChevronRight
} from 'lucide-react';

// Configuration API
const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [currentView, setCurrentView] = useState('landing');
  const [activeTab, setActiveTab] = useState('general');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiOptions, setApiOptions] = useState(null);
  const [touchedFields, setTouchedFields] = useState({});

  // États du formulaire - TOUS LES CHAMPS SONT OBLIGATOIRES
  const [formData, setFormData] = useState({
    // Général
    prix_achat: '',
    marque: '',
    etat: '',
    mouvement: '',
    genre: '',
    annee_production: '',
    forme: '',
    surface_cadran: '',
    contenu_livraison: '',
    horizon_annees: 3,
    
    // Matériaux
    materiau_boitier: '',
    materiau_bracelet: '',
    verre: '',
    cadran: '',
    couleur_bracelet: '',
    disponibilite: '',
    
    // Vendeur
    montres_vendues: '',
    annonces_actives: '',
    avis_vendeur: '',
    expedition_rapide: 0,
    vendeur_confiance: 0,
    ponctualite: 0
  });

  // Vérifier si tous les champs sont remplis
  const isFormComplete = () => {
    // Vérifier tous les champs sauf les toggles (0/1 sont valides)
    const requiredFields = [
      'prix_achat', 'marque', 'etat', 'mouvement', 'genre',
      'annee_production', 'forme', 'surface_cadran', 'contenu_livraison',
      'materiau_boitier', 'materiau_bracelet', 'verre', 'cadran',
      'couleur_bracelet', 'disponibilite',
      'montres_vendues', 'annonces_actives', 'avis_vendeur'
    ];

    return requiredFields.every(field => {
      const value = formData[field];
      return value !== '' && value !== null && value !== undefined && value !== 0;
    });
  };

  // Vérifier la validité d'un champ spécifique
  const isFieldValid = (field) => {
    const value = formData[field];
    return value !== '' && value !== null && value !== undefined;
  };

  // Charger les options au démarrage
  useEffect(() => {
    fetchOptions();
  }, []);

  const fetchOptions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/options`);
      if (response.ok) {
        const data = await response.json();
        setApiOptions(data);
      }
    } catch (err) {
      console.error('Erreur chargement options:', err);
    }
  };

  const updateField = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setTouchedFields(prev => ({ ...prev, [field]: true }));
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!isFormComplete()) {
      setError('Veuillez remplir tous les champs obligatoires');
      // Marquer tous les champs comme touchés pour afficher les erreurs
      const allFields = Object.keys(formData).reduce((acc, key) => {
        acc[key] = true;
        return acc;
      }, {});
      setTouchedFields(allFields);
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Erreur lors de l\'analyse');
      }

      const data = await response.json();
      setResult(data);
      setCurrentView('results');
      
    } catch (err) {
      setError(err.message || 'Erreur de connexion au serveur');
      console.error('Erreur:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExplain = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/explain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        const data = await response.json();
        setResult(prev => ({ ...prev, explanations: data }));
      }
    } catch (err) {
      console.error('Erreur explications:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Navigation entre onglets avec validation
  const goToNextTab = () => {
    // Marquer les champs de l'onglet actuel comme touchés
    const currentFields = getTabFields(activeTab);
    setTouchedFields(prev => ({
      ...prev,
      ...currentFields.reduce((acc, field) => ({ ...acc, [field]: true }), {})
    }));

    if (activeTab === 'general') setActiveTab('materials');
    else if (activeTab === 'materials') setActiveTab('seller');
  };

  const goToPrevTab = () => {
    if (activeTab === 'materials') setActiveTab('general');
    else if (activeTab === 'seller') setActiveTab('materials');
  };

  const getTabFields = (tab) => {
    switch(tab) {
      case 'general':
        return ['prix_achat', 'marque', 'etat', 'mouvement', 'genre', 'annee_production', 'forme', 'surface_cadran', 'contenu_livraison', 'horizon_annees'];
      case 'materials':
        return ['materiau_boitier', 'materiau_bracelet', 'verre', 'cadran', 'couleur_bracelet', 'disponibilite'];
      case 'seller':
        return ['montres_vendues', 'annonces_actives', 'avis_vendeur', 'expedition_rapide', 'vendeur_confiance', 'ponctualite'];
      default:
        return [];
    }
  };

  // Vérifier si l'onglet actuel est complet
  const isCurrentTabComplete = () => {
    const currentFields = getTabFields(activeTab);
    return currentFields.every(field => isFieldValid(field));
  };

  if (currentView === 'landing') {
    return <LandingPage onStart={() => setCurrentView('form')} />;
  }

  if (currentView === 'results' && result) {
    return (
      <ResultsPage 
        result={result} 
        onBack={() => setCurrentView('form')}
        onExplain={handleExplain}
        isLoading={isLoading}
      />
    );
  }

  const formComplete = isFormComplete();

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-serif font-bold mb-2">
            Saisissez les caractéristiques de la montre
          </h1>
          <p className="text-gray-400">
            Tous les champs sont obligatoires pour une analyse précise.
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
            <span>Progression</span>
            <span>{formComplete ? '100%' : activeTab === 'general' ? '33%' : activeTab === 'materials' ? '66%' : '90%'}</span>
          </div>
          <div className="h-2 bg-[#1a1a1a] rounded-full overflow-hidden">
            <div 
              className="h-full bg-amber-500 transition-all duration-500"
              style={{ 
                width: formComplete ? '100%' : activeTab === 'general' ? '33%' : activeTab === 'materials' ? '66%' : '90%' 
              }}
            />
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-3 text-red-400">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* Card */}
        <div className="bg-[#111] rounded-2xl border border-white/10 overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-white/10">
            <TabButton 
              active={activeTab === 'general'} 
              onClick={() => setActiveTab('general')}
              icon={<Watch size={16} />}
              label="Général"
              completed={isTabComplete('general')}
            />
            <TabButton 
              active={activeTab === 'materials'} 
              onClick={() => setActiveTab('materials')}
              icon={<Gem size={16} />}
              label="Matériaux"
              completed={isTabComplete('materials')}
            />
            <TabButton 
              active={activeTab === 'seller'} 
              onClick={() => setActiveTab('seller')}
              icon={<Store size={16} />}
              label="Vendeur"
              completed={isTabComplete('seller')}
            />
          </div>

          {/* Content */}
          <div className="p-6">
            {activeTab === 'general' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <SelectField 
                  label="Marque *" 
                  value={formData.marque} 
                  onChange={(v) => updateField('marque', v)}
                  options={['', 'Rolex', 'Patek Philippe', 'Audemars Piguet', 'Omega', 'Cartier', 'Tudor', 'TAG Heuer', 'Breitling', 'IWC', 'Jaeger-LeCoultre', 'Panerai', 'Hublot', 'Zenith', 'Seiko', 'Casio', 'Citizen', 'Tissot', 'Longines', 'Rado', 'Hamilton', 'Cyma', 'Concord', 'Squale', 'Gucci', 'Cuervo y Sobrinos', 'Norqain', 'Davosa', 'Poljot', 'Revue Thommen', 'CWC', 'Richard Mille', 'Greubel Forsey', 'Graf', 'Laurent Ferrier', 'Dolce & Gabbana', 'Mercure']}
                  icon={<Watch size={16} />}
                  required
                  error={touchedFields.marque && !isFieldValid('marque')}
                />
                <SelectField 
                  label="Mouvement *" 
                  value={formData.mouvement} 
                  onChange={(v) => updateField('mouvement', v)}
                  options={['', 'Automatic', 'Manual winding', 'Quartz']}
                  icon={<Settings size={16} />}
                  required
                  error={touchedFields.mouvement && !isFieldValid('mouvement')}
                />
                <SelectField 
                  label="État *" 
                  value={formData.etat} 
                  onChange={(v) => updateField('etat', v)}
                  options={['', 'New', 'Unworn', 'Like new & unworn', 'Used (Mint)', 'Used (Very good)', 'Used (Good)', 'Used (Fair)', 'Used (Poor)']}
                  icon={<Package size={16} />}
                  required
                  error={touchedFields.etat && !isFieldValid('etat')}
                />
                <SelectField 
                  label="Genre *" 
                  value={formData.genre} 
                  onChange={(v) => updateField('genre', v)}
                  options={['', "Men's watch", "Women's watch", 'Unisex']}
                  icon={<User size={16} />}
                  required
                  error={touchedFields.genre && !isFieldValid('genre')}
                />
                <NumberField 
                  label="Prix d'achat ($) *" 
                  value={formData.prix_achat} 
                  onChange={(v) => updateField('prix_achat', v)}
                  icon={<DollarSign size={16} />}
                  min={100}
                  required
                  error={touchedFields.prix_achat && !isFieldValid('prix_achat')}
                  placeholder="Ex: 10000"
                />
                <NumberField 
                  label="Année de production *" 
                  value={formData.annee_production} 
                  onChange={(v) => updateField('annee_production', v)}
                  icon={<Calendar size={16} />}
                  min={1900}
                  max={2030}
                  required
                  error={touchedFields.annee_production && !isFieldValid('annee_production')}
                  placeholder="Ex: 2020"
                />
                <SelectField 
                  label="Forme *" 
                  value={formData.forme} 
                  onChange={(v) => updateField('forme', v)}
                  options={['', 'Round', 'Square', 'Rectangular', 'Tonneau', 'Cushion', 'Octagonal']}
                  icon={<Maximize size={16} />}
                  required
                  error={touchedFields.forme && !isFieldValid('forme')}
                />
                <NumberField 
                  label="Surface du cadran (mm²) *" 
                  value={formData.surface_cadran} 
                  onChange={(v) => updateField('surface_cadran', v)}
                  icon={<Maximize size={16} />}
                  min={1}
                  required
                  error={touchedFields.surface_cadran && !isFieldValid('surface_cadran')}
                  placeholder="Ex: 40"
                />
                <SelectField 
                  label="Contenu de la livraison *" 
                  value={formData.contenu_livraison} 
                  onChange={(v) => updateField('contenu_livraison', v)}
                  options={['', 'Original box, original papers', 'Original box, no original papers', 'Original papers, no original box', 'No original box, no original papers']}
                  icon={<Package size={16} />}
                  required
                  error={touchedFields.contenu_livraison && !isFieldValid('contenu_livraison')}
                />
                <NumberField 
                  label="Horizon d'investissement (années) *" 
                  value={formData.horizon_annees} 
                  onChange={(v) => updateField('horizon_annees', v)}
                  icon={<Clock size={16} />}
                  min={1}
                  max={10}
                  required
                  error={touchedFields.horizon_annees && !isFieldValid('horizon_annees')}
                />
              </div>
            )}

            {activeTab === 'materials' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <SelectField 
                  label="Matériau du boîtier *" 
                  value={formData.materiau_boitier} 
                  onChange={(v) => updateField('materiau_boitier', v)}
                  options={['', 'Steel', 'Yellow gold', 'Rose gold', 'White gold', 'Platinum', 'Titanium', 'Ceramic', 'Carbon', 'Plastic', 'Red gold', 'Bronze', 'Aluminum', 'Sapphire crystal']}
                  icon={<Gem size={16} />}
                  required
                  error={touchedFields.materiau_boitier && !isFieldValid('materiau_boitier')}
                />
                <SelectField 
                  label="Matériau du bracelet *" 
                  value={formData.materiau_bracelet} 
                  onChange={(v) => updateField('materiau_bracelet', v)}
                  options={['', 'Steel', 'Yellow gold', 'Rose gold', 'White gold', 'Platinum', 'Titanium', 'Ceramic', 'Carbon', 'Plastic', 'Red gold', 'Bronze', 'Aluminum', 'Sapphire crystal']}
                  icon={<Gem size={16} />}
                  required
                  error={touchedFields.materiau_bracelet && !isFieldValid('materiau_bracelet')}
                />
                <SelectField 
                  label="Verre *" 
                  value={formData.verre} 
                  onChange={(v) => updateField('verre', v)}
                  options={['', 'Sapphire crystal', 'Mineral crystal', 'Plexiglass', 'Glass', 'Plastic']}
                  icon={<Gem size={16} />}
                  required
                  error={touchedFields.verre && !isFieldValid('verre')}
                />
                <SelectField 
                  label="Cadran *" 
                  value={formData.cadran} 
                  onChange={(v) => updateField('cadran', v)}
                  options={['', 'Black', 'Silver', 'White', 'Blue', 'Green', 'Brown', 'Champagne', 'Grey', 'Steel', 'Orange', 'Pink', 'Purple', 'Bronze', 'Red']}
                  icon={<Maximize size={16} />}
                  required
                  error={touchedFields.cadran && !isFieldValid('cadran')}
                />
                <SelectField 
                  label="Couleur du bracelet *" 
                  value={formData.couleur_bracelet} 
                  onChange={(v) => updateField('couleur_bracelet', v)}
                  options={['', 'Black', 'Silver', 'White', 'Blue', 'Green', 'Brown', 'Champagne', 'Grey', 'Steel', 'Orange', 'Pink', 'Purple', 'Bronze', 'Red']}
                  icon={<Settings size={16} />}
                  required
                  error={touchedFields.couleur_bracelet && !isFieldValid('couleur_bracelet')}
                />
                <SelectField 
                  label="Disponibilité *" 
                  value={formData.disponibilite} 
                  onChange={(v) => updateField('disponibilite', v)}
                  options={['', 'Item is in stock', 'Item is in stock at a partner boutique', 'Item is being serviced', 'Item is on hold']}
                  icon={<Package size={16} />}
                  required
                  error={touchedFields.disponibilite && !isFieldValid('disponibilite')}
                />
              </div>
            )}

            {activeTab === 'seller' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <NumberField 
                    label="Montres vendues par le vendeur *" 
                    value={formData.montres_vendues} 
                    onChange={(v) => updateField('montres_vendues', v)}
                    icon={<Store size={16} />}
                    min={0}
                    required
                    error={touchedFields.montres_vendues && !isFieldValid('montres_vendues')}
                    placeholder="Ex: 50"
                  />
                  <NumberField 
                    label="Annonces actives du vendeur *" 
                    value={formData.annonces_actives} 
                    onChange={(v) => updateField('annonces_actives', v)}
                    icon={<Package size={16} />}
                    min={0}
                    required
                    error={touchedFields.annonces_actives && !isFieldValid('annonces_actives')}
                    placeholder="Ex: 10"
                  />
                  <NumberField 
                    label="Avis du vendeur *" 
                    value={formData.avis_vendeur} 
                    onChange={(v) => updateField('avis_vendeur', v)}
                    icon={<Activity size={16} />}
                    min={0}
                    required
                    error={touchedFields.avis_vendeur && !isFieldValid('avis_vendeur')}
                    placeholder="Ex: 100"
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ToggleCard 
                    title="Expédition rapide *"
                    description="Le vendeur expédie rapidement"
                    checked={formData.expedition_rapide === 1}
                    onChange={(v) => updateField('expedition_rapide', v ? 1 : 0)}
                    icon={<Clock size={20} />}
                  />
                  <ToggleCard 
                    title="Vendeur de confiance *"
                    description="Le vendeur est vérifié"
                    checked={formData.vendeur_confiance === 1}
                    onChange={(v) => updateField('vendeur_confiance', v ? 1 : 0)}
                    icon={<Shield size={20} />}
                  />
                  <ToggleCard 
                    title="Ponctualité *"
                    description="Le vendeur est ponctuel"
                    checked={formData.ponctualite === 1}
                    onChange={(v) => updateField('ponctualite', v ? 1 : 0)}
                    icon={<Clock size={20} />}
                  />
                </div>

                {/* Message si formulaire incomplet */}
                {!formComplete && (
                  <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-center gap-3 text-amber-400">
                    <Info size={20} />
                    <span className="text-sm">
                      Veuillez remplir tous les champs des onglets précédents pour activer l'analyse
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Navigation Buttons */}
            <div className="flex gap-4 mt-8">
              {/* Bouton Précédent (sauf sur premier onglet) */}
              {activeTab !== 'general' && (
                <button 
                  onClick={goToPrevTab}
                  className="flex-1 py-4 px-6 bg-[#1a1a1a] hover:bg-[#2a2a2a] text-white font-semibold rounded-xl transition-colors border border-white/10"
                >
                  ← Précédent
                </button>
              )}
              
              {/* Bouton Suivant ou Analyser */}
              {activeTab !== 'seller' ? (
                // Bouton Continuer (onglets General et Materials)
                <button 
                  onClick={goToNextTab}
                  className="flex-1 py-4 px-6 bg-amber-600 hover:bg-amber-500 text-black font-semibold rounded-xl transition-colors flex items-center justify-center gap-2"
                >
                  <span>Suivant</span>
                  <ChevronRight size={20} />
                </button>
              ) : (
                // Bouton Analyser (uniquement sur onglet Vendeur)
                <button 
                  onClick={handleAnalyze}
                  disabled={isLoading || !formComplete}
                  className={`flex-1 py-4 px-6 font-semibold rounded-xl transition-all duration-300 flex items-center justify-center gap-2 ${
                    formComplete && !isLoading
                      ? 'bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-black cursor-pointer'
                      : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  {isLoading ? (
                    <>
                      <Loader2 size={20} className="animate-spin" />
                      <span>Analyse en cours...</span>
                    </>
                  ) : (
                    <>
                      <TrendingUp size={20} />
                      <span>Analyser l'investissement</span>
                      <ArrowRight size={20} />
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Fonction helper pour vérifier si un onglet est complet
  function isTabComplete(tab) {
    const fields = getTabFields(tab);
    return fields.every(field => isFieldValid(field));
  }
}

// ... (reste des composants inchangés: ResultsPage, TabButton, SelectField, NumberField, ToggleCard, MetricCard, ProbabilityBar, LandingPage, FeatureCard)

// Composants réutilisables

function TabButton({ active, onClick, icon, label, completed }) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-2 py-4 text-sm font-medium transition-colors relative ${
        active 
          ? 'bg-amber-500/20 text-amber-400 border-b-2 border-amber-500' 
          : completed
            ? 'text-green-400 hover:text-green-300'
            : 'text-gray-400 hover:text-white'
      }`}
    >
      {icon}
      {label}
      {completed && !active && (
        <CheckCircle size={14} className="text-green-400" />
      )}
    </button>
  );
}

function SelectField({ label, value, onChange, options, icon, required, error }) {
  return (
    <div className="space-y-2">
      <label className={`flex items-center gap-2 text-sm ${error ? 'text-red-400' : 'text-gray-400'}`}>
        {icon}
        {label}
      </label>
      <div className="relative">
        <select 
          value={value} 
          onChange={(e) => onChange(e.target.value)}
          className={`w-full appearance-none bg-[#1a1a1a] border rounded-lg px-4 py-3 text-white focus:outline-none focus:border-amber-500/50 transition-colors cursor-pointer ${
            error ? 'border-red-500/50' : 'border-white/10'
          }`}
        >
          {options.map(opt => (
            <option key={opt} value={opt}>{opt || 'Sélectionnez...'}</option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" size={16} />
      </div>
      {error && <span className="text-xs text-red-400">Ce champ est obligatoire</span>}
    </div>
  );
}

function NumberField({ label, value, onChange, icon, min, max, required, error, placeholder }) {
  return (
    <div className="space-y-2">
      <label className={`flex items-center gap-2 text-sm ${error ? 'text-red-400' : 'text-gray-400'}`}>
        {icon}
        {label}
      </label>
      <input 
        type="number" 
        value={value} 
        min={min}
        max={max}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
        className={`w-full bg-[#1a1a1a] border rounded-lg px-4 py-3 text-white focus:outline-none focus:border-amber-500/50 transition-colors ${
          error ? 'border-red-500/50' : 'border-white/10'
        }`}
      />
      {error && <span className="text-xs text-red-400">Ce champ est obligatoire</span>}
    </div>
  );
}

function ToggleCard({ title, description, checked, onChange, icon }) {
  return (
    <div className="flex items-center justify-between p-4 bg-[#1a1a1a] rounded-xl border border-white/10">
      <div className="flex items-start gap-3">
        <div className="text-amber-400 mt-1">{icon}</div>
        <div>
          <h4 className="font-medium mb-1 text-white">{title}</h4>
          <p className="text-sm text-gray-400">{description}</p>
        </div>
      </div>
      <button 
        onClick={() => onChange(!checked)}
        className={`relative w-12 h-6 rounded-full transition-colors ${checked ? 'bg-amber-500' : 'bg-gray-600'}`}
      >
        <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${checked ? 'translate-x-7' : 'translate-x-1'}`} />
      </button>
    </div>
  );
}

function MetricCard({ label, value, icon, highlight, positive }) {
  return (
    <div className={`p-4 rounded-xl border ${
      highlight ? 'bg-amber-500/10 border-amber-500/30' : 'bg-[#1a1a1a] border-white/10'
    }`}>
      <div className="flex items-center gap-2 text-gray-400 mb-2">
        {icon}
        <span className="text-sm">{label}</span>
      </div>
      <div className={`text-xl font-bold ${
        positive === true ? 'text-green-400' : 
        positive === false ? 'text-red-400' : 
        highlight ? 'text-amber-400' : 'text-white'
      }`}>
        {value}
      </div>
    </div>
  );
}

function ProbabilityBar({ label, value, color }) {
  const percentage = Math.round(value * 100);
  return (
    <div className="flex items-center gap-4">
      <div className="w-32 text-sm text-gray-400">{label}</div>
      <div className="flex-1 h-2 bg-[#1a1a1a] rounded-full overflow-hidden">
        <div 
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="w-12 text-right text-sm font-medium">{percentage}%</div>
    </div>
  );
}

// Page de résultats avec XAI intégré et section Recommandation
function ResultsPage({ result, onBack, onExplain, isLoading }) {
  const [activeXaiTab, setActiveXaiTab] = useState('shap');
  
  const getEvaluationColor = (evaluation) => {
    switch (evaluation) {
      case 'Bon': return 'text-green-400 bg-green-400/10 border-green-400/30';
      case 'Moyen': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30';
      case 'Risqué': return 'text-red-400 bg-red-400/10 border-red-400/30';
      default: return 'text-gray-400 bg-gray-400/10 border-gray-400/30';
    }
  };

  const getEvaluationIcon = (evaluation) => {
    switch (evaluation) {
      case 'Bon': return <CheckCircle size={24} className="text-green-400" />;
      case 'Moyen': return <Info size={24} className="text-yellow-400" />;
      case 'Risqué': return <XCircle size={24} className="text-red-400" />;
      default: return <Info size={24} />;
    }
  };

  // Calculer la position du ROI sur l'échelle (-50% à +50%)
  const roiPercent = result.roi_percent || 0;
  const roiPosition = Math.min(Math.max((roiPercent + 50) / 100 * 100, 0), 100);
  
  // Déterminer la couleur du gradient basé sur le ROI
  const getRoiColor = () => {
    if (roiPercent >= 25) return 'from-green-500 to-green-400';
    if (roiPercent >= 15) return 'from-green-400 to-yellow-400';
    if (roiPercent >= 5) return 'from-yellow-400 to-orange-400';
    if (roiPercent >= 0) return 'from-orange-400 to-red-400';
    return 'from-red-500 to-red-600';
  };

  // Données XAI
  const xaiData = result.explanations;
  const shapData = xaiData?.shap;
  const limeData = xaiData?.lime;

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-serif font-bold mb-2">
            Résultat de l'analyse
          </h1>
          <p className="text-gray-400">
            Voici l'évaluation de votre investissement basée sur nos modèles ML.
          </p>
        </div>

        {/* Main Result Card */}
        <div className="bg-[#111] rounded-2xl border border-white/10 p-8 mb-6">
          <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-xl border ${getEvaluationColor(result.evaluation_simple)} mb-6`}>
            {getEvaluationIcon(result.evaluation_simple)}
            <div>
              <div className="text-sm opacity-80">Évaluation</div>
              <div className="text-xl font-bold">{result.evaluation_simple}</div>
            </div>
          </div>

          <h2 className="text-2xl font-bold mb-6 text-amber-400">{result.recommandation}</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <MetricCard 
              label="Prix d'achat"
              value={`$${result.prix_achat.toLocaleString()}`}
              icon={<DollarSign size={20} />}
            />
            <MetricCard 
              label="Prix futur estimé"
              value={`$${result.prix_futur_estime.toLocaleString()}`}
              icon={<TrendingUp size={20} />}
              highlight
            />
            <MetricCard 
              label="Plus-value"
              value={`$${result.plus_value.toLocaleString()}`}
              icon={<Activity size={20} />}
              positive={result.plus_value > 0}
            />
            <MetricCard 
              label="ROI Total"
              value={`${result.roi_percent}%`}
              icon={<Clock size={20} />}
              positive={result.roi_percent > 0}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="p-4 bg-[#1a1a1a] rounded-xl">
              <div className="text-gray-400 text-sm mb-1">ROI Annualisé</div>
              <div className={`text-2xl font-bold ${result.roi_annualise >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {result.roi_annualise}%
              </div>
            </div>
            <div className="p-4 bg-[#1a1a1a] rounded-xl">
              <div className="text-gray-400 text-sm mb-1">Horizon</div>
              <div className="text-2xl font-bold text-white">
                {result.horizon_annees} ans
              </div>
            </div>
            <div className="p-4 bg-[#1a1a1a] rounded-xl">
              <div className="text-gray-400 text-sm mb-1">Niveau de confiance</div>
              <div className="text-2xl font-bold text-amber-400">
                {result.confiance}
              </div>
            </div>
          </div>

          {/* NOUVELLE SECTION RECOMMANDATION - Visualisation du ROI */}
          <div className="bg-[#0a0a0a] rounded-2xl p-6 mb-8 border border-white/5">
            <div className="flex items-center gap-3 mb-6">
              <BarChart3 size={20} className="text-amber-400" />
              <h3 className="text-lg font-semibold">Visualisation du ROI</h3>
            </div>
            
            {/* Barre de progression du ROI */}
            <div className="relative mb-6">
              {/* Barre de fond avec gradient */}
              <div className="h-16 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-lg relative overflow-hidden">
                {/* Masque pour ne montrer que jusqu'à la valeur actuelle */}
                <div 
                  className="absolute top-0 left-0 h-full bg-[#1a1a1a] opacity-90"
                  style={{ width: `${100 - roiPosition}%`, right: 0, left: 'auto' }}
                />
                {/* Valeur actuelle */}
                <div className="absolute top-0 left-0 h-full flex items-center px-4">
                  <span className={`text-2xl font-bold ${roiPercent >= 0 ? 'text-white' : 'text-red-400'}`}>
                    {roiPercent > 0 ? '+' : ''}{roiPercent.toFixed(1)}%
                  </span>
                </div>
              </div>
              
              {/* Marqueurs de pourcentage */}
              <div className="flex justify-between mt-2 text-xs text-gray-500 px-2">
                <span>-50%</span>
                <span>-25%</span>
                <span>0%</span>
                <span>+25%</span>
                <span>+50%</span>
              </div>
            </div>

            {/* Légende Perte/Gain */}
            <div className="flex items-center justify-between mb-6">
              <span className="text-sm text-gray-400">Perte potentielle</span>
              <div className="flex-1 mx-4 h-2 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500" />
              <span className="text-sm text-gray-400">Gain potentiel</span>
            </div>

            {/* Texte de recommandation */}
            <div className="bg-[#111] rounded-xl p-6 border border-white/5">
              <h4 className="text-lg font-serif font-semibold mb-3 text-white">Recommandation</h4>
              <p className="text-gray-300 leading-relaxed text-sm">
                {result.roi_percent >= 25 ? (
                  "Cette montre présente un excellent profil d'investissement avec un ROI très positif attendu. Les caractéristiques techniques, la réputation de la marque et les conditions de marché actuelles soutiennent une valorisation future exceptionnelle. Recommandé fortement pour les investisseurs recherchant un actif stable et fortement valorisant."
                ) : result.roi_percent >= 15 ? (
                  "Cette montre présente un bon profil d'investissement avec un ROI positif attendu. Les caractéristiques techniques et la réputation de la marque soutiennent une valorisation future. Recommandé pour les investisseurs recherchant un actif stable et valorisant."
                ) : result.roi_percent >= 5 ? (
                  "Cette montre présente un profil d'investissement modéré. Le ROI attendu est positif mais limité. Convient aux investisseurs prudents ou souhaitant diversifier leur portefeuille avec un risque contrôlé."
                ) : result.roi_percent >= 0 ? (
                  "Cette montre présente un profil d'investissement incertain. Le ROI attendu est faiblement positif. L'investissement peut être considéré pour des raisons de collection plutôt que de spéculation financière."
                ) : (
                  "Cette montre présente un profil d'investissement défavorable avec un ROI négatif attendu. Les caractéristiques actuelles ne soutiennent pas une valorisation future positive. Déconseillé comme investissement financier."
                )}
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-col sm:flex-row gap-4">
            <button 
              onClick={onBack}
              className="flex-1 py-3 px-6 bg-[#1a1a1a] hover:bg-[#2a2a2a] text-white rounded-xl transition-colors"
            >
              Nouvelle analyse
            </button>
            {!xaiData && (
              <button 
                onClick={onExplain}
                disabled={isLoading}
                className="flex-1 py-3 px-6 bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-black font-semibold rounded-xl transition-all disabled:opacity-50"
              >
                {isLoading ? (
                  <>
                    <Loader2 size={20} className="animate-spin inline mr-2" />
                    Chargement...
                  </>
                ) : (
                  <>
                    <Activity size={20} className="inline mr-2" />
                    Expliquer la prédiction (XAI)
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* SECTION XAI - Style comme vos captures */}
        {xaiData && (
          <div className="bg-[#111] rounded-2xl border border-white/10 overflow-hidden">
            {/* Header XAI */}
            <div className="p-6 border-b border-white/10 flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center">
                <Activity size={20} className="text-amber-400" />
              </div>
              <h3 className="text-xl font-bold">Explicabilité (XAI)</h3>
            </div>

            {/* Onglets SHAP / LIME */}
            <div className="flex border-b border-white/10">
              <button
                onClick={() => setActiveXaiTab('shap')}
                className={`flex-1 flex items-center justify-center gap-2 py-4 text-sm font-medium transition-all ${
                  activeXaiTab === 'shap'
                    ? 'bg-[#c9a227]/20 text-[#c9a227] border-b-2 border-[#c9a227]'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Globe size={16} />
                SHAP (Global)
              </button>
              <button
                onClick={() => setActiveXaiTab('lime')}
                className={`flex-1 flex items-center justify-center gap-2 py-4 text-sm font-medium transition-all ${
                  activeXaiTab === 'lime'
                    ? 'bg-[#c9a227]/20 text-[#c9a227] border-b-2 border-[#c9a227]'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Lightbulb size={16} />
                LIME (Local)
              </button>
            </div>

            {/* Contenu SHAP */}
            {activeXaiTab === 'shap' && shapData?.available && (
              <div className="p-6">
                {/* Titre et description */}
                <div className="flex items-start gap-3 mb-6">
                  <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                    <Activity size={16} className="text-amber-400" />
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold mb-2">{shapData.title}</h4>
                    <p className="text-sm text-gray-400 leading-relaxed">
                      {shapData.description}
                    </p>
                  </div>
                </div>

                {/* Graphique SHAP */}
                <div className="bg-[#0a0a0a] rounded-xl p-6 mb-6">
                  <ShapChart data={shapData.chart_data} />
                  
                  {/* Légende */}
                  <div className="flex items-center justify-center gap-6 mt-4 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                      <span className="text-gray-400">{shapData.legend?.positive || "Augmente le prix"}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500"></div>
                      <span className="text-gray-400">{shapData.legend?.negative || "Diminue le prix"}</span>
                    </div>
                  </div>
                </div>

                {/* Top Features */}
                <div>
                  <h5 className="text-sm font-semibold text-gray-400 mb-4 flex items-center gap-2">
                    <BarChart3 size={16} />
                    Caractéristiques les plus importantes
                  </h5>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {shapData.top_features?.map((feat, idx) => (
                      <div key={idx} className="flex items-center gap-3 p-3 bg-[#1a1a1a] rounded-lg border border-white/5">
                        <span className="w-6 h-6 flex items-center justify-center bg-[#c9a227] text-black rounded-full text-xs font-bold">
                          {feat.rank}
                        </span>
                        <span className="text-sm">{feat.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Contenu LIME */}
            {activeXaiTab === 'lime' && limeData?.available && (
              <div className="p-6">
                {/* Titre et description */}
                <div className="flex items-start gap-3 mb-6">
                  <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                    <Lightbulb size={16} className="text-amber-400" />
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold mb-2">{limeData.title}</h4>
                    <p className="text-sm text-gray-400 leading-relaxed">
                      {limeData.description}
                    </p>
                  </div>
                </div>

                {/* Graphique LIME */}
                <div className="bg-[#0a0a0a] rounded-xl p-6 mb-6">
                  <LimeChart data={limeData.chart_data} />
                  
                  {/* Légende */}
                  <div className="flex items-center justify-center gap-6 mt-4 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                      <span className="text-gray-400">{limeData.legend?.positive || "Favorise l'investissement"}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500"></div>
                      <span className="text-gray-400">{limeData.legend?.negative || "Défavorise l'investissement"}</span>
                    </div>
                  </div>
                </div>

                {/* Détails des contributions */}
                <div>
                  <h5 className="text-sm font-semibold text-gray-400 mb-4 flex items-center gap-2">
                    <Info size={16} />
                    Détails des contributions
                  </h5>
                  <div className="space-y-3">
                    {limeData.contributions?.map((contrib, idx) => (
                      <div key={idx} className="flex items-center justify-between p-4 bg-[#1a1a1a] rounded-xl border border-white/5">
                        <div className="flex items-start gap-3">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                            contrib.impact === 'favorable' ? 'bg-green-500/20' : 'bg-red-500/20'
                          }`}>
                            {contrib.impact === 'favorable' ? (
                              <TrendingUp size={16} className="text-green-400" />
                            ) : (
                              <TrendingDown size={16} className="text-red-400" />
                            )}
                          </div>
                          <div>
                            <div className="font-medium mb-1">
                              {contrib.feature} {contrib.value && <span className="text-gray-400">{contrib.value}</span>}
                            </div>
                            <div className="text-sm text-gray-400">{contrib.description}</div>
                          </div>
                        </div>
                        <div className={`font-mono font-bold ${
                          contrib.impact === 'favorable' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {contrib.impact === 'favorable' ? '+' : ''}{contrib.raw_value?.toFixed(4) || contrib.contribution.toFixed(4)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Bouton Nouvelle analyse */}
                <div className="mt-8 flex justify-center">
                  <button 
                    onClick={onBack}
                    className="flex items-center gap-2 px-6 py-3 bg-[#1a1a1a] hover:bg-[#2a2a2a] text-white rounded-xl transition-colors border border-white/10"
                  >
                    <RefreshCw size={18} />
                    Nouvelle analyse
                  </button>
                </div>
              </div>
            )}

            {/* Erreurs */}
            {activeXaiTab === 'shap' && !shapData?.available && (
              <div className="p-8 text-center text-gray-400">
                <AlertCircle size={48} className="mx-auto mb-4 text-red-400" />
                <p>SHAP non disponible: {shapData?.error}</p>
              </div>
            )}
            {activeXaiTab === 'lime' && !limeData?.available && (
              <div className="p-8 text-center text-gray-400">
                <AlertCircle size={48} className="mx-auto mb-4 text-red-400" />
                <p>LIME non disponible: {limeData?.error}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Composant pour le graphique SHAP (barres horizontales)
function ShapChart({ data }) {
  if (!data || data.length === 0) return null;
  
  const maxVal = Math.max(...data.map(d => Math.abs(d.value)));
  
  return (
    <div className="space-y-2">
      {data.map((item, idx) => {
        const isPositive = item.value >= 0;
        const width = (Math.abs(item.value) / maxVal) * 100;
        
        return (
          <div key={idx} className="flex items-center gap-4">
            <div className="w-32 text-right text-sm text-gray-400 truncate">{item.feature}</div>
            <div className="flex-1 h-8 bg-[#1a1a1a] rounded relative overflow-hidden">
              {/* Barre */}
              <div 
                className={`absolute top-0 h-full ${isPositive ? 'left-1/2' : 'right-1/2'}`}
                style={{ 
                  width: `${width / 2}%`,
                  backgroundColor: item.color || (isPositive ? '#22c55e' : '#ef4444')
                }}
              />
              {/* Ligne centrale */}
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/20" />
            </div>
            <div className="w-16 text-right text-sm font-mono">{item.value > 0 ? '+' : ''}{item.value.toFixed(2)}</div>
          </div>
        );
      })}
      {/* Axe X */}
      <div className="flex justify-between text-xs text-gray-500 mt-2 px-[140px]">
        <span>-{maxVal.toFixed(2)}</span>
        <span>0.00</span>
        <span>+{maxVal.toFixed(2)}</span>
      </div>
    </div>
  );
}

// Composant pour le graphique LIME (similaire)
function LimeChart({ data }) {
  if (!data || data.length === 0) return null;
  
  const maxVal = Math.max(...data.map(d => Math.abs(d.value)));
  
  return (
    <div className="space-y-2">
      {data.map((item, idx) => {
        const isPositive = item.value >= 0;
        const width = (Math.abs(item.value) / maxVal) * 100;
        
        return (
          <div key={idx} className="flex items-center gap-4">
            <div className="w-40 text-right text-sm text-gray-400 truncate">{item.feature}</div>
            <div className="flex-1 h-8 bg-[#1a1a1a] rounded relative overflow-hidden">
              <div 
                className={`absolute top-0 h-full ${isPositive ? 'left-1/2' : 'right-1/2'}`}
                style={{ 
                  width: `${width / 2}%`,
                  backgroundColor: item.color || (isPositive ? '#22c55e' : '#ef4444')
                }}
              />
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/20" />
            </div>
            <div className="w-16 text-right text-sm font-mono">{item.value > 0 ? '+' : ''}{item.value.toFixed(3)}</div>
          </div>
        );
      })}
      <div className="flex justify-between text-xs text-gray-500 mt-2 px-[180px]">
        <span>-{maxVal.toFixed(3)}</span>
        <span>0.000</span>
        <span>+{maxVal.toFixed(3)}</span>
      </div>
    </div>
  );
}

// Landing Page
function LandingPage({ onStart }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 relative overflow-hidden bg-[#0a0a0a]">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-gradient-to-b from-amber-500/10 to-transparent rounded-full blur-3xl pointer-events-none" />
      
      <div className="text-center max-w-4xl mx-auto relative z-10">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm mb-8">
          <Watch size={16} />
          <span>Système d'aide à l'investissement</span>
        </div>
        
        <h1 className="text-5xl md:text-7xl font-serif font-bold mb-6 leading-tight text-white">
          Investissez dans les{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-200 to-amber-500">
            montres de luxe
          </span>{' '}
          en toute confiance
        </h1>
        
        <p className="text-gray-400 text-lg md:text-xl max-w-2xl mx-auto mb-12">
          Notre intelligence artificielle analyse les caractéristiques de votre montre pour prédire sa valeur future et vous guider dans votre décision d'investissement.
        </p>
        
        <button 
          onClick={onStart}
          className="group inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-black font-semibold rounded-xl transition-all duration-300 shadow-lg shadow-amber-500/25 hover:shadow-amber-500/40"
        >
          <Watch size={20} />
          <span>Analyser une montre</span>
          <ChevronDown size={20} className="group-hover:translate-y-1 transition-transform" />
        </button>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-20 max-w-4xl mx-auto">
          <FeatureCard 
            icon={<Watch className="text-amber-400" size={24} />}
            title="Machine Learning"
            description="Algorithmes avancés pour des prédictions précises"
          />
          <FeatureCard 
            icon={<TrendingUp className="text-amber-400" size={24} />}
            title="ROI Estimé"
            description="Calcul du retour sur investissement potentiel"
          />
          <FeatureCard 
            icon={<Shield className="text-amber-400" size={24} />}
            title="XAI Explicable"
            description="Comprenez les facteurs influençant la valorisation"
          />
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="p-6 rounded-2xl bg-white/5 border border-white/10 hover:border-amber-500/30 transition-colors text-left">
      <div className="w-12 h-12 rounded-xl bg-amber-500/10 flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-lg font-semibold mb-2 text-white">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );
}

export default App;