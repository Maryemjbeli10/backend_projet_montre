import React, { useState } from 'react';
import { 
  Watch, 
  Settings, 
  Package, 
  User, 
  DollarSign, 
  Calendar, 
  Maximize, 
  Droplets, 
  Gem, 
  Store, 
  Activity,
  Clock,
  Shield,
  TrendingUp,
  ArrowRight,
  ChevronDown
} from 'lucide-react';

function App() {
  const [currentView, setCurrentView] = useState('landing');
  const [activeTab, setActiveTab] = useState('general');

  // États du formulaire
  const [formData, setFormData] = useState({
    // Général
    brand: 'Rolex',
    condition: 'Like new & unworn',
    price: 10000,
    shape: 'Round',
    movement: 'Automatic',
    gender: "Men's watch",
    year: 2020,
    waterResistance: 100,
    dialSize: 40,
    boxPapers: 'Original box, original papers',
    // Matériaux
    caseMaterial: 'Steel',
    braceletMaterial: 'Steel',
    crystal: 'Sapphire crystal',
    dialColor: 'Black',
    braceletColor: 'Silver',
    clasp: 'Fold clasp',
    availability: 'Item is in stock',
    // Vendeur
    sellerWatches: 50,
    activeListings: 10,
    sellerRating: 100,
    fastShipping: true,
    trustedSeller: true,
    punctuality: true
  });

  const updateField = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleAnalyze = () => {
    console.log('Analyzing:', formData);
    alert('Analyse lancée ! (À connecter avec votre backend)');
  };

  if (currentView === 'landing') {
    return <LandingPage onStart={() => setCurrentView('form')} />;
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-serif font-bold mb-2">
            Saisissez les caractéristiques de la montre
          </h1>
          <p className="text-gray-400">
            Remplissez le formulaire ci-dessous avec les détails de la montre que vous souhaitez analyser.
          </p>
        </div>

        {/* Card */}
        <div className="bg-[#111] rounded-2xl border border-white/10 overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-white/10">
            <TabButton 
              active={activeTab === 'general'} 
              onClick={() => setActiveTab('general')}
              icon={<Watch size={16} />}
              label="Général"
            />
            <TabButton 
              active={activeTab === 'materials'} 
              onClick={() => setActiveTab('materials')}
              icon={<Gem size={16} />}
              label="Matériaux"
            />
            <TabButton 
              active={activeTab === 'seller'} 
              onClick={() => setActiveTab('seller')}
              icon={<Store size={16} />}
              label="Vendeur"
            />
          </div>

          {/* Content */}
          <div className="p-6">
            {activeTab === 'general' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <SelectField 
                  label="Marque" 
                  value={formData.brand} 
                  onChange={(v) => updateField('brand', v)}
                  options={['Rolex', 'Patek Philippe', 'Audemars Piguet', 'Omega', 'Cartier']}
                  icon={<Watch size={16} />}
                />
                <SelectField 
                  label="Mouvement" 
                  value={formData.movement} 
                  onChange={(v) => updateField('movement', v)}
                  options={['Automatic', 'Manual', 'Quartz']}
                  icon={<Settings size={16} />}
                />
                <SelectField 
                  label="État" 
                  value={formData.condition} 
                  onChange={(v) => updateField('condition', v)}
                  options={['Like new & unworn', 'Excellent', 'Very good', 'Good', 'Fair']}
                  icon={<Package size={16} />}
                />
                <SelectField 
                  label="Genre" 
                  value={formData.gender} 
                  onChange={(v) => updateField('gender', v)}
                  options={["Men's watch", "Women's watch", 'Unisex']}
                  icon={<User size={16} />}
                />
                <NumberField 
                  label="Prix d'achat ($)" 
                  value={formData.price} 
                  onChange={(v) => updateField('price', v)}
                  icon={<DollarSign size={16} />}
                />
                <NumberField 
                  label="Année de production" 
                  value={formData.year} 
                  onChange={(v) => updateField('year', v)}
                  icon={<Calendar size={16} />}
                />
                <SelectField 
                  label="Forme" 
                  value={formData.shape} 
                  onChange={(v) => updateField('shape', v)}
                  options={['Round', 'Square', 'Rectangular', 'Tonnea']}
                  icon={<Maximize size={16} />}
                />
                <NumberField 
                  label="Résistance à l'eau (m)" 
                  value={formData.waterResistance} 
                  onChange={(v) => updateField('waterResistance', v)}
                  icon={<Droplets size={16} />}
                />
                <NumberField 
                  label="Surface du cadran (mm²)" 
                  value={formData.dialSize} 
                  onChange={(v) => updateField('dialSize', v)}
                  icon={<Maximize size={16} />}
                />
                <SelectField 
                  label="Contenu de la livraison" 
                  value={formData.boxPapers} 
                  onChange={(v) => updateField('boxPapers', v)}
                  options={['Original box, original papers', 'Original box only', 'Papers only', 'None']}
                  icon={<Package size={16} />}
                />
              </div>
            )}

            {activeTab === 'materials' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <SelectField 
                  label="Matériau du boîtier" 
                  value={formData.caseMaterial} 
                  onChange={(v) => updateField('caseMaterial', v)}
                  options={['Steel', 'Gold', 'Rose Gold', 'Platinum', 'Titanium', 'Ceramic']}
                  icon={<Gem size={16} />}
                />
                <SelectField 
                  label="Matériau du bracelet" 
                  value={formData.braceletMaterial} 
                  onChange={(v) => updateField('braceletMaterial', v)}
                  options={['Steel', 'Leather', 'Rubber', 'Gold', 'Fabric']}
                  icon={<Gem size={16} />}
                />
                <SelectField 
                  label="Verre" 
                  value={formData.crystal} 
                  onChange={(v) => updateField('crystal', v)}
                  options={['Sapphire crystal', 'Mineral glass', 'Plexiglass']}
                  icon={<Gem size={16} />}
                />
                <SelectField 
                  label="Cadran" 
                  value={formData.dialColor} 
                  onChange={(v) => updateField('dialColor', v)}
                  options={['Black', 'White', 'Blue', 'Green', 'Silver', 'Champagne']}
                  icon={<Maximize size={16} />}
                />
                <SelectField 
                  label="Couleur du bracelet" 
                  value={formData.braceletColor} 
                  onChange={(v) => updateField('braceletColor', v)}
                  options={['Silver', 'Black', 'Brown', 'Blue', 'Gold']}
                  icon={<Settings size={16} />}
                />
                <SelectField 
                  label="Fermoir" 
                  value={formData.clasp} 
                  onChange={(v) => updateField('clasp', v)}
                  options={['Fold clasp', 'Buckle', 'Deployant clasp']}
                  icon={<Settings size={16} />}
                />
                <SelectField 
                  label="Disponibilité" 
                  value={formData.availability} 
                  onChange={(v) => updateField('availability', v)}
                  options={['Item is in stock', 'Pre-order', 'Out of stock']}
                  icon={<Package size={16} />}
                />
              </div>
            )}

            {activeTab === 'seller' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <NumberField 
                    label="Montres vendues par le vendeur" 
                    value={formData.sellerWatches} 
                    onChange={(v) => updateField('sellerWatches', v)}
                    icon={<Store size={16} />}
                  />
                  <NumberField 
                    label="Annonces actives du vendeur" 
                    value={formData.activeListings} 
                    onChange={(v) => updateField('activeListings', v)}
                    icon={<Package size={16} />}
                  />
                  <NumberField 
                    label="Avis du vendeur" 
                    value={formData.sellerRating} 
                    onChange={(v) => updateField('sellerRating', v)}
                    icon={<Activity size={16} />}
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ToggleCard 
                    title="Expédition rapide"
                    description="Le vendeur expédie rapidement"
                    checked={formData.fastShipping}
                    onChange={(v) => updateField('fastShipping', v)}
                    icon={<Clock size={20} />}
                  />
                  <ToggleCard 
                    title="Vendeur de confiance"
                    description="Le vendeur est vérifié"
                    checked={formData.trustedSeller}
                    onChange={(v) => updateField('trustedSeller', v)}
                    icon={<Shield size={20} />}
                  />
                  <ToggleCard 
                    title="Ponctualité"
                    description="Le vendeur est ponctuel"
                    checked={formData.punctuality}
                    onChange={(v) => updateField('punctuality', v)}
                    icon={<Clock size={20} />}
                  />
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button 
              onClick={handleAnalyze}
              className="w-full mt-8 flex items-center justify-center gap-2 py-4 bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-black font-semibold rounded-xl transition-all duration-300"
            >
              <TrendingUp size={20} />
              <span>Analyser l'investissement</span>
              <ArrowRight size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Composants réutilisables

function TabButton({ active, onClick, icon, label }) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-2 py-4 text-sm font-medium transition-colors ${
        active 
          ? 'bg-amber-500/20 text-amber-400 border-b-2 border-amber-500' 
          : 'text-gray-400 hover:text-white'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function SelectField({ label, value, onChange, options, icon }) {
  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 text-sm text-gray-400">
        {icon}
        {label}
      </label>
      <div className="relative">
        <select 
          value={value} 
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none bg-[#1a1a1a] border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-amber-500/50 transition-colors cursor-pointer"
        >
          {options.map(opt => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" size={16} />
      </div>
    </div>
  );
}

function NumberField({ label, value, onChange, icon }) {
  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 text-sm text-gray-400">
        {icon}
        {label}
      </label>
      <input 
        type="number" 
        value={value} 
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-amber-500/50 transition-colors"
      />
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