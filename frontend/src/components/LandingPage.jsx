import React from 'react';
import { Watch, Brain, TrendingUp, Shield, ChevronDown } from 'lucide-react';

export default function LandingPage({ onStart }) {
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
            icon={<Brain className="text-amber-400" size={24} />}
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