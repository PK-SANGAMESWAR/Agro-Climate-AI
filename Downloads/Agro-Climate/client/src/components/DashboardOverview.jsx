import React from 'react';
import { motion } from 'framer-motion';
import { Leaf, CloudRain, Zap, BarChart3, ArrowUpRight, Menu } from 'lucide-react';

const DashboardOverview = ({ onNavigate }) => {
  const cards = [
    {
      id: 'classification',
      title: 'Plant Species DL',
      desc: 'Identify weed vs. crop species using deep CNN architectures with 95%+ accuracy.',
      icon: Leaf,
      color: 'from-emerald-500/20 to-emerald-400/5',
      accent: 'text-emerald-400',
      metric: '98.2% Accuracy',
    },
    {
      id: 'yield',
      title: 'Yield Forecasting',
      desc: 'Temporal sequence modeling with LSTM/GRU for multi-year agricultural yield prediction.',
      icon: CloudRain,
      color: 'from-blue-500/20 to-blue-400/5',
      accent: 'text-blue-400',
      metric: '0.92 R²',
    },
    {
      id: 'generative',
      title: 'Generative Studio',
      desc: 'Explore GAN and Autoencoder latent spaces to generate synthetic plant biology.',
      icon: Zap,
      color: 'from-purple-500/20 to-purple-400/5',
      accent: 'text-purple-400',
      metric: 'Latent Space Map',
    }
  ];

  return (
    <div className="space-y-12">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { label: 'Models Trained', val: '8', icon: Zap },
          { label: 'Dataset Size', val: '92.8MB', icon: BarChart3 },
          { label: 'GPU Utilization', val: 'High', icon: Leaf },
          { label: 'Project Phase', val: 'Deployment', icon: ArrowUpRight },
        ].map((stat, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="bg-white/5 border border-white/10 p-6 rounded-2xl backdrop-blur-md"
          >
            <div className="flex justify-between items-start mb-4">
              <div className="p-2 bg-white/5 rounded-lg text-slate-400">
                <stat.icon size={20} />
              </div>
              <span className="text-emerald-500 text-xs font-bold font-mono">+12.5%</span>
            </div>
            <h4 className="text-slate-400 text-sm font-medium">{stat.label}</h4>
            <div className="text-2xl font-bold text-white mt-1">{stat.val}</div>
          </motion.div>
        ))}
      </div>

      {/* Main Project Cards */}
      <section>
        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
          <Menu size={20} className="text-slate-400" />
          Core Intelligence Modules
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {cards.map((card, i) => (
            <motion.div
              key={card.id}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.1 }}
              whileHover={{ y: -5 }}
              className={`relative overflow-hidden group bg-gradient-to-br ${card.color} border border-white/10 p-8 rounded-3xl backdrop-blur-xl cursor-pointer`}
              onClick={() => onNavigate(card.id)}
            >
              <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
                <card.icon size={120} />
              </div>
              
              <div className={`p-4 bg-white/10 w-fit rounded-2xl mb-6 ${card.accent}`}>
                <card.icon size={32} />
              </div>
              
              <h4 className="text-2xl font-bold text-white mb-3 group-hover:text-emerald-400 transition-colors">
                {card.title}
              </h4>
              <p className="text-slate-400 leading-relaxed mb-8">
                {card.desc}
              </p>
              
              <div className="flex items-center justify-between mt-auto">
                <span className={`text-sm font-bold font-mono px-3 py-1 bg-white/5 rounded-full ${card.accent}`}>
                  {card.metric}
                </span>
                <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center group-hover:bg-emerald-500 group-hover:text-slate-900 transition-all">
                  <ArrowUpRight size={20} />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Integration Banner */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.7 }}
        className="bg-gradient-to-r from-emerald-600/20 via-blue-600/20 to-purple-600/20 border border-white/10 p-10 rounded-3xl backdrop-blur-2xl flex flex-col md:flex-row items-center justify-between gap-8"
      >
        <div className="max-w-2xl">
          <h3 className="text-2xl font-bold text-white mb-4">End-to-End System Performance</h3>
          <p className="text-slate-400 text-lg leading-relaxed">
            All modules are integrated into a single unified pipeline, from raw multi-modal data ingestion to production-ready inference. Designed for field researchers and climate analysts.
          </p>
        </div>
        <div className="flex gap-4">
          <button 
            onClick={() => onNavigate('yield')}
            className="px-6 py-3 bg-white text-slate-900 font-bold rounded-xl hover:bg-emerald-400 transition-colors flex items-center gap-2"
          >
            View Analytics <BarChart3 size={18} />
          </button>
          <button 
            onClick={() => window.location.href = 'mailto:team@agroclimate.ai'}
            className="px-6 py-3 bg-white/5 border border-white/10 text-white font-bold rounded-xl hover:bg-white/10 transition-colors"
          >
            Contact Team
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default DashboardOverview;
