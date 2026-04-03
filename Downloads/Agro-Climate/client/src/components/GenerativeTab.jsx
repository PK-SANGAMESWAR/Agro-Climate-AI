import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Zap, Hexagon, Image as ImageIcon, Sliders, RefreshCcw, Download } from 'lucide-react';

const GenerativeTab = () => {
  const [latentBatch, setLatentBatch] = useState(Array.from({ length: 8 }, (_, i) => i));
  const [isGenerating, setIsGenerating] = useState(false);
  const [latentVector, setLatentVector] = useState(0.5);

  const generateNew = () => {
    setIsGenerating(true);
    setTimeout(() => {
      setLatentBatch(prev => prev.map(v => v + 1));
      setIsGenerating(false);
    }, 2000);
  };

  return (
    <div className="space-y-10">
      {/* Controls Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/5 border border-white/10 p-8 rounded-3xl backdrop-blur-xl flex flex-col md:flex-row items-center gap-8 justify-between"
      >
        <div className="flex items-center gap-6">
          <div className="p-4 bg-purple-500/10 rounded-2xl text-purple-400">
            <Zap size={32} />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-white">Plant Studio GAN</h3>
            <p className="text-slate-400">Synthetic biological generation using DCGAN Generator</p>
          </div>
        </div>

        <div className="flex gap-4 w-full md:w-auto">
          <div className="flex-1 md:w-64">
            <div className="flex justify-between mb-2">
              <label className="text-xs font-bold uppercase tracking-widest text-slate-500">Latent Shift</label>
              <span className="text-purple-400 font-mono text-sm">{latentVector.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="0" max="1" step="0.01"
              value={latentVector}
              onChange={e => setLatentVector(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-white/10 rounded-full accent-purple-500"
            />
          </div>
          <button 
            onClick={generateNew}
            disabled={isGenerating}
            className="px-8 py-4 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-xl transition-all shadow-lg shadow-purple-500/20 flex items-center gap-2"
          >
            {isGenerating ? 'Synthesizing...' : 'Re-Generate'}
            <RefreshCcw size={18} className={isGenerating ? 'animate-spin' : ''} />
          </button>
        </div>
      </motion.div>

      {/* Grid Results */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {latentBatch.map((id, i) => (
          <motion.div
            key={id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.05 }}
            whileHover={{ y: -5, scale: 1.02 }}
            className="group relative bg-white/5 border border-white/10 aspect-square rounded-2xl overflow-hidden cursor-pointer backdrop-blur-sm"
          >
            {/* Using a placeholder for generative models in the demo UI */}
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-blue-500/10 flex items-center justify-center">
               <ImageIcon size={48} className="text-white/10" />
            </div>
            
            {/* Show real images once trained if available */}
            <div className={`absolute inset-0 transition-opacity duration-1000 ${isGenerating ? 'opacity-0' : 'opacity-100'}`}>
              <img 
                src={`https://picsum.photos/seed/agro${id}/400`} 
                alt="Generated Plant" 
                className="w-full h-full object-cover"
              />
            </div>

            <div className="absolute inset-0 bg-gradient-to-t from-slate-900/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-6 flex flex-col justify-end">
              <div className="flex justify-between items-center">
                <div>
                  <div className="text-xs font-bold uppercase text-purple-400">Sample #{id * 100}</div>
                  <div className="text-white font-bold">GAN Synthesis</div>
                </div>
                <button className="p-2 bg-white/10 hover:bg-white rounded-lg text-white hover:text-slate-900 transition-all">
                  <Download size={16} />
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white/5 border border-white/10 p-8 rounded-3xl backdrop-blur-xl">
          <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Sliders size={20} className="text-slate-400" />
            Feature Map Analysis
          </h4>
          <div className="h-64 flex items-center justify-center border border-white/5 rounded-2xl bg-white/5 font-mono text-xs text-slate-500">
            {/* Latent Space Visualizer Graphic */}
            <div className="grid grid-cols-8 gap-2">
              {Array.from({length: 64}).map((_, i) => (
                <div key={i} className="w-4 h-4 rounded-sm bg-purple-500/20" style={{ opacity: Math.random() }} />
              ))}
            </div>
          </div>
        </div>
        
        <div className="bg-white/5 border border-white/10 p-8 rounded-3xl backdrop-blur-xl">
          <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Hexagon size={20} className="text-slate-400" />
            Architectural Stats
          </h4>
          <div className="space-y-4">
            {[
              { label: 'Network Type', val: 'DCGAN' },
              { label: 'Latent Dimension', val: '100' },
              { label: 'Activation', val: 'Leaky ReLU / Tanh' },
              { label: 'Stability Index', val: '0.88' },
            ].map(item => (
              <div key={item.label} className="flex justify-between items-center py-3 border-b border-white/5">
                <span className="text-slate-400">{item.label}</span>
                <span className="text-white font-bold">{item.val}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GenerativeTab;
