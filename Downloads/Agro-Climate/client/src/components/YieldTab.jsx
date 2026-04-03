import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { CloudRain, BarChart3, Wind, Thermometer, MapPin, Search, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const YieldTab = () => {
  const [params, setParams] = useState({
    county_id: 1,
    gdd: 1200,
    ppt: 850,
  });

  const [prediction, setPrediction] = useState(null);

  // Mock historical data for the chart
  const data = [
    { year: '2018', yield: 165 },
    { year: '2019', yield: 158 },
    { year: '2020', yield: 172 },
    { year: '2021', yield: 168 },
    { year: '2022', yield: 180 },
    { year: '2023', yield: 175 },
    { year: '2024 (P)', yield: 184 },
  ];

  const handlePredict = () => {
    // Simulated prediction logic
    setPrediction({
      val: 184.2,
      conf: 0.94,
      trend: '+5.2%',
    });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Input Panel */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="lg:col-span-1 bg-white/5 border border-white/10 p-8 rounded-3xl backdrop-blur-xl h-fit"
      >
        <div className="flex items-center gap-3 mb-8 text-emerald-400">
          <MapPin size={24} />
          <h3 className="text-xl font-bold text-white tracking-tight">Environmental Parameters</h3>
        </div>

        <div className="space-y-6">
          <div>
            <label className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2 block">County ID</label>
            <input 
              type="number" 
              className="w-full bg-white/5 border border-white/10 p-4 rounded-xl text-white outline-none focus:border-emerald-500 transition-colors"
              value={params.county_id}
              onChange={e => setParams({...params, county_id: e.target.value})}
            />
          </div>
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-xs font-bold uppercase tracking-widest text-slate-500">Cumulative GDD</label>
              <span className="text-emerald-400 font-mono text-sm">{params.gdd}</span>
            </div>
            <input 
              type="range" min="0" max="3000"
              className="w-full accent-emerald-500 bg-white/10 h-1.5 rounded-full"
              value={params.gdd}
              onChange={e => setParams({...params, gdd: e.target.value})}
            />
          </div>
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-xs font-bold uppercase tracking-widest text-slate-500">Total Precipitation (mm)</label>
              <span className="text-blue-400 font-mono text-sm">{params.ppt}</span>
            </div>
            <input 
              type="range" min="0" max="2000"
              className="w-full accent-blue-500 bg-white/10 h-1.5 rounded-full"
              value={params.ppt}
              onChange={e => setParams({...params, ppt: e.target.value})}
            />
          </div>
          
          <button 
            onClick={handlePredict}
            className="w-full py-4 bg-gradient-to-r from-emerald-500 to-blue-600 text-slate-900 font-black rounded-xl uppercase tracking-widest hover:scale-[1.02] active:scale-[0.98] transition-all shadow-xl shadow-emerald-500/10 mt-6"
          >
            Predict Harvest Yield
          </button>
        </div>
      </motion.div>

      {/* Prediction & Visuals */}
      <div className="lg:col-span-2 space-y-8">
        {/* Results Card */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/5 border border-white/10 p-8 rounded-3xl backdrop-blur-xl relative overflow-hidden"
        >
          <div className="absolute top-0 right-0 p-8 text-white/5 select-none">
            <CloudRain size={120} />
          </div>

          <header className="flex justify-between items-center mb-10">
            <div>
              <h4 className="text-sm font-bold text-emerald-500 uppercase tracking-widest mb-1">Forecast Analysis</h4>
              <h3 className="text-3xl font-black text-white">Projected Yield 2024</h3>
            </div>
            {prediction && (
              <div className="flex items-center gap-2 bg-emerald-500/20 text-emerald-400 px-4 py-2 rounded-full font-bold">
                <TrendingUp size={18} /> {prediction.trend}
              </div>
            )}
          </header>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
            <div className="p-6 bg-white/5 rounded-2xl border border-white/5">
              <div className="text-slate-500 text-xs font-bold uppercase mb-1">LTSM Estimate</div>
              <div className="text-3xl font-bold text-white">{prediction?.val || '---'}<span className="text-sm font-normal text-slate-400 ml-2">bu/ac</span></div>
            </div>
            <div className="p-6 bg-white/5 rounded-2xl border border-white/5">
              <div className="text-slate-500 text-xs font-bold uppercase mb-1">Model R²</div>
              <div className="text-3xl font-bold text-white">{prediction?.conf || '0.92'}</div>
            </div>
            <div className="p-6 bg-white/5 rounded-2xl border border-white/5">
              <div className="text-slate-500 text-xs font-bold uppercase mb-1">Risk Factor</div>
              <div className="text-3xl font-bold text-white text-emerald-500">Low</div>
            </div>
          </div>

          <div className="h-[300px] w-full -ml-2 md:ml-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                <XAxis dataKey="year" stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #ffffff10', borderRadius: '12px' }}
                  itemStyle={{ color: '#10b981' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="yield" 
                  stroke="#10b981" 
                  strokeWidth={4} 
                  dot={{ r: 6, fill: '#10b981', strokeWidth: 0 }}
                  activeDot={{ r: 8, strokeWidth: 0 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default YieldTab;
