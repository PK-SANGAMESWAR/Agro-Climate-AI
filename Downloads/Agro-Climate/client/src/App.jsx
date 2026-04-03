import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Leaf, CloudRain, Zap, BarChart3, Upload, Menu, X, ChevronRight } from 'lucide-react';
import ClassificationTab from './components/ClassificationTab';
import YieldTab from './components/YieldTab';
import GenerativeTab from './components/GenerativeTab';
import DashboardOverview from './components/DashboardOverview';

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth >= 768);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) setIsSidebarOpen(false);
      else setIsSidebarOpen(true);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const tabs = [
    { id: 'dashboard', label: 'Overview', icon: BarChart3 },
    { id: 'classification', label: 'Species Classification', icon: Leaf },
    { id: 'yield', label: 'Yield Prediction', icon: CloudRain },
    { id: 'generative', label: 'Plant Studio', icon: Zap },
  ];

  return (
    <div className="min-h-screen bg-[#0a0f1e] text-slate-200 font-sans selection:bg-emerald-500/30">
      {/* Background Orbs */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-600/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
      </div>

      {/* Sidebar */}
      {isMobile && isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}
      <motion.aside 
        initial={false}
        animate={{ width: isMobile ? (isSidebarOpen ? 280 : 0) : (isSidebarOpen ? 280 : 80) }}
        className="fixed left-0 top-0 h-full bg-[#0a0f1e]/95 md:bg-white/5 backdrop-blur-xl md:border-r border-white/10 z-50 overflow-hidden"
      >
        <div className="p-6 mb-8 flex items-center gap-4">
          <div className="w-10 h-10 bg-gradient-to-br from-emerald-400 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-500/20 text-slate-900">
            <Zap size={24} fill="currentColor" />
          </div>
          {isSidebarOpen && (
            <motion.h1 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="font-bold text-xl tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent"
            >
              AgroClimate AI
            </motion.h1>
          )}
        </div>

        <nav className="px-3 space-y-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  if (isMobile) setIsSidebarOpen(false);
                }}
                className={`w-full flex items-center gap-4 p-4 rounded-xl transition-all duration-300 group ${
                  isActive 
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                    : 'hover:bg-white/5 text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon size={22} className={isActive ? 'text-emerald-400' : 'group-hover:translate-x-1 transition-transform'} />
                {isSidebarOpen && <span className="font-medium">{tab.label}</span>}
                {isActive && isSidebarOpen && (
                  <motion.div layoutId="activeBall" className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                )}
              </button>
            );
          })}
        </nav>

        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="absolute bottom-6 left-1/2 -translate-x-1/2 p-3 hover:bg-white/5 rounded-full text-slate-400 transition-colors"
        >
          {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </motion.aside>

      <main className={`transition-all duration-300 ${!isMobile ? (isSidebarOpen ? 'ml-72' : 'ml-20') : 'ml-0'} p-4 sm:p-8 xl:p-12 pb-24 md:pb-12 min-h-screen`}>
        <div className="max-w-7xl mx-auto">
          <header className="mb-8 md:mb-12 flex justify-between items-start">
            <div className="flex items-center gap-4">
              {isMobile && (
                <button 
                  onClick={() => setIsSidebarOpen(true)}
                  className="p-2 bg-white/5 rounded-lg text-slate-400"
                >
                  <Menu size={24} />
                </button>
              )}
              <div>
              <motion.h2 
                key={activeTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-3xl font-bold text-white mb-2"
              >
                {tabs.find(t => t.id === activeTab)?.label}
              </motion.h2>
              <p className="text-slate-400">Deep Learning for Agricultural Intelligence & Climate Resilience</p>
            </div>
            </div>
            
            <div className="flex gap-4">
              <div className="bg-white/5 backdrop-blur-md border border-white/10 px-4 py-2 rounded-lg flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400">System Live</span>
              </div>
            </div>
          </header>

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              {activeTab === 'dashboard' && <DashboardOverview onNavigate={setActiveTab} />}
              {activeTab === 'classification' && <ClassificationTab />}
              {activeTab === 'yield' && <YieldTab />}
              {activeTab === 'generative' && <GenerativeTab />}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
};

export default App;
