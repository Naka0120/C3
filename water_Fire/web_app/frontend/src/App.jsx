
import React, { useState, useEffect, useRef } from 'react';
import FireGrid from './components/FireGrid';
import { Play, Pause, RefreshCw, Droplets } from 'lucide-react';
import axios from 'axios';

// API Base URL
const API_URL = 'http://localhost:8000';

function App() {
  const [gridData, setGridData] = useState([]);
  const [stats, setStats] = useState({ step: 0, active_count: 0, burned_count: 0, water_mode_available: false });
  const [isRunning, setIsRunning] = useState(false);
  const [isWaterMode, setIsWaterMode] = useState(false);
  const [loading, setLoading] = useState(true);

  // Simulation Loop ref
  const intervalRef = useRef(null);

  const fetchState = async () => {
    try {
      const res = await axios.get(`${API_URL}/state`);
      setGridData(res.data.grid);
      setStats(res.data.stats);
      setLoading(false);
    } catch (err) {
      console.error("Error fetching state:", err);
    }
  };

  const stepSimulation = async () => {
    try {
      // Advance 2 steps to match average animation speed
      const res = await axios.post(`${API_URL}/step`, { steps: 2 });
      setGridData(res.data.grid);
      setStats(res.data.stats);
    } catch (err) {
      console.error("Error stepping simulation:", err);
    }
  };

  const resetSimulation = async () => {
    try {
      await axios.post(`${API_URL}/reset`);
      fetchState();
      setIsRunning(false);
    } catch (err) {
      console.error("Error reseting simulation:", err);
    }
  };

  const toggleRun = () => {
    setIsRunning(!isRunning);
  };

  const toggleWaterMode = () => {
    if (stats.water_mode_available) {
      setIsWaterMode(!isWaterMode);
    }
  };

  // Loop effect
  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(stepSimulation, 200); // 100ms interval
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [isRunning]);

  // Initial fetch
  useEffect(() => {
    resetSimulation();
  }, []);

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-4 font-sans">
      <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-orange-500 to-red-600 bg-clip-text text-transparent">
        Forest Fire Simulation
      </h1>

      <div className="flex gap-8 w-full max-w-6xl">
        {/* Main Simulation View */}
        <div className="flex-1 bg-slate-800 rounded-xl p-4 shadow-2xl border border-slate-700 relative">
          <FireGrid
            gridData={gridData}
            isWaterMode={isWaterMode}
            onPlaceWater={(x, y) => axios.post(`${API_URL}/action/water`, { x, y })}
          />

          {/* Stats Overlay */}
          <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-md p-3 rounded-lg text-sm border border-white/10">
            <div>Step: <span className="font-mono text-blue-300">{stats.step}</span></div>
            <div>Active Fires: <span className="font-mono text-red-500">{stats.active_count}</span></div>
            <div>Burned: <span className="font-mono text-gray-400">{stats.burned_count}</span></div>
          </div>
        </div>

        {/* Controls Panel */}
        <div className="w-64 flex flex-col gap-4">
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 flex flex-col gap-3">
            <h2 className="text-gray-400 text-xs uppercase tracking-wider font-semibold">Controls</h2>

            <button
              onClick={toggleRun}
              className={`flex items-center justify-center gap-2 w-full py-3 rounded-lg font-bold transition-all ${isRunning
                ? 'bg-yellow-600 hover:bg-yellow-500 text-white'
                : 'bg-green-600 hover:bg-green-500 text-white'
                }`}
            >
              {isRunning ? <><Pause size={18} /> Pause</> : <><Play size={18} /> Start</>}
            </button>

            <button
              onClick={resetSimulation}
              className="flex items-center justify-center gap-2 w-full py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white font-medium transition-all"
            >
              <RefreshCw size={16} /> Reset
            </button>
          </div>

          <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 flex flex-col gap-3">
            <h2 className="text-gray-400 text-xs uppercase tracking-wider font-semibold">Actions</h2>

            <button
              onClick={toggleWaterMode}
              disabled={!stats.water_mode_available}
              className={`flex items-center justify-center gap-2 w-full py-3 rounded-lg font-bold transition-all ${isWaterMode
                ? 'bg-blue-500 text-white ring-2 ring-blue-300'
                : stats.water_mode_available
                  ? 'bg-blue-900/50 text-blue-300 hover:bg-blue-900'
                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                }`}
            >
              <Droplets size={18} />
              {isWaterMode ? 'Water Mode ON' : 'Water Mode OFF'}
            </button>

            {!stats.water_mode_available && (
              <p className="text-xs text-slate-500 text-center">
                Available when fires &gt; 50
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
