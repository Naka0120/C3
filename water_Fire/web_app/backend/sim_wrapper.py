
import sys
import os
import numpy as np

# Add parent directory to path to import existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ForestFire_Auto_numba import SIRCellularAutomataInteractive, ACTIVE, BURNED, GREEN, RIVER, WATER

class SimulationManager:
    def __init__(self):
        self.sim = None
        self.config = {
            "grid_size": 200,
            "infection_probability": 0.058,
            "recovery_time": 217,
            "cell_size_m": 10,
            "terrain_mode": "CSV",
            "csv_filepath_elev": r"C:\Users\souta\Work\C3\water_Fire\USA_Fire\elevation_grid.csv",
            "csv_filepath_vege": r"C:\Users\souta\Work\C3\water_Fire\USA_Fire\vegetation_grid.csv",
            "csv_filepath_ign": r"C:\Users\souta\Work\C3\water_Fire\USA_Fire\ignition_synced_wide.csv"
        }
        self.initialize()

    def initialize(self):
        print("Initializing Simulation...")
        self.sim = SIRCellularAutomataInteractive(**self.config)
        # Initial Ignition (Manual point if needed, otherwise dynamic)
        # self.sim.grid[112, 103].state = ACTIVE

    def step(self):
        if not self.sim:
            return
        
        t = self.sim.current_step
        
        # Dynamic Ignition Check (Manually invoking logic from simulate_interactive)
        if hasattr(self.sim, 'ignition_events') and t in self.sim.ignition_events:
            for (r, c) in self.sim.ignition_events[t]:
                if self.sim.grid[r, c].state != ACTIVE and self.sim.grid[r, c].state != BURNED and self.sim.grid[r, c].state != RIVER:
                        self.sim.grid[r, c].state = ACTIVE
                        self.sim.state_grid[r, c] = ACTIVE
                        self.sim.infection_time[r, c] = 0
        
        self.sim.update_grid()
        self.sim.current_step += 1

    def get_grid_state(self):
        # Return flattened grid for easy JSON transfer
        # To optimize, we could return bytes, but JSON is safer for start
        return self.sim.state_grid.tolist()

    def get_stats(self):
        return {
            "step": self.sim.current_step,
            "active_count": int(np.sum(self.sim.state_grid == ACTIVE)),
            "burned_count": int(np.sum(self.sim.state_grid == BURNED)),
            "water_mode_available": self.sim.active_threshold_reached
        }

    def place_water(self, x, y, radius=1):
        # Frontend works in canvas coordinates (pixels), assuming 1:1 map to grid_size
        # x: col, y: row
        r, c = int(y), int(x)
        if 0 <= r < self.sim.grid_size and 0 <= c < self.sim.grid_size:
            # Reusing place_water logic simplified
             for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = r + di, c + dj
                    if 0 <= ni < self.sim.grid_size and 0 <= nj < self.sim.grid_size:
                        prev = self.sim.grid[ni, nj].state
                        is_eligible = False
                        
                        if prev == ACTIVE:
                             t_inf = self.sim.infection_time[ni, nj]
                             if t_inf <= 0.2 * self.sim.recovery_time:
                                 is_eligible = True
                        elif prev in [GREEN, BURNED]: # Allow placing on non-river
                             is_eligible = True
                             
                        if is_eligible:
                            self.sim.grid[ni, nj].state = WATER
                            self.sim.state_grid[ni, nj] = WATER # Important for get_grid_state
                            self.sim.water_timer[ni, nj] = 0
                            self.sim.water_prev_state[ni, nj] = prev
