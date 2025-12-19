
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
        # Map raw states to visualization indices to match frontend expectation
        # GREEN (0) -> 0-3 based on density
        # ACTIVE (1) -> 4-6 based on infection time
        # BURNED (2) -> 7
        # RIVER (4) -> 8
        # WATER (5) -> 9
        
        # Create a copy to store visualization indices
        vis_grid = np.zeros_like(self.sim.state_grid, dtype=int)
        
        # GREEN
        is_green = self.sim.state_grid == self.sim.params['GREEN']
        if np.any(is_green):
            density_on_green = self.sim.density_grid[is_green]
            conditions = [
                density_on_green < 0.25,
                density_on_green < 0.5,
                density_on_green < 0.75,
                density_on_green >= 0.75
            ]
            choices = [0, 1, 2, 3]
            vis_grid[is_green] = np.select(conditions, choices)
            
        # ACTIVE
        is_active = self.sim.state_grid == self.sim.params['ACTIVE']
        if np.any(is_active):
            # We need infection time for these cells
            t_inf = self.sim.infection_time[is_active]
            n = self.sim.recovery_time
            # Using logical indexing for vectorization
            # t <= 0.2*n -> 6 (Bright Red - Extinguishable)
            # t > 0.8*n -> 4 (Dark Red - Dying)
            # Else -> 5 (Red)
            
            c4 = t_inf > 0.8 * n
            c6 = t_inf <= 0.2 * n
            # Default to 5, then override
            active_vals = np.full(t_inf.shape, 5)
            active_vals[c4] = 4
            active_vals[c6] = 6
            vis_grid[is_active] = active_vals

        # BURNED
        vis_grid[self.sim.state_grid == self.sim.params['BURNED']] = 7
        
        # RIVER
        vis_grid[self.sim.state_grid == self.sim.params['RIVER']] = 8
        
        # WATER
        vis_grid[self.sim.state_grid == self.sim.params['WATER']] = 9
        
        return vis_grid.tolist()

    def get_stats(self):
        return {
            "step": self.sim.current_step,
            "active_count": int(np.sum(self.sim.state_grid == self.sim.params['ACTIVE'])),
            "burned_count": int(np.sum(self.sim.state_grid == self.sim.params['BURNED'])),
            "water_mode_available": self.sim.active_threshold_reached
        }

    def get_elevation_grid(self):
        # Normalize to 0-255 or return raw?
        # Let's return raw and handle coloring in frontend or backend
        # Actually returning raw is better for flexibility
        return self.sim.height_grid.tolist()

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
                        
                        if prev == self.sim.params['ACTIVE']:
                             t_inf = self.sim.infection_time[ni, nj]
                             if t_inf <= 0.2 * self.sim.recovery_time:
                                 is_eligible = True
                        elif prev in [self.sim.params['GREEN'], self.sim.params['BURNED']]: # Allow placing on non-river
                             is_eligible = True
                             
                        if is_eligible:
                            self.sim.grid[ni, nj].state = self.sim.params['WATER']
                            self.sim.state_grid[ni, nj] = self.sim.params['WATER'] # Important for get_grid_state
                            self.sim.water_timer[ni, nj] = 0
                            self.sim.water_prev_state[ni, nj] = prev
