# update_grid.py (修正版)
import numpy as np
import math
import random
import copy
from numba import njit, prange

thetas = [[45, 0, 45],
          [90, 0, 90],
          [135, 180, 135]]

class GridUpdater:
    def __init__(self, params):
        self.params = params

    def active_function(t, n):
        if t < 0 or t > n:
            return 0.0
        
        t_peak = n / 5

        if t <= t_peak:
            return t / t_peak
        else:
            return (1- (t-t_peak) / (n - t_peak)) **2
        
    # def calc_pw(theta):
    #     c_1 = 0.045
    #     c_2 = 0.131
    #     V = 10
    #     t = math.radians(theta)
    #     ft = math.exp(V*c_2*(math.cos(t)-1))
    #     return math.exp(c_1*V)*ft
    
    # def get_wind():
    #     wind_matrix = [[0 for col in [0,1,2]] for row in [0,1,2]]
    #     for row in [0,1,2]:
    #         for col in [0,1,2]:
    #             wind_matrix[row][col] = calc_pw(thetas[row][col])
    #     wind_matrix[1][1] = 0
    #     if wind == False:
    #         wind_matrix = [[1 for col in [0,1,2]] for row in [0,1,2]]
    #     return wind_matrix

    def update_grid(self, grid, infection_time, get_neighbors, recovery_time, P_h, cell_size_m):
        new_grid = copy.deepcopy(grid)
        grid_size = grid.shape[0]
        slope_factor = 0.078 # 傾斜の影響度を調整する係数(論文値:0.078)
        wind = 4.166 # 風速（ここでは固定値とする）
        theta_w = 5*math.pi /4 # 風向き(北風:0, 東風:π/2, 南風:π, 西風:3π/2, 北東:π/4, 南東:3π/4, 南西:5π/4, 北西:7π/4)
        P_burn = 0.0
        # 風の影響用パラメータ
        c_1 = 0.045
        c_2 = 0.131
        V = 10
        # theta = 0.0 # 風向き(北風:0, 東風:π/2, 南風:π, 西風:3π/2, 北東:π/4, 南東:3π/4, 南西:5π/4, 北西:7π/4)

        for i in range(grid_size):
            for j in range(grid_size):
                cell = new_grid[i, j]
                # get_neighborsは更新前のgridを参照するので、このままでOK
                neighbors = get_neighbors(i, j)
                recovery_step = 10*(60*(cell.density **1.5))/(1+0.2*wind) 

                if cell.state in [self.params['GREEN']]:
                    # 密度による係数
                    if cell.density < 0.25: P_den = -0.6
                    elif cell.density < 0.5: P_den = -0.3
                    elif cell.density < 0.75: P_den = -0.1
                    else: P_den = 0.1
                    
                    P_veg = 0.0 # 植生の種類(野原:-0.3, 広葉樹:0, 針葉樹:0.3)
                    # P_w = 1.0 # 風速の影響

                    for neighbor, direction in neighbors:
                        if neighbor.state == self.params['ACTIVE']:
                            # --- 動的な傾斜計算 ---
                            current_height = cell.height
                            neighbor_height = neighbor.height
                            
                            if direction in ["North", "South", "East", "West"]:
                                distance = cell_size_m
                            else: # 対角方向
                                distance = cell_size_m * math.sqrt(2)

                            gradient = (current_height - neighbor_height) / distance
                            slope_angle = math.atan(gradient)
                            P_s = math.exp(slope_factor * slope_angle) # 傾斜係数を計算
                            # --- 傾斜計算ここまで ---

                            # --- 風の影響計算 --- 
                            if direction == "North": theta_d = 0
                            elif direction == "North-East": theta_d = math.pi /4
                            elif direction == "East": theta_d = math.pi /2
                            elif direction == "South-East": theta_d = 3*math.pi /4
                            elif direction == "South": theta_d = math.pi
                            elif direction == "South-West": theta_d = 5*math.pi /4
                            elif direction == "West": theta_d = 3*math.pi /2
                            elif direction == "North-West": theta_d = 7*math.pi /4
                            else: theta_d = 0.0

                            theta = abs(theta_w - theta_d)
                            P_w = math.exp(c_1 * wind) * math.exp(c_2*wind*(math.cos(theta) -1))


                            tau = random.random()
                            if direction in ["North-East", "South-West", "North-West", "South-East"]:
                                tau = tau * math.sqrt(2)

                            P_burn = P_h * (1 + P_veg) * (1 + P_den) * P_w * P_s
                            if tau < P_burn:
                                cell.state = self.params['ACTIVE']
                                break
                
                elif cell.state == self.params['ACTIVE']:
                    infection_time[i, j] += 1
                    if infection_time[i, j] >= recovery_step:
                        cell.state = self.params['BURNED']
                
                elif cell.state == self.params['RIVER']:
                    continue
                    
        return new_grid, infection_time