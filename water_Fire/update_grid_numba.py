# update_grid.py (高速化修正版)

import numpy as np
import math
import random
# import copy # njitと互換性がないため削除
from numba import njit, prange # numbaとprangeをインポート
# ... (thetasの定義などはそのまま) ...

# GridUpdaterクラスの代わりに、numbaでコンパイル可能な独立した関数として定義します。
# Cellオブジェクトの代わりに、状態、高さ、密度を直接NumPy配列として受け取ります。

# @njit(parallel=True) を追加し、prangeで並列処理を可能にする
@njit(parallel=True)
def update_grid_numba(
    state_grid_in, height_grid, density_grid, infection_time_in, 
    recovery_time, P_h, cell_size_m, 
    # 状態定数を引数として渡す
    GREEN, ACTIVE, BURNED, DILUTED, RIVER, WATER,
    # 物理パラメータを引数として追加
    slope_factor, wind, theta_w, c_1, c_2
):
    # 処理の高速化のため、グリッドをコピーして結果を格納する
    grid_size = state_grid_in.shape[0]
    new_state_grid = np.copy(state_grid_in)
    new_infection_time = np.copy(infection_time_in)
    
    # 必要な定数 (引数から使用するため削除)
    # slope_factor = 0.078
    # wind = 4.166
    # theta_w = 7 * math.pi / 4
    # c_1 = 0.045
    # c_2 = 0.131
    
    # 近傍の相対座標と方向の角度（theta_d）を事前に定義
    # (di, dj, theta_d_index)
    NEIGHBORS_DATA = np.array([
        (-1,  0, 0), (-1,  1, 1), (0,  1, 2), (1,  1, 3),
        (1,  0, 4), (1, -1, 5), (0, -1, 6), (-1, -1, 7)
    ])
    
    # theta_dの値を格納した配列 (N, NE, E, SE, S, SW, W, NW)
    THETA_D_VALUES = np.array([
        0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, 
        math.pi, 5 * math.pi / 4, 3 * math.pi / 2, 7 * math.pi / 4
    ])

    # prange を使用してiのループを並列化
    for i in prange(grid_size):
        for j in prange(grid_size):
            current_state = state_grid_in[i, j]
            P_burn = 0.0
            
            # --- 状態更新ロジック ---

            if current_state == GREEN:
                
                # 密度による係数 P_den
                density = density_grid[i, j]
                if density < 0.25: P_den = -0.6
                elif density < 0.5: P_den = -0.3
                elif density < 0.75: P_den = -0.1
                else: P_den = 0.1
                
                P_veg = 0.0 # 植生の影響は固定
                
                # 8近傍をチェック
                for k in range(8):
                    di, dj, theta_d_index = NEIGHBORS_DATA[k]
                    ni, nj = i + di, j + dj
                    
                    # 境界チェック
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbor_state = state_grid_in[ni, nj]

                        if neighbor_state == ACTIVE:
                            # --- 傾斜計算 ---
                            current_height = height_grid[i, j]
                            neighbor_height = height_grid[ni, nj]
                            
                            if k in [0, 2, 4, 6]: # N, E, S, W
                                distance = cell_size_m
                            else: # 対角方向
                                distance = cell_size_m * math.sqrt(2)

                            gradient = (neighbor_height - current_height) / distance # 隣接セルから現在のセルへの傾斜
                            slope_angle = math.atan(gradient)
                            P_s = math.exp(slope_factor * slope_angle) # 傾斜係数を計算
                            
                            # --- 風の影響計算 --- 
                            theta_d = THETA_D_VALUES[theta_d_index]
                            theta = abs(theta_w - theta_d)
                            P_w = math.exp(c_1 * wind) * math.exp(c_2 * wind * (math.cos(theta) - 1))

                            tau = random.random()
                            if k in [1, 3, 5, 7]: # 対角方向
                                tau = tau * math.sqrt(2)

                            P_burn = P_h * (1 + P_veg) * (1 + P_den) * P_w * P_s
                            
                            if tau < P_burn:
                                new_state_grid[i, j] = ACTIVE
                                new_infection_time[i, j] = 0 # 着火したセルはタイマーをリセット
                                break # 燃え移りが決定したらループを抜ける

            elif current_state == ACTIVE:
                # ACTIVEセルの燃焼時間管理
                recovery_step = 10 * (60 * (density_grid[i, j] ** 1.5)) / (1 + 0.2 * wind) 
                
                new_infection_time[i, j] += 1
                if new_infection_time[i, j] >= recovery_step:
                    new_state_grid[i, j] = BURNED
                    new_infection_time[i, j] = 0 # 燃え尽きたらタイマーをリセット
            
            elif current_state == RIVER or current_state == WATER or current_state == BURNED:
                # これらの状態は、この関数内では変化しない
                pass

    return new_state_grid, new_infection_time

# GridUpdaterクラスは、numba関数を呼び出すためのラッパーとして残します
class GridUpdater:
    def __init__(self, params, config=None):
        self.params = params
        # Configからパラメータを取得、なければデフォルト値 (configがNoneの場合の互換性のため)
        if config:
            self.wind = config.WIND_SPEED
            self.theta_w = config.THETA_W
            self.slope_factor = config.SLOPE_FACTOR
            self.c_1 = config.C_1
            self.c_2 = config.C_2
        else:
            # フォールバック (configが渡されなかった場合)
            self.wind = 4.166
            self.theta_w = 7 * math.pi / 4
            self.slope_factor = 0.078
            self.c_1 = 0.045
            self.c_2 = 0.131

    # CellオブジェクトをNumPy配列に変換し、numba関数を呼び出す
    def update_grid(self, grid, infection_time, get_neighbors, recovery_time, P_h, cell_size_m):
        
        # 既存のCellオブジェクトをNumPy配列に変換
        grid_size = grid.shape[0]
        state_grid_in = np.zeros((grid_size, grid_size), dtype=np.int32)
        height_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        density_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = grid[i, j]
                state_grid_in[i, j] = cell.state
                height_grid[i, j] = cell.height
                density_grid[i, j] = cell.density
                
        # numbaでコンパイルされた高速な関数を呼び出す
        new_state_grid, new_infection_time = update_grid_numba(
            state_grid_in, height_grid, density_grid, infection_time, 
            recovery_time, P_h, cell_size_m,
            # 状態定数を渡す
            self.params['GREEN'], self.params['ACTIVE'], self.params['BURNED'], 
            self.params['DILUTED'], self.params['RIVER'], self.params['WATER'],
            # 物理パラメータを渡す
            self.slope_factor, self.wind, self.theta_w, self.c_1, self.c_2
        )
        
        # 結果をCellオブジェクトグリッドに戻す
        for i in range(grid_size):
            for j in range(grid_size):
                cell = grid[i, j]
                cell.state = new_state_grid[i, j]
                # infection_timeはCellオブジェクトではなく、別配列で管理されているため、そのまま返す
        
        return grid, new_infection_time
    
    # numba関数内で利用するため、active_functionもGridUpdaterクラスから分離
    # 使用しないかもしれませんが、一応保持しておきます
    @staticmethod
    @njit
    def active_function(t, n):
        if t < 0 or t > n or n == 0:
            return 0.0
        
        t_peak = n / 5

        if t <= t_peak:
            return t / t_peak if t_peak > 0 else 1.0
        else:
            return (1 - (t - t_peak) / (n - t_peak)) ** 2
