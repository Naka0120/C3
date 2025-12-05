import matplotlib.pyplot as plt
import numpy as np
import math
import os
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button
# ★★★ ファイル名をupdate_grid_numbaに変更 ★★★
from update_grid_numba import GridUpdater 
# ★★★ -------------------------------- ★★★
from cells import Cell
# GsiFetcherはここではインポートしませんが、必要に応じて追加してください。

# --- 状態定数（WATERを追加） ---
GREEN, ACTIVE, BURNED, DILUTED, RIVER, WATER = 0, 1, 2, 3, 4, 5

class SIRCellularAutomataInteractive:
    # --- UI関連のクラス変数 ---
    is_paused = False
    water_mode_active = False
    active_threshold_reached = False
    is_drawing = False  # ドラッグ中かどうか
    ACTIVE_THRESHOLD = 50  # ACTIVEセルがこの数を超えたら水設置可能
    MAX_WATER_CELLS_PER_DRAG_STEP = 9   # 1回のイベントで設置できる最大セル数 (3x3=9セル)
    is_eligible_for_water = False

    def __init__(self, grid_size=200, infection_probability=0.058, recovery_time=217, cell_size_m=10, 
                 terrain_mode="DUMMY", csv_filepath_elev=None, csv_filepath_vege=None, base_lat=None, base_lon=None):

        self.grid_size = grid_size
        self.infection_probability = infection_probability
        self.recovery_time = recovery_time
        self.cell_size_m = cell_size_m
        self.current_step = 0

        # --- 地形情報の準備 (ダミーのみ) ---
        print(f"--- 地形モード: {terrain_mode} ---")

        if terrain_mode == "API":
            # API処理をスキップしダミー値を設定
            self.height_grid = np.full((grid_size, grid_size), 50.0, dtype=float) 
        
        elif terrain_mode == "CSV":
            # CSV処理をスキップしダミー値を設定
            self.height_grid = np.array([[j*3 for j in range(grid_size)] for i in range(grid_size)], dtype=float)
            self.vegetation_grid = np.full((grid_size, grid_size), 10.0, dtype=float) 
        elif terrain_mode == "DUMMY":
            self.height_grid = np.array([[j for j in range(grid_size)] for i in range(grid_size)], dtype=float)
        else:
             raise ValueError(f"無効な地形モードです: {terrain_mode}。'API', 'CSV', 'DUMMY'のいずれかを選択してください。")
        
        # --- 植生と密度の設定 (ダミー) ---
        center = grid_size // 2
        sigma = grid_size / 4
        x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        distance_sq = (x - center)**2 + (y - center)**2
        self.density_grid = np.exp(-distance_sq / (2 * sigma**2))
        
        # 密度が0.0001以下のセルをRIVER状態に設定
        self.state_grid = np.full((grid_size, grid_size), GREEN, dtype=np.int32)
        river_mask = self.density_grid <= 0.0001
        self.state_grid[river_mask] = RIVER

        self.infection_time = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # --- WATERのライフサイクル管理（タイマーと元状態を保持）---
        self.water_timer = np.zeros((grid_size, grid_size), dtype=np.int32)           # WATER設置からの経過ステップ
        self.water_prev_state = np.full((grid_size, grid_size), -1, dtype=np.int32)   # WATERを置いた時の元状態を保持
        # デフォルトの経過ステップ（2ステップ後に消滅）
        self.WATER_ON_ACTIVE_DURATION = 2   # ACTIVE上のWATERがこのステップ数経過でBURNEDに変化
        self.WATER_ON_GREEN_DURATION = 2   # GREEN上のWATERがこのステップ数経過で再びGREENに戻る
        self.WATER_ON_BURNED_DURATION = 2   # BURNED上のWATERがこのステップ数経過で再びBURNEDに戻る

        # --- Cellオブジェクトグリッドの生成 --- 
        self.grid = np.empty((grid_size, grid_size), dtype=object)
        for i in range(grid_size):
            for j in range(grid_size):
                self.grid[i, j] = Cell(
                    state=self.state_grid[i, j],
                    height=self.height_grid[i, j],
                    density=self.density_grid[i, j]
                )

        # --- GridUpdaterの準備 ---
        self.params = {
            'GREEN': GREEN,
            'ACTIVE': ACTIVE,
            'BURNED': BURNED,
            'DILUTED': DILUTED,
            'RIVER': RIVER,
            'WATER': WATER # 新しい状態を追加
        }
        self.grid_updater = GridUpdater(self.params)

    # active_functionはGridUpdaterクラスの静的メソッドとして定義されているため、ここでは削除

    def get_neighbors(self, i, j):
        """Cellオブジェクトグリッド用の8近傍取得（numba関数では使用されないが、CellオブジェクトからState_gridに状態を反映するために残す）"""
        directions = [
            (-1,  0, "North"), (-1,  1, "North-East"), (0,  1, "East"), (1,  1, "South-East"),
            (1,  0, "South"), (1, -1, "South-West"), (0, -1, "West"), (-1, -1, "North-West")
        ]
        neighbors = []
        for di, dj, dname in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                neighbors.append((self.grid[ni, nj], dname))
        return neighbors

    def update_grid(self):
        # ACTIVEセルの数をチェックし、閾値を超えていたらフラグを立てる
        if not self.active_threshold_reached:
            active_count = np.sum(self.state_grid == ACTIVE)
            if active_count >= self.ACTIVE_THRESHOLD:
                self.active_threshold_reached = True
                print(f"\n🔥🔥🔥 **火災が深刻化: ACTIVEセルが{self.ACTIVE_THRESHOLD}個を超えました！** 🔥🔥🔥")
                print("--- '水設置モード'ボタンが有効化されました。---")

        # Cellオブジェクトグリッドを更新 (numbaで高速化された処理)
        self.grid, self.infection_time = self.grid_updater.update_grid(
            self.grid,
            self.infection_time,
            self.get_neighbors,
            self.recovery_time,
            self.infection_probability,
            self.cell_size_m
        )

        # --- ★★★ 水の消滅ロジックをNumPy操作で高速化 ★★★ ---
        
        # 1. WATERタイマーをインクリメント
        water_mask = (self.state_grid == WATER)
        self.water_timer[water_mask] += 1
        
        # 2. ACTIVE -> BURNED への遷移判定 (2ステップ後)
        active_to_burned_mask = (self.water_prev_state == ACTIVE) & (self.water_timer >= self.WATER_ON_ACTIVE_DURATION)
        
        # 3. GREEN -> GREEN への遷移判定 (2ステップ後)
        green_to_green_mask = (self.water_prev_state == GREEN) & (self.water_timer >= self.WATER_ON_GREEN_DURATION)
        
        # 4. BURNED-> BURNED への遷移判定 (2ステップ後)
        burned_to_burned_mask = (self.water_prev_state == BURNED) & (self.water_timer >= self.WATER_ON_BURNED_DURATION)

        # 5. 状態を更新
        self.state_grid[active_to_burned_mask] = BURNED
        self.state_grid[green_to_green_mask] = GREEN
        self.state_grid[burned_to_burned_mask] = BURNED

        # 6. Cellオブジェクトの状態とタイマーをリセット
        reset_mask = active_to_burned_mask | green_to_green_mask
        self.water_timer[reset_mask] = 0
        self.water_prev_state[reset_mask] = -1
        
        # Cellオブジェクトの状態をstate_gridの最終結果に同期
        i_coords, j_coords = np.where(reset_mask)
        for i, j in zip(i_coords, j_coords):
            self.grid[i, j].state = self.state_grid[i, j]
        
        # --- ★★★ 水の消滅ロジックをNumPy操作で高速化完了 ★★★ ---

        # GridUpdaterで更新された状態をstate_gridに完全に反映（水消滅処理後のCell状態も含む）
        # NOTE: GridUpdater後のCell状態は既に更新されているが、水消滅処理でstate_gridが更新されているため、
        # Cell状態をstate_gridの最終結果に同期させる必要あり (上記5.で完了している)

        # 念のため、GridUpdaterで更新された全セルをstate_gridに反映
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.state_grid[i, j] = self.grid[i, j].state


    # --- UI/インタラクション関連 ---
    def toggle_pause(self, event):
        """シミュレーションの一時停止/再開を切り替える"""
        SIRCellularAutomataInteractive.is_paused = not SIRCellularAutomataInteractive.is_paused
        print(f"シミュレーション: {'一時停止中' if SIRCellularAutomataInteractive.is_paused else '再開'}")
        self.pause_button.label.set_text('Restart' if SIRCellularAutomataInteractive.is_paused else 'Pause')

    def toggle_water_mode(self, event):
        """水設置モードの切り替え"""
        if not self.active_threshold_reached:
            print(f"⚠️ **火災規模が小さすぎます。ACTIVEセルが{self.ACTIVE_THRESHOLD}個を超えるまで待ってください。** ⚠️")
            return
            
        SIRCellularAutomataInteractive.water_mode_active = not SIRCellularAutomataInteractive.water_mode_active
        print(f"水設置モード: {'ON (click/drag)' if SIRCellularAutomataInteractive.water_mode_active else 'OFF'}")
        self.water_button.label.set_text(
            'water (OFF)' if SIRCellularAutomataInteractive.water_mode_active else 'water (ON)'
        )

    # 水設置ロジックを分離
    def place_water(self, event):
        """マウスイベントに基づいてセルに水を設置する"""
        if SIRCellularAutomataInteractive.water_mode_active and event.xdata is not None and event.ydata is not None:
            j = int(round(event.xdata))
            i = int(round(event.ydata))

            if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                # 水を設置（例：3x3の範囲）
                water_placed = False
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            # 樹木、火災、燃え跡のセルのみを水に変換（川はそのまま）
                            prev = self.grid[ni, nj].state
                            if prev == ACTIVE:
                                t = self.infection_time[ni, nj]
                                n = self.recovery_time
                                if t <= 0.2 * n:  # 消火可能なACTIVEセルのみ
                                    is_eligible_for_water = True

                            if is_eligible_for_water:
                                # 元状態を記録してタイマーをリセット
                                self.grid[ni, nj].state = WATER
                                self.water_timer[ni, nj] = 0
                                self.water_prev_state[ni, nj] = prev
                                water_placed = True
                
                # 描画を強制更新
                if water_placed:
                    self.state_grid_update_from_grid()
                    self.visualize(self.current_step, 0, self.ax1)
                    self.fig.canvas.draw_idle()

    def onclick(self, event):
        """マウスボタンが押されたときにドラッグを開始し、水を設置する"""
        if SIRCellularAutomataInteractive.water_mode_active and event.button == 1: # 左クリックのみ
            SIRCellularAutomataInteractive.is_drawing = True
            self.place_water(event) # 押し始めの1点を設置

    def onrelease(self, event):
        """マウスボタンが離されたときにドラッグを終了する"""
        if event.button == 1:
            SIRCellularAutomataInteractive.is_drawing = False

    def on_motion(self, event):
        """ドラッグ中にマウスが移動したときに水を連続設置する"""
        if SIRCellularAutomataInteractive.is_drawing:
            self.place_water(event)


    def state_grid_update_from_grid(self):
        """Cellオブジェクトグリッドからstate_gridを更新する"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.state_grid[i, j] = self.grid[i, j].state
    
    # --- シミュレーション実行 ---
    def simulate_interactive(self, t_end, ax1, ax3):
        self.fig, self.ax1 = plt.subplots(1, 2, figsize=(15, 7))
        self.ax3 = self.ax1[1]
        self.ax1 = self.ax1[0]
        self.fig.suptitle(f'Forest Fire Simulation: Water Threshold={self.ACTIVE_THRESHOLD}', fontsize=16)

        self.visualize_heatmap(self.ax3)
        self.visualize(0, t_end, self.ax1) # 初回描画

        # UIボタンの設定
        ax_pause = self.fig.add_axes([0.1, 0.01, 0.1, 0.04]) # [left, bottom, width, height]
        self.pause_button = Button(ax_pause, 'Pause')
        self.pause_button.on_clicked(self.toggle_pause)

        ax_water = self.fig.add_axes([0.22, 0.01, 0.15, 0.04])
        self.water_button = Button(ax_water, 'Water (ON)')
        self.water_button.on_clicked(self.toggle_water_mode)
        
        # マウスイベントを接続
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)    # 押す
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)  # 離す
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)  # 移動

        
        for t in range(t_end):
            self.current_step = t
            if not SIRCellularAutomataInteractive.is_paused:
                self.update_grid()
                self.visualize(t, t_end, self.ax1)
            
            plt.pause(0.05) # 描画更新間隔
            if not plt.get_fignums(): # ウィンドウが閉じられたら終了
                break

    # --- 可視化 ---
    def visualize(self, time_step, t_end, ax1):
        # 描画の高速化のため、ここではax1.clear()を残す安定版を採用
        ax1.clear() 
        # カラーマップを定義 (GREEN4段階 + ACTIVE3段階 + BURNED + RIVER + WATER)
        cmap = ListedColormap([
            '#e0ffe0', '#80ff80', '#00cc44', '#006622', # GREEN (密度4段階) -> Index 0-3
            '#8B0000', '#DC143C', '#FF5050',           # ACTIVE (燃焼強度3段階) -> Index 4-6
            '#646464',                                 # BURNED -> Index 7
            'deepskyblue',                             # RIVER -> Index 8
            'cyan'                                     # WATER -> Index 9
        ])
        
        color_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # GREENセルの色を密度に応じて設定 (Index 0-3)
        is_green = self.state_grid == GREEN
        density_on_green = self.density_grid[is_green]
        conditions = [
            density_on_green < 0.25,
            density_on_green < 0.5,
            density_on_green < 0.75,
            density_on_green >= 0.75
        ]
        choices = [0, 1, 2, 3]
        color_grid[is_green] = np.select(conditions, choices)
        
        # BURNEDセル (Index 7)
        color_grid[self.state_grid == BURNED] = 7
        # RIVERセル (Index 8)
        color_grid[self.state_grid == RIVER] = 8
        # WATERセル (Index 9)
        color_grid[self.state_grid == WATER] = 9

        # ACTIVEセルの色を燃焼時間に応じて設定 (Index 4-6)
        active_coords = np.argwhere(self.state_grid == ACTIVE)
        for i, j in active_coords:
            t = self.infection_time[i, j]
            n = self.recovery_time
            # 時間によって色を変える
            if t <= 0.2*n:
                color_grid[i, j] = 6 # 明るい赤(消火可能)
            elif t > 0.8*n:
                color_grid[i, j] = 4 # 暗いの赤(もうすぐ消える)
            else:
                color_grid[i, j] = 5 # 中間の赤(消火不可)

        ax1.imshow(color_grid, cmap=cmap, vmin=0, vmax=9)
        ax1.set_title(f"Fire Spread at Time: {time_step + 1} | Water Mode: {'ON' if self.water_mode_active else 'OFF'}")
        ax1.set_xticks([]); ax1.set_yticks([]) # 軸の表示をオフ

    def visualize_heatmap(self, ax):
        """標高のヒートマップを指定された軸(ax)に描画する"""
        ax.clear()
        im = ax.imshow(self.height_grid, cmap='terrain')
        ax.set_title("Elevation Heatmap (m)")
        fig = ax.get_figure()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Elevation (meters)')
        ax.set_xticks([]); ax.set_yticks([]) # 軸の表示をオフ


# --- メイン処理 ---
if __name__ == '__main__':
    
    TERRAIN_MODE = "DUMMY"  # "DUMMY"で動作を確認してください

    sim_params = {
        "grid_size": 200,
        "infection_probability": 0.058,
        "recovery_time": 217,
        "cell_size_m": 10
    }

    # ダミーモード用の設定のみを適用
    all_params = sim_params.copy()
    all_params["terrain_mode"] = TERRAIN_MODE

    # シミュレータのインスタンスを作成
    sir_ca = SIRCellularAutomataInteractive(**all_params)
    
    # 中央に火をつける
    center = sim_params["grid_size"] // 2
    sir_ca.grid[center, center].state = ACTIVE

    # シミュレーション実行 (plt.show()はsimulate_interactive内で制御されます)
    sir_ca.simulate_interactive(500, None, None)