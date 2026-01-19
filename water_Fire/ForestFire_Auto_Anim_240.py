# 240ステップ分のアニメーションをGIFとして出力

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import csv
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button
import imageio
# ★★★ ファイル名をupdate_grid_numbaに変更 ★★★
from update_grid_numba import GridUpdater 
from cells import Cell
from gsi_fetcher import GsiFetcher
import config

# --- 状態定数（WATERを追加） ---
GREEN, ACTIVE, BURNED, DILUTED, RIVER, WATER = 0, 1, 2, 3, 4, 5

class SIRCellularAutomataInteractive:
    # --- UI関連のクラス変数 ---
    is_paused = False
    water_mode_active = False
    active_threshold_reached = False
    is_drawing = False  # ドラッグ中かどうか
    ACTIVE_THRESHOLD = config.ACTIVE_THRESHOLD  # ACTIVEセルがこの数を超えたら水設置可能
    MAX_WATER_CELLS_PER_DRAG_STEP = config.MAX_WATER_CELLS_PER_DRAG_STEP   # 1回のイベントで設置できる最大セル数 (3x3=9セル)
    is_eligible_for_water = False

    # --- 座標変換定数 (USA_Fire/L2F_inputmap.csvより導出) ---
    GRID_ORIGIN_X = 521863
    GRID_ORIGIN_Y = 3383228
    GRID_RES = 44.835

    def __init__(self, grid_size=config.GRID_SIZE, infection_probability=config.P_H, recovery_time=config.RECOVERY_TIME, cell_size_m=config.CELL_SIZE_M, 
                 terrain_mode="DUMMY", csv_filepath_elev=None, csv_filepath_vege=None, csv_filepath_ign=None, base_lat=None, base_lon=None):

        self.grid_size = grid_size
        self.infection_probability = infection_probability
        self.recovery_time = recovery_time
        self.cell_size_m = cell_size_m
        self.current_step = 0

        # --- 地形情報の準備 (ダミーのみ) ---
        print(f"--- 地形モード: {terrain_mode} ---")

        if terrain_mode == "API":
            if base_lat is None or base_lon is None:
                raise ValueError("APIモードでは'base_lat'と'base_lon'の指定が必要です。")
            fetcher = GsiFetcher(base_lat, base_lon, grid_size, cell_size_m)
            self.height_grid = fetcher.fetch_elevation_grid()

        elif terrain_mode == "CSV":
            if csv_filepath_elev is None or not os.path.exists(csv_filepath_elev):
                raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_filepath_elev}")
            print(f"標高CSVファイル '{csv_filepath_elev}' を読み込みます...")
            self.height_grid = np.loadtxt(csv_filepath_elev, delimiter=',')
            # 左右反転してから左に90度回転
            # self.height_grid = np.fliplr(self.height_grid)
            # self.height_grid = np.rot90(self.height_grid, k=1)  # 左に90度回転
            # if self.height_grid.shape != (grid_size, grid_size):
            #     raise ValueError(f"CSVのサイズ{self.height_grid.shape}がgrid_size({grid_size},{grid_size})と一致しません。")

            if csv_filepath_vege is None or not os.path.exists(csv_filepath_vege):
                raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_filepath_vege}")
            print(f"植生CSVファイル '{csv_filepath_vege}' を読み込みます...")
            self.vegetation_grid = np.loadtxt(csv_filepath_vege, delimiter=',')
            # self.vegetation_grid = np.fliplr(self.vegetation_grid)
            # self.vegetation_grid = np.rot90(self.vegetation_grid, k=1)  # 左に90度回転
            if self.height_grid.shape != (grid_size, grid_size):
                raise ValueError(f"CSVのサイズ{self.height_grid.shape}がgrid_size({grid_size},{grid_size})と一致しません。")
            if self.vegetation_grid.shape != (grid_size, grid_size):
                raise ValueError(f"CSVのサイズ{self.vegetation_grid.shape}がgrid_size({grid_size},{grid_size})と一致しません。")

            # 着火点データの読み込み
            self.ignition_events = {}
            if csv_filepath_ign and os.path.exists(csv_filepath_ign):
                print(f"着火点CSVファイル '{csv_filepath_ign}'を読み込みます...")
                self.load_ignition_data(csv_filepath_ign)
            else:
                print("着火点CSVファイルが指定されていないか見つかりません。動的着火は無効です。")


        elif terrain_mode == "DUMMY":
            self.height_grid = np.array([[j for j in range(grid_size)] for i in range(grid_size)], dtype=float)
        else:
             raise ValueError(f"無効な地形モードです: {terrain_mode}。'API', 'CSV', 'DUMMY'のいずれかを選択してください。")
        
        # self.vegetation_gridの値に基づいてself.density_gridを設定
        if terrain_mode == "CSV":
            # 植生タイプに応じた密度マッピング
            # チリ用
            # veg_to_density = {
            #     50.0: 0.00001,  # 水域, 市街地
            #     40.0: 0.1,   # 荒地
            #     30.0: 0.3,   # 草地
            #     20.0: 0.6,   # 低木
            #     10.0: 0.9,   # 樹林
            # }

            # USA用
            # 植生CSVの値を0.0〜1.0に正規化してdensity_gridとする
            veg = self.vegetation_grid.astype(float)
            vmin = np.nanmin(veg)
            vmax = np.nanmax(veg)
            if vmax == vmin:
                # 全要素が同じ値の場合は一律0.0に（必要なら別の値に変更）
                self.density_grid = np.zeros_like(veg)
            else:
                self.density_grid = (veg - vmin) / (vmax - vmin)
            # 数値の丸めや範囲外の値対策
            self.density_grid = np.clip(self.density_grid, 0.0, 1.0)
        else:
            center = grid_size // 2
            sigma = grid_size / 4
            x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            distance_sq = (x - center)**2 + (y - center)**2
        
        # 密度が0.0001以下のセルをRIVER状態に設定
        self.state_grid = np.full((grid_size, grid_size), GREEN, dtype=np.int32)
        river_mask = self.density_grid <= 0.0001
        self.state_grid[river_mask] = RIVER

        self.infection_time = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # --- WATERのライフサイクル管理（タイマーと元状態を保持）---
        self.water_timer = np.zeros((grid_size, grid_size), dtype=np.int32)           # WATER設置からの経過ステップ
        self.water_prev_state = np.full((grid_size, grid_size), -1, dtype=np.int32)   # WATERを置いた時の元状態を保持
        # デフォルトの経過ステップ（2ステップ後に消滅）
        self.WATER_ON_ACTIVE_DURATION = config.WATER_ON_ACTIVE_DURATION   # ACTIVE上のWATERがこのステップ数経過でBURNEDに変化
        self.WATER_ON_GREEN_DURATION = config.WATER_ON_GREEN_DURATION   # GREEN上のWATERがこのステップ数経過で再びGREENに戻る
        self.WATER_ON_BURNED_DURATION = config.WATER_ON_BURNED_DURATION   # BURNED上のWATERがこのステップ数経過で再びBURNEDに戻る

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
        self.grid_updater = GridUpdater(self.params, config=config)

    # active_functionはGridUpdaterクラスの静的メソッドとして定義されているため、ここでは削除

    def load_ignition_data(self, filepath):
        """
        ignition_synced_wide.csvを読み込み、タイムステップごとの着火点リストを作成する。
        self.ignition_events = { time_step: [(r, c), ...], ... }
        """
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        elapsed_sec = float(row['Elapsed_Sec'])
                        time_step = int(elapsed_sec / 6) # 6秒を1ステップとして扱う
                        
                        points = []
                        i = 1
                        while True:
                            col_x = f'Ign{i}_X'
                            col_y = f'Ign{i}_Y'
                            if col_x not in row or col_y not in row:
                                break
                            
                            val_x = row[col_x]
                            val_y = row[col_y]
                            
                            if val_x and val_y: # 空でなければ
                                try:
                                    utm_x = float(val_x)
                                    utm_y = float(val_y)
                                    
                                    c = int((utm_x - self.GRID_ORIGIN_X) / self.GRID_RES)
                                    r = int((self.GRID_ORIGIN_Y - utm_y) / self.GRID_RES)
                                    
                                    if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                                        points.append((r, c))
                                except ValueError:
                                    pass
                            i += 1
                        
                        if points:
                            if time_step not in self.ignition_events:
                                self.ignition_events[time_step] = []
                            self.ignition_events[time_step].extend(points)

                    except ValueError:
                        continue
            print(f"着火データを読み込みました: {len(self.ignition_events)} タイムステップ分のイベント")
            
        except Exception as e:
            print(f"着火データの読み込みに失敗しました: {e}")

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
                # print("--- '水設置モード'ボタンが有効化されました。---")

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
    # GIF生成用スクリプトなので、UI関連メソッドは使用しませんが、クラス構造維持のため残します
    def toggle_pause(self, event):
        pass
    def toggle_water_mode(self, event):
        pass
    def place_water(self, event):
        pass
    def onclick(self, event):
        pass
    def onrelease(self, event):
        pass
    def on_motion(self, event):
        pass
    def state_grid_update_from_grid(self):
         for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.state_grid[i, j] = self.grid[i, j].state
    def simulate_interactive(self, t_end, ax1, ax3):
        pass

    def simulate_and_save_gif(self, t_end, filename="forestfire_simulation_usa.gif", fps=10):
        """非インタラクティブでシミュレーションを実行し、GIFファイルとして保存する"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        frames = []
        
        print(f"シミュレーションを開始します (GIF出力、全{t_end}ステップ、間隔=1ステップ)...")
        for t in range(t_end):
            # 動的着火判定
            if hasattr(self, 'ignition_events') and t in self.ignition_events:
                for (r, c) in self.ignition_events[t]:
                    if self.grid[r, c].state != ACTIVE and self.grid[r, c].state != BURNED and self.grid[r, c].state != RIVER:
                            self.grid[r, c].state = ACTIVE
                            self.state_grid[r, c] = ACTIVE # state_gridも同期
                            self.infection_time[r, c] = 0
                            # print(f"Time {t}: Ignition at ({r}, {c})")

            self.update_grid()
            
            # 描画間隔を1ステップごとに設定
            if t % 1 == 0:
                self.visualize(t, t_end, ax)
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)
            
            if t % 10 == 0:
                print(f"Processing step {t}/{t_end}...")

        plt.close(fig)
        print(f"GIF動画を保存中... ({filename})")
        # imageioでGIF保存 (loop=0 for infinite loop)
        imageio.mimsave(filename, frames, fps=fps, loop=0)
        print(f"{filename} を保存しました。")

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
        # ax1.set_title(f"Fire Spread at Time: {time_step + 1}")
        ax1.text(0.5, 1.05, f"Fire Spread at Time: {time_step + 1}", 
                 size=12, ha="center", transform=ax1.transAxes)
        # 固定中心 (121, 89) から指定サイズのウィンドウを設定
        center_r, center_c = 121, 89
        half_size = config.VIEW_WINDOW_SIZE // 2
        
        ax1.set_xlim(center_c - half_size, center_c + half_size)
        ax1.set_ylim(center_r + half_size, center_r - half_size) # Note: inverted Y axis
        ax1.set_xticks([]); ax1.set_yticks([]) # 軸の表示をオフ


# --- メイン処理 ---
if __name__ == '__main__':
    
    TERRAIN_MODE = config.TERRAIN_MODE  # "DUMMY"で動作を確認してください

    # APIモード用の地理空間設定
    api_params = {
        "base_lat": config.API_BASE_LAT,
        "base_lon": config.API_BASE_LON,
    }

    # CSVモード用のファイルパス設定 (ignitionを追加)
    csv_params = {
        "csv_filepath_elev": config.CSV_FILEPATH_ELEV,
        "csv_filepath_vege": config.CSV_FILEPATH_VEGE,
        "csv_filepath_ign": config.CSV_FILEPATH_IGN
    }

    sim_params = {
        "grid_size": config.GRID_SIZE,
        "infection_probability": config.P_H,
        "recovery_time": config.RECOVERY_TIME,
        "cell_size_m": config.CELL_SIZE_M
    }

    all_params = sim_params.copy()
    all_params["terrain_mode"] = TERRAIN_MODE

    if TERRAIN_MODE == "API":
        all_params.update(api_params)
    elif TERRAIN_MODE == "CSV":
        all_params.update(csv_params)

    # シミュレータのインスタンスを作成
    sir_ca = SIRCellularAutomataInteractive(**all_params)
    
    # GIF出力フラグ
    EXPORT_GIF = True

    if EXPORT_GIF:
        # 非インタラクティブで実行してGIFとして保存 (240ステップ)
        sir_ca.simulate_and_save_gif(240, filename="forestfire_simulation_240.gif", fps=config.ANIMATION_FPS)
