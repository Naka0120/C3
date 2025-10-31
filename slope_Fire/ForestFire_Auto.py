import matplotlib.pyplot as plt
import numpy as np
import math
import os # osモジュールをインポート
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button
from update_grid import GridUpdater
from cells import Cell
from gsi_fetcher import GsiFetcher

# --- 状態定数 ---
GREEN, ACTIVE, BURNED, DILUTED, RIVER = 0, 1, 2, 3, 4

class SIRCellularAutomataSimple:
    def __init__(self, grid_size=200, infection_probability=0.58, recovery_time=217, cell_size_m=10, 
                 terrain_mode="DUMMY", csv_filepath_elev=None, csv_filepath_vege=None, base_lat=None, base_lon=None):

        self.grid_size = grid_size
        self.infection_probability = infection_probability
        self.recovery_time = recovery_time
        self.cell_size_m = cell_size_m
        
        # --- 地形情報の準備（モードに応じて切り替え） ---
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
            # 左右反転してから左に90度回転
            self.height_grid = np.loadtxt(csv_filepath_elev, delimiter=',')
            self.height_grid = np.fliplr(self.height_grid)
            self.height_grid = np.rot90(self.height_grid, k=1)  # 左に90度回転
            if self.height_grid.shape != (grid_size, grid_size):
                raise ValueError(f"CSVのサイズ{self.height_grid.shape}がgrid_size({grid_size},{grid_size})と一致しません。")

            if csv_filepath_vege is None or not os.path.exists(csv_filepath_vege):
                raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_filepath_vege}")
            print(f"植生CSVファイル '{csv_filepath_vege}' を読み込みます...")
            self.vegetation_grid = np.loadtxt(csv_filepath_vege, delimiter=',')
            self.vegetation_grid = np.fliplr(self.vegetation_grid)
            self.vegetation_grid = np.rot90(self.vegetation_grid, k=1)  # 左に90度回転
            if self.height_grid.shape != (grid_size, grid_size):
                raise ValueError(f"CSVのサイズ{self.height_grid.shape}がgrid_size({grid_size},{grid_size})と一致しません。")

        elif terrain_mode == "DUMMY":
            print("ダミーの地形情報を生成します...")
            # ダミー地形1：単純な坂（左から右へ高くなる）
            self.height_grid = np.array([[j for j in range(grid_size)] for i in range(grid_size)], dtype=float)
            # # ダミー地形2：中央に山がある地形
            # center = grid_size // 2; max_height = 500.0
            # x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            # distance = np.sqrt((x - center)**2 + (y - center)**2)
            # self.height_grid = max_height - distance; self.height_grid[self.height_grid < 0] = 0
            # # ダミー地形3：完全に平坦な地面
            # self.height_grid = np.full((grid_size, grid_size), 50.0, dtype=float)
        else:
            raise ValueError(f"無効な地形モードです: {terrain_mode}。'API', 'CSV', 'DUMMY'のいずれかを選択してください。")
        
        print("--------------------------")

        # --- セルとグリッドの初期化 ---
        self.state_grid = np.full((grid_size, grid_size), GREEN, dtype=np.int32)
        
        # 密度をガウス分布で初期化
        center = grid_size // 2
        sigma = grid_size / 4
        x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        distance_sq = (x - center)**2 + (y - center)**2
        # self.density_grid = np.exp(-distance_sq / (2 * sigma**2))
        # 密度を0.5で一定に
        # self.density_grid = np.full((grid_size, grid_size), 0.5, dtype=np.float32)

        self.infection_time = np.zeros((grid_size, grid_size), dtype=np.int32)

        # self.vegetation_gridの値に基づいてself.density_gridを設定
        if terrain_mode == "CSV":
            # 植生タイプに応じた密度マッピング
            veg_to_density = {
                50.0: 0.00001,  # 水域, 市街地
                40.0: 0.1,   # 荒地
                30.0: 0.3,   # 草地
                20.0: 0.6,   # 低木
                10.0: 0.9,   # 樹林
            }
            self.density_grid = np.vectorize(veg_to_density.get)(self.vegetation_grid)
        else:
            self.density_grid = np.exp(-distance_sq / (2 * sigma**2))
        
        # 密度が0.0001以下のセルをRIVER状態に設定
        river_mask = self.density_grid <= 0.0001
        self.state_grid[river_mask] = RIVER

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
            'RIVER': RIVER
        }
        self.grid_updater = GridUpdater(self.params)

    @staticmethod
    def active_function(t, n):
        if t < 0 or t > n or n == 0: # n=0のゼロ除算を回避
            return 0.0
        
        t_peak = n / 5

        if t <= t_peak:
            return t / t_peak if t_peak > 0 else 1.0 # t_peak=0のゼロ除算を回避
        else:
            return (1 - (t - t_peak) / (n - t_peak)) ** 2

    def get_neighbors(self, i, j):
        """Cellオブジェクトグリッド用の8近傍取得（方向名付き）"""
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
        # Cellオブジェクトグリッドを更新
        self.grid, self.infection_time = self.grid_updater.update_grid(
            self.grid,
            self.infection_time,
            self.get_neighbors,
            self.recovery_time,
            self.infection_probability,
            self.cell_size_m
        )
        # 状態グリッドも更新
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.state_grid[i, j] = self.grid[i, j].state

    def simulate(self, t_end, ax1, ax3):
        self.visualize_heatmap(ax3)
        for t in range(t_end):
            self.update_grid() # 引数は不要
            self.visualize(t, t_end, ax1)
    
    # ★★★★★ visualizeメソッドの引数からax3を削除し、グラフ描画機能を追加 ★★★★★
    def visualize(self, time_step, t_end, ax1):
        # 延焼状況の描画 (ax1)
        ax1.clear()
        color_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # カラーマップを定義 (緑4段階 + 赤3段階 + 灰1段階)
        cmap = ListedColormap([
            '#e0ffe0', '#80ff80', '#00cc44', '#006622', # GREEN (密度4段階)
            '#8B0000', '#DC143C', '#FF5050',           # ACTIVE (燃焼強度3段階)
            '#646464',                                 # BURNED
            'deepskyblue'                              # RIVER
        ])
        # GREENセルの色を密度に応じて設定
        is_green = self.state_grid == GREEN
        density_on_green = self.density_grid[is_green]
        
        # np.selectで条件に応じてインデックスを割り当て
        conditions = [
            density_on_green < 0.25,
            density_on_green < 0.5,
            density_on_green < 0.75,
            density_on_green >= 0.75
        ]
        choices = [0, 1, 2, 3]
        color_grid[is_green] = np.select(conditions, choices)
        
        # BURNEDセルの色を設定
        color_grid[self.state_grid == BURNED] = 7
        # RIVERセル
        color_grid[self.state_grid == RIVER] = 8
        # ACTIVEセルの色を燃焼強度に応じて設定
        active_coords = np.argwhere(self.state_grid == ACTIVE)
        for i, j in active_coords:
            t = self.infection_time[i, j]
            n = self.recovery_time
            burn_intensity = self.active_function(t, n)
            if burn_intensity > 0.66:
                color_grid[i, j] = 4 # 暗い赤
            elif burn_intensity > 0.33:
                color_grid[i, j] = 5 # 中間の赤
            else:
                color_grid[i, j] = 6 # 明るい赤
        ax1.imshow(color_grid, cmap=cmap, vmin=0, vmax=8)
        ax1.set_title(f"Fire Spread at Time: {time_step + 1}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.pause(0.01)


    # ★★★★★ ヒートマップ描画用のメソッドを新規追加 ★★★★★
    def visualize_heatmap(self, ax):
        """標高のヒートマップを指定された軸(ax)に描画する"""
        ax.clear()
        # 'terrain' カラーマップを使用して地形を表現
        im = ax.imshow(self.height_grid, cmap='terrain')
        ax.set_title("Elevation Heatmap (m)")
        
        # カラーバーを追加して標高のスケールを示す
        fig = ax.get_figure()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Elevation (meters)')

# --- メイン処理 ---
if __name__ == '__main__':
    
    # ★★★★★ ここで地形モードを切り替え ★★★★★
    TERRAIN_MODE = "CSV"  # "DUMMY", "CSV", "API" のいずれかを選択

    # --- 共通のシミュレーション設定 ---
    sim_params = {
        "grid_size": 200,
        "infection_probability": 0.58,
        "recovery_time": 217,
        "cell_size_m": 10
    }

    # --- 各モード用の設定 ---
    # APIモード用の地理空間設定
    api_params = {
        "base_lat": 34.776,
        "base_lon": 135.252,
    }
    # CSVモード用のファイルパス設定
    csv_params = {
        "csv_filepath_elev": "C:\\Users\\souta\\AppData\\Roaming\\Code\\sorcecode\\C_Cube\\slope_Fire\\Chiri_Fire\\elevation_grid.csv",
        "csv_filepath_vege": "C:\\Users\\souta\\AppData\\Roaming\\Code\\sorcecode\\C_Cube\\slope_Fire\\Chiri_Fire\\vegetation_grid.csv"
    }

    # --- モードに応じてパラメータを組み立て ---
    all_params = sim_params.copy()
    all_params["terrain_mode"] = TERRAIN_MODE

    if TERRAIN_MODE == "API":
        all_params.update(api_params)
    elif TERRAIN_MODE == "CSV":
        all_params.update(csv_params)

    # シミュレータのインスタンスを作成
    sir_ca = SIRCellularAutomataSimple(**all_params)
    
    # 中央に火をつける
    center = sim_params["grid_size"] // 2
    sir_ca.grid[center, center].state = ACTIVE

    # 右下に火をつける
    # sir_ca.state_grid[sim_params["grid_size"] - 1, sim_params["grid_size"] - 1].state = ACTIVE

    # 左上に火をつける
    # sir_ca.state_grid[0, 0] = ACTIVE

    # シミュレーション実行
    # ★★★★★ グラフ描画エリアを3つに分割 ★★★★★
    fig, (p1, p3) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Forest Fire Simulation', fontsize=16)

    sir_ca.simulate(500, p1, p3)
    plt.show()


    # 300セル燃えたら検知

    # チリの火災データ
    # row_index,col_index = セル番号
    # BurnedAreasum = 燃えた面積(m^2)
    # Heightmean = 標高(m)
    # Covermajority = 植生