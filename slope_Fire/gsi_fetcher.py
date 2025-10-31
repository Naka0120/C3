# gsi_fetcher.py
import numpy as np
import requests
import os
from tqdm import tqdm
import time
import concurrent.futures

class GsiFetcher:
    API_URL = "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php"
    CACHE_FILE = "C:\\Users\\souta\\AppData\\Roaming\\Code\\sorcecode\\C_Cube\\elevation_cache.npy"

    DEG_TO_M_LAT = 111000
    
    def __init__(self, base_lat, base_lon, grid_size, cell_size_m):
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.grid_size = grid_size
        self.cell_size_m = cell_size_m
        self.deg_to_m_lon = self.DEG_TO_M_LAT * np.cos(np.deg2rad(base_lat))

    def _convert_grid_to_lonlat(self, i, j):
        lat = self.base_lat - (i * self.cell_size_m / self.DEG_TO_M_LAT)
        lon = self.base_lon + (j * self.cell_size_m / self.deg_to_m_lon)
        return lon, lat

    def fetch_single_elevation(self, ij):
        i, j = ij
        lon, lat = self._convert_grid_to_lonlat(i, j)
        try:
            params = {'lon': lon, 'lat': lat, 'outtype': 'JSON'}
            response = requests.get(self.API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'elevation' in data:
                return i, j, float(data['elevation'])
            else:
                return i, j, 0.0
        except Exception:
            return i, j, 0.0

    def fetch_elevation_grid(self):
        if os.path.exists(self.CACHE_FILE):
            print(f"キャッシュファイル '{self.CACHE_FILE}' を読み込みます。")
            return np.load(self.CACHE_FILE)

        print("APIから標高データを並列処理で取得します...")
        coordinates = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        
        height_grid = np.zeros((self.grid_size, self.grid_size))
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(self.fetch_single_elevation, coordinates), total=len(coordinates), desc="標高データ取得中"))

        for i, j, elevation in results:
            height_grid[i, j] = elevation
            
        np.save(self.CACHE_FILE, height_grid)
        print(f"標高データを '{self.CACHE_FILE}' に保存しました。")
        return height_grid
    

    # 六甲山燃やす場合: 100mごとに緯度経度ばらまきで標高取得



    