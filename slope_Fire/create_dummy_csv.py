import numpy as np

# 作成するダミーCSVの設定
GRID_SIZE = 200
CSV_FILENAME = "dummy_terrain.csv"

print(f"'{CSV_FILENAME}' を作成中...")

# 簡単な坂道（左から右へ高くなる）の地形データを作成
# 標高は0mから199mになります
dummy_grid = np.array([[j*3 for j in range(GRID_SIZE)] for i in range(GRID_SIZE)], dtype=float)

# CSVファイルとして保存
np.savetxt(CSV_FILENAME, dummy_grid, delimiter=',', fmt='%.2f')

print(f"サイズ {GRID_SIZE}x{GRID_SIZE} のダミー地形ファイル '{CSV_FILENAME}' を作成しました。")