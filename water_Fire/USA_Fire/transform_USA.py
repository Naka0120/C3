import numpy as np
import pandas as pd

# 1. Read the CSV file
file_name = "L2F_inputmap.csv"
data = pd.read_csv(file_name)

# 2. Define grid size (derived from the data shape 40000 rows and indices 0-199)
GRID_SIZE = 200

# Function to reshape the data and save to CSV
def create_grid_csv(df, value_column, output_filename, grid_size):
    # Pivot the DataFrame to reshape it into a 2D grid
    # row_index becomes index (rows), col_index becomes columns, value_column becomes values
    grid_pivot = df.pivot(index='row_index', columns='col_index', values=value_column)
    
    # ★★★ 修正: 欠損値 (NaN) を 0 で埋める ★★★
    grid_pivot = grid_pivot.fillna(0)
    
    grid_array = grid_pivot.values
    
    # Save the NumPy array to CSV
    # fmt='%f' を使用して浮動小数点形式で保存
    np.savetxt(output_filename, grid_array, delimiter=',', fmt='%f')
    print(f"'{output_filename}' を作成しました。データの最初の5x5 (NaNは0に置換済み):\n{grid_array[:5, :5]}")
    return grid_array.shape


# 3-5. Create elevation_grid.csv (using Latitudemean)
elevation_shape = create_grid_csv(data, 'Latitudemean', 'elevation_grid.csv', GRID_SIZE)

# 6-8. Create vegetation_grid.csv (using Fuelmean)
vegetation_shape = create_grid_csv(data, 'Fuelmean', 'vegetation_grid.csv', GRID_SIZE)

print(f"\n作成されたファイルの形状: elevation_grid.csv: {elevation_shape}, vegetation_grid.csv: {vegetation_shape}")