import csv
import sys
import os

# Coordinates from ForestFire_Auto_Anim_240.py
GRID_ORIGIN_X = 521863
GRID_ORIGIN_Y = 3383228
GRID_RES = 44.835
GRID_SIZE = 200 # From config.py, though the user mentioned 500x500 in the prompt? 
# The user said "Currently drawing range is 500x500". 
# But config.py says GRID_SIZE = 200.
# ForestFire_Auto_Anim_240.py imports config and uses config.GRID_SIZE.
# Let's check if the visualizer sets a different range or if the user is mistaken about 500x500 vs 200x200.
# Or maybe the actual grid size used in the anim script is different.
# ForestFire_Auto_Anim_240.py line 33: def __init__(self, grid_size=config.GRID_SIZE, ...
# So it uses 200 unless passed otherwise.
# But the user said "Currently drawing range is 500x500". Maybe they mean pixels? 
# Or maybe they are running it with a different config?
# I will just calculate the grid indices (r, c). 

csv_path = r"USA_Fire/ignition_synced_wide.csv"

min_r, max_r = float('inf'), float('-inf')
min_c, max_c = float('inf'), float('-inf')

valid_points = 0

try:
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = 1
            while True:
                col_x = f'Ign{i}_X'
                col_y = f'Ign{i}_Y'
                if col_x not in row or col_y not in row:
                    break
                
                val_x = row[col_x]
                val_y = row[col_y]
                
                if val_x and val_y:
                    try:
                        utm_x = float(val_x)
                        utm_y = float(val_y)
                        
                        c = int((utm_x - GRID_ORIGIN_X) / GRID_RES)
                        r = int((GRID_ORIGIN_Y - utm_y) / GRID_RES)
                        
                        min_r = min(min_r, r)
                        max_r = max(max_r, r)
                        min_c = min(min_c, c)
                        max_c = max(max_c, c)
                        valid_points += 1
                    except ValueError:
                        pass
                i += 1

    print(f"Total valid ignition points processed: {valid_points}")
    print(f"Row range: {min_r} to {max_r} (Height: {max_r - min_r + 1})")
    print(f"Col range: {min_c} to {max_c} (Width: {max_c - min_c + 1})")
    
    center_r = (min_r + max_r) // 2
    center_c = (min_c + max_c) // 2
    print(f"Center: ({center_r}, {center_c})")

except Exception as e:
    print(f"Error: {e}")
