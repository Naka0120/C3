
import csv
import os

GRID_ORIGIN_X = 521863
GRID_ORIGIN_Y = 3383228
GRID_RES = 44.835
GRID_SIZE = 200

csv_filepath = r"C:\Users\souta\Work\C3\water_Fire\USA_Fire\ignition_synced_wide.csv"

def test_mapping():
    if not os.path.exists(csv_filepath):
        print(f"File not found: {csv_filepath}")
        return

    print(f"Testing mapping with Origin: ({GRID_ORIGIN_X}, {GRID_ORIGIN_Y}), Res: {GRID_RES}")
    
    with open(csv_filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            elapsed = row['Elapsed_Sec']
            # Try first ignition point
            if 'Ign1_X' in row and row['Ign1_X'] and 'Ign1_Y' in row and row['Ign1_Y']:
                val_x = float(row['Ign1_X'])
                val_y = float(row['Ign1_Y'])
                
                c = int((val_x - GRID_ORIGIN_X) / GRID_RES)
                r = int((GRID_ORIGIN_Y - val_y) / GRID_RES)
                
                print(f"Time {elapsed}: ({val_x}, {val_y}) -> Grid ({r}, {c})")
                
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    print("  [OK] Inside Grid")
                else:
                    print("  [FAIL] Outside Grid")
                
                count += 1
                if count >= 5: # check first 5 non-empty point
                    break

if __name__ == "__main__":
    test_mapping()
