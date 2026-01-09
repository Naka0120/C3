import math

# --- Simulation Parameters ---
# Grid Size
GRID_SIZE = 200
# Cell Size (meters)
CELL_SIZE_M = 10

# Infection (Burn) Probability Base
P_H = 0.025  # 論文値 0.58 corresponds to 0.025 in this scale/implementation

# Recovery Time (Steps or Arbitrary Time Unit)
RECOVERY_TIME = 217

# --- Physics / Environmental Constants ---
# Slope Effect
SLOPE_FACTOR = 0.078  # 論文値 0.078

# Wind Parameters
WIND_SPEED = 4.5  # m/s?
THETA_W = 3 * math.pi / 4  # Wind Direction (Radians)

# Wind Effect Coefficients (from param.txt)
C_1 = 0.4   # 論文値 0.045 -> param.txt says 0.4
C_2 = 0.64  # 論文値 0.131 -> param.txt says 0.64

# --- Interaction / Water Control ---
# Threshold of ACTIVE cells to enable water placement
ACTIVE_THRESHOLD = 50 

# Water Duration Steps
WATER_ON_ACTIVE_DURATION = 2
WATER_ON_GREEN_DURATION = 2
WATER_ON_BURNED_DURATION = 2

# Max cells per drag event
MAX_WATER_CELLS_PER_DRAG_STEP = 9

# --- File Paths ---
# Default to CSV mode paths relative to the script execution or absolute
TERRAIN_MODE = "CSV"  # "DUMMY", "CSV", "API"

CSV_FILEPATH_ELEV = r"USA_Fire\elevation_grid.csv"
CSV_FILEPATH_VEGE = r"USA_Fire\vegetation_grid.csv"
CSV_FILEPATH_IGN = r"USA_Fire\ignition_synced_wide.csv"

# API Mode Settings
API_BASE_LAT = 34.776
API_BASE_LON = 135.252
