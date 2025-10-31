import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import imageio
from ForestFire_Auto import SIRCellularAutomataSimple, GREEN, ACTIVE, BURNED, DILUTED, RIVER

if __name__ == '__main__':
    # --- シミュレーション設定 ---
    TERRAIN_MODE = "CSV"  # "DUMMY", "CSV", "API" から選択
    sim_params = {
        "grid_size": 200,
        "infection_probability": 0.58,
        "recovery_time": 217,
        "cell_size_m": 10
    }
    api_params = {
        "base_lat": 34.776,
        "base_lon": 135.252,
    }
    csv_params = {
        "csv_filepath_elev": "Chiri_Fire\\elevation_grid.csv",
        "csv_filepath_vege": "Chiri_Fire\\vegetation_grid.csv"
    }
    all_params = sim_params.copy()
    all_params["terrain_mode"] = TERRAIN_MODE
    if TERRAIN_MODE == "API":
        all_params.update(api_params)
    elif TERRAIN_MODE == "CSV":
        all_params.update(csv_params)

    sir_ca = SIRCellularAutomataSimple(**all_params)
    center = sim_params["grid_size"] // 2
    sir_ca.grid[center, center].state = ACTIVE

    t_end = 150
    frames = []
    cmap = ListedColormap([
        '#e0ffe0', '#80ff80', '#00cc44', '#006622', # GREEN (密度4段階)
        '#8B0000', '#DC143C', '#FF5050',           # ACTIVE (燃焼強度3段階)
        '#646464',                                 # BURNED
        'deepskyblue'                              # RIVER
    ])

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')

    print("シミュレーションとフレーム生成を開始します...")
    for t in tqdm(range(t_end), desc="進行状況", ncols=80):
        sir_ca.update_grid()
        color_grid = np.zeros((sir_ca.grid_size, sir_ca.grid_size), dtype=int)
        is_green = sir_ca.state_grid == GREEN
        density_on_green = sir_ca.density_grid[is_green]
        conditions = [
            density_on_green < 0.25,
            density_on_green < 0.5,
            density_on_green < 0.75,
            density_on_green >= 0.75
        ]
        choices = [0, 1, 2, 3]
        color_grid[is_green] = np.select(conditions, choices)
        color_grid[sir_ca.state_grid == BURNED] = 7
        color_grid[sir_ca.state_grid == RIVER] = 8
        active_coords = np.argwhere(sir_ca.state_grid == ACTIVE)
        for i, j in active_coords:
            tt = sir_ca.infection_time[i, j]
            n = sir_ca.recovery_time
            burn_intensity = sir_ca.active_function(tt, n)
            if burn_intensity > 0.66:
                color_grid[i, j] = 4
            elif burn_intensity > 0.33:
                color_grid[i, j] = 5
            else:
                color_grid[i, j] = 6
        ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=8)
        ax.set_title(f"Step {t+1}")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        ax.clear()
        plt.axis('off')


    plt.close(fig)
    print("GIF動画を保存中...")
    imageio.mimsave("forestfire_simulation_chile_rev.gif", frames, fps=10)
    print("forestfire_simulation_chile_rev.gif を保存しました。")
    print("MP4動画を保存中...")
    with imageio.get_writer("forestfire_simulation_chile_rev.mp4", fps=10, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)
    print("forestfire_simulation_chile_rev.mp4 を保存しました。")