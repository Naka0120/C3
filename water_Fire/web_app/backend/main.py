
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from sim_wrapper import SimulationManager
import uvicorn

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sim_manager = SimulationManager()

@app.get("/")
def root():
    return {"message": "Forest Fire Simulation API is running"}

@app.post("/reset")
def reset_simulation():
    sim_manager.initialize()
    return {"message": "Simulation reset"}

@app.get("/state")
def get_state():
    return {
        "grid": sim_manager.get_grid_state(),
        "stats": sim_manager.get_stats()
    }

@app.post("/step")
def step(steps: int = Body(1, embed=True)):
    for _ in range(steps):
        sim_manager.step()
    return get_state()

@app.post("/action/water")
def place_water(x: int = Body(...), y: int = Body(...)):
    sim_manager.place_water(x, y)
    return {"message": "Water placed"}

@app.get("/elevation")
def get_elevation():
    return {
        "elevation": sim_manager.get_elevation_grid()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
