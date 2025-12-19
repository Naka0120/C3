# Forest Fire Simulation - Usage Guide

This guide explains how to start and use the Forest Fire Simulation web application. The system consists of two parts: a Python **Backend** (API) and a React **Frontend** (UI).

## 1. Start the Backend
The backend runs the simulation logic (Python).

1.  Open a terminal.
2.  Navigate to the backend directory:
    ```powershell
    cd C:\Users\souta\Work\C3\water_Fire\web_app\backend
    ```
3.  Start the server:
    ```powershell
    python main.py
    ```
    *   *Note: This server usually runs on `http://localhost:8000`.*

## 2. Start the Frontend
The frontend displays the simulation grid and controls.

1.  Open a **new** terminal (do not close the backend terminal).
2.  Navigate to the frontend directory:
    ```powershell
    cd C:\Users\souta\Work\C3\water_Fire\web_app\frontend
    ```
3.  Start the development server:
    ```powershell
    npm run dev
    ```
    *(You have likely already done this step!)*

## 3. Open in Browser
1.  Look at the `npm run dev` terminal output for the "Local" URL.
2.  It will usually be: http://localhost:5173 (or `5174` if `5173` was busy).
3.  Open that link in your web browser (Chrome, Edge, etc.).

## How to Use
-   **Start/Pause**: Click the "Start" button to begin the fire spread.
-   **Reset**: Click "Reset" to restart with a fresh grid.
-   **Water Mode**:
    *   When the fire is large enough (>50 active fires), "Water Mode" becomes available.
    *   Click "Water Mode" to enable it.
    *   Click on the grid to drop water and extinguish fires.
