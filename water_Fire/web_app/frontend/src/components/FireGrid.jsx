
import React, { useRef, useEffect } from 'react';

const GRID_SIZE = 200;

// Color mapping to match Matplotlib's ListedColormap roughly
// GREEN (0-3), ACTIVE (4-6), BURNED (7), RIVER (8), WATER (9)
const COLORS = [
    '#e0ffe0', '#80ff80', '#00cc44', '#006622', // GREEN
    '#8B0000', '#DC143C', '#FF5050', // ACTIVE
    '#646464',                       // BURNED
    '#00bfff',                       // RIVER
    '#00ffff'                        // WATER
];

const FireGrid = ({ gridData, isWaterMode, onPlaceWater }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);

    // Initial and Update Render
    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (!gridData || gridData.length === 0) {
            // Draw placeholder or clear
            ctx.fillStyle = '#1e293b'; // Slate-800
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            return;
        }

        // Draw Grid
        // gridData is flat ? No, python returns .tolist() on 2D array, so it is [[...], [...]]
        // Let's assume 200x200

        const rows = gridData.length;
        const cols = gridData[0].length;

        // Pixel size calculation
        // We want to fit in the container.
        // But for performance, let's keep canvas resolution at 200x200 (or 400x400) and scale via CSS

        // Actually, drawing 200x200 rectangles is heavy if using fillRect 40,000 times JS.
        // Better to use ImageData.

        const imgData = ctx.createImageData(cols, rows);
        const data = imgData.data;

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const state = gridData[i][j];
                const colorHex = COLORS[state] || '#000000';

                // Hex to RGB
                const r = parseInt(colorHex.slice(1, 3), 16);
                const g = parseInt(colorHex.slice(3, 5), 16);
                const b = parseInt(colorHex.slice(5, 7), 16);

                const idx = (i * cols + j) * 4;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
                data[idx + 3] = 255; // Alpha
            }
        }

        // Put data to a small offscreen canvas or just directly if size matches
        // Here we render to a small canvas (matches grid size) and scale up with CSS
        ctx.putImageData(imgData, 0, 0);

    }, [gridData]);

    const handleInteraction = (e) => {
        if (!isWaterMode) return;

        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();

        // Mouse coordinate relative to canvas
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Scale to grid coordinates
        // Canvas internal resolution is Grid Size (200x200)
        // Canvas display size is `rect.width` x `rect.height`

        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const gridX = Math.floor(x * scaleX);
        const gridY = Math.floor(y * scaleY);

        onPlaceWater(gridX, gridY);
    };

    return (
        <div ref={containerRef} className="w-full aspect-square relative cursor-crosshair">
            <canvas
                ref={canvasRef}
                width={GRID_SIZE}
                height={GRID_SIZE}
                className="w-full h-full block rounded-lg rendering-pixelated"
                style={{ imageRendering: 'pixelated' }}
                onMouseDown={(e) => {
                    // For drag support, we might need state, but for now just click interaction
                    if (e.buttons === 1) handleInteraction(e);
                }}
                onMouseMove={(e) => {
                    if (e.buttons === 1) handleInteraction(e);
                }}
                onClick={handleInteraction}
            />
        </div>
    );
};

export default FireGrid;
