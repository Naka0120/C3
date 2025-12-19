
import React, { useRef, useEffect } from 'react';

const ElevationMap = ({ elevationData }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (!elevationData || elevationData.length === 0) {
            ctx.fillStyle = '#2c3e50';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            return;
        }

        const rows = elevationData.length;
        const cols = elevationData[0].length;
        const imgData = ctx.createImageData(cols, rows);
        const data = imgData.data;

        // Find min/max for normalization
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const val = elevationData[i][j];
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }

        const range = max - min || 1;

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const val = elevationData[i][j];
                const normalized = (val - min) / range;

                // Terrain Color Map: Blue -> Green -> Brown -> White
                let r, g, b;

                // Simple gradient logic (approximate "terrain" feel)
                if (normalized < 0.2) {
                    // Lower ground (Green)
                    r = 34 + (normalized / 0.2) * (139 - 34);
                    g = 139 + (normalized / 0.2) * (180 - 139);
                    b = 34 + (normalized / 0.2) * (100 - 34);
                } else if (normalized < 0.5) {
                    // Mid ground (Dark Green to Brown)
                    const ratio = (normalized - 0.2) / 0.3;
                    r = 139 + ratio * (160 - 139);
                    g = 180 - ratio * (180 - 82);
                    b = 100 - ratio * (100 - 45);
                } else if (normalized < 0.8) {
                    // Higher ground (Brown to Gray)
                    const ratio = (normalized - 0.5) / 0.3;
                    r = 160 + ratio * (169 - 160);
                    g = 82 + ratio * (169 - 82);
                    b = 45 + ratio * (169 - 45);
                } else {
                    // Peaks (White/Snow)
                    const ratio = (normalized - 0.8) / 0.2;
                    r = 169 + ratio * (255 - 169);
                    g = 169 + ratio * (255 - 169);
                    b = 169 + ratio * (255 - 169);
                }

                const idx = (i * cols + j) * 4;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
                data[idx + 3] = 255;
            }
        }

        ctx.putImageData(imgData, 0, 0);

    }, [elevationData]);

    return (
        <div className="w-full aspect-square relative border border-slate-700 rounded-lg overflow-hidden">
            <canvas
                ref={canvasRef}
                width={200}
                height={200}
                className="w-full h-full block rendering-pixelated"
                style={{ imageRendering: 'pixelated' }}
            />
            <div className="absolute bottom-2 right-2 text-xs text-white/80 bg-black/50 px-2 py-1 rounded">
                Elevation
            </div>
        </div>
    );
};

export default ElevationMap;
