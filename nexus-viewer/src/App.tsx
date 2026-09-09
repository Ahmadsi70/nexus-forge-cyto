import { useEffect, useRef, useState } from 'react';
import './App.css';

interface PolygonPoint {
  x: number;
  y: number;
}

interface CellResult {
  id: number;
  is_malignant: number;
  polygon: PolygonPoint[];
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cells, setCells] = useState<CellResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch data from Axum API
    fetch('http://localhost:8811/api/results')
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch results");
        return res.json();
      })
      .then(data => {
        setCells(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!canvasRef.current || cells.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Find bounding box to scale properly
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    cells.forEach(cell => {
      cell.polygon.forEach(pt => {
        if (pt.x < minX) minX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y > maxY) maxY = pt.y;
      });
    });

    const scaleX = canvas.width / (maxX - minX + 10);
    const scaleY = canvas.height / (maxY - minY + 10);
    const scale = Math.min(scaleX, scaleY) * 0.9;
    const offsetX = (canvas.width - (maxX - minX) * scale) / 2;
    const offsetY = (canvas.height - (maxY - minY) * scale) / 2;

    // Draw cells
    cells.forEach(cell => {
      ctx.beginPath();
      cell.polygon.forEach((pt, i) => {
        const px = (pt.x - minX) * scale + offsetX;
        const py = (pt.y - minY) * scale + offsetY;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      
      // Styling
      if (cell.is_malignant === 1) {
        ctx.fillStyle = 'rgba(255, 60, 60, 0.6)'; // Red for malignant
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
      } else {
        ctx.fillStyle = 'rgba(60, 200, 255, 0.4)'; // Blue for normal
        ctx.strokeStyle = 'rgba(0, 150, 255, 0.7)';
      }
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });
  }, [cells]);

  return (
    <div className="dashboard">
      <header className="header">
        <h1>Nexus-Forge Viewer</h1>
        <div className="stats">
          <span>Total Cells: {cells.length}</span>
          <span className="malignant">Malignant: {cells.filter(c => c.is_malignant).length}</span>
          <span className="normal">Normal: {cells.filter(c => !c.is_malignant).length}</span>
        </div>
      </header>

      <main className="content">
        {loading && <div className="spinner">Loading pathology data...</div>}
        {error && <div className="error">{error}</div>}
        
        <div className="canvas-container">
          <canvas 
            ref={canvasRef} 
            width={1200} 
            height={800} 
            className="wsi-canvas"
          />
        </div>
      </main>
    </div>
  );
}

export default App;
