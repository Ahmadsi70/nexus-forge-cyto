const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const btnBrowse = document.getElementById('btn-browse');
const resultsSection = document.getElementById('results-section');
const canvas = document.getElementById('image-canvas');
const ctx = canvas.getContext('2d');
const scannerOverlay = document.getElementById('scanner-overlay');
const btnReset = document.getElementById('btn-reset');
const legendCard = document.getElementById('legend-card');

// Stats Elements
const statTotal = document.getElementById('stat-total');
const statNodes = document.getElementById('stat-nodes');
const statEdges = document.getElementById('stat-edges');
const statPrep = document.getElementById('stat-prep');
const statMojo = document.getElementById('stat-mojo');

// API Endpoint (FastAPI backend running on port 8810 via docker-compose)
const API_URL = 'http://localhost:8810/analyze-image';

// Event Listeners
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation(); // prevent clicking dropZone again
    fileInput.click();
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

btnReset.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    legendCard.classList.add('hidden');
    dropZone.classList.remove('hidden');
    resetStats();
});

let currentImage = null;

function resetStats() {
    statTotal.textContent = '-- ms';
    statNodes.textContent = '--';
    statEdges.textContent = '--';
    statPrep.textContent = '-- ms';
    statMojo.textContent = '-- ms';
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            
            // Set canvas size (we will scale it via CSS, but keep internal resolution high)
            canvas.width = img.width;
            canvas.height = img.height;
            
            // Draw original image
            ctx.drawImage(img, 0, 0);
            
            // Show results section and hide drop zone
            dropZone.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            scannerOverlay.classList.remove('hidden'); // Show scanner
            
            resetStats();
            
            // Send to Backend
            processImage(file);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

async function processImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Hide scanner
        scannerOverlay.classList.add('hidden');
        
        if (data.status === 'success') {
            updateStats(data);
            drawResults(data.cells);
            legendCard.classList.remove('hidden');
        } else {
            alert('Error from AI Engine: ' + data.message);
            console.error(data);
        }
        
    } catch (error) {
        scannerOverlay.classList.add('hidden');
        alert('Failed to connect to Antigravity Engine. Ensure SSH tunnel is active.');
        console.error(error);
    }
}

function updateStats(data) {
    // Smooth counter animation
    animateValue(statNodes, 0, data.nodes_count, 1500, '');
    animateValue(statEdges, 0, data.edges_count, 1500, '');
    
    const totalMs = data.timing_ms.total.toFixed(1);
    animateValue(statTotal, 0, parseFloat(totalMs), 1500, ' ms');
    
    statPrep.textContent = data.timing_ms.image_processing.toFixed(1) + ' ms';
    statMojo.textContent = data.timing_ms.mojo_engine.toFixed(2) + ' ms';
}

function drawResults(cells) {
    // Redraw image to clear any previous overlays
    ctx.drawImage(currentImage, 0, 0);
    
    // Optional: Draw connecting lines (edges) to simulate the graph if we wanted, 
    // but the API currently returns edges count, not the edge pairs.
    // We will draw the nodes and highlight cancer cells.

    cells.forEach(cell => {
        if (cell.is_cancer) {
            // Cancer Cell -> Glowing Pink Box + Dot
            ctx.beginPath();
            ctx.arc(cell.x, cell.y, 8, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(244, 63, 94, 0.4)'; // Pink Glow
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(cell.x, cell.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#f43f5e'; 
            ctx.fill();
            
            ctx.strokeStyle = '#f43f5e';
            ctx.lineWidth = 3;
            ctx.strokeRect(cell.x - 16, cell.y - 16, 32, 32);
        } else {
            // Normal Cell -> Purple Dot
            ctx.beginPath();
            ctx.arc(cell.x, cell.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#8B5CF6'; 
            ctx.fill();
            
            ctx.strokeStyle = 'rgba(139, 92, 246, 0.5)';
            ctx.lineWidth = 1;
            ctx.strokeRect(cell.x - 8, cell.y - 8, 16, 16);
        }
    });
}

function animateValue(obj, start, end, duration, suffix) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        
        // Use ease-out easing
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(easeOut * (end - start) + start);
        
        // If it's the statTotal element, we might need a decimal point if we passed a float, but we floor it for animation simplicity.
        obj.innerHTML = current + suffix;
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            // Ensure exact final value
            obj.innerHTML = end + suffix;
        }
    };
    window.requestAnimationFrame(step);
}
