/* ═══════════════════════════════════════════════════════════
   AeroInverse — Application Logic
   ═══════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────
let currentResult = null;
let selectedMethod = 'gradient';

// ── Slider Sync ────────────────────────────────
document.getElementById('machSlider').addEventListener('input', e => {
    document.getElementById('mach').value = e.target.value;
});
document.getElementById('mach').addEventListener('input', e => {
    document.getElementById('machSlider').value = e.target.value;
});
document.getElementById('alphaSlider').addEventListener('input', e => {
    document.getElementById('alpha').value = e.target.value;
});
document.getElementById('alpha').addEventListener('input', e => {
    document.getElementById('alphaSlider').value = e.target.value;
});

// ── Toggle Sync ────────────────────────────────
document.getElementById('clToggle').addEventListener('change', e => {
    document.getElementById('targetCL').disabled = !e.target.checked;
});
document.getElementById('cdToggle').addEventListener('change', e => {
    document.getElementById('targetCD').disabled = !e.target.checked;
});
document.getElementById('cmToggle').addEventListener('change', e => {
    document.getElementById('targetCM').disabled = !e.target.checked;
});

// ── Method Selector ────────────────────────────
document.querySelectorAll('.method-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.method-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedMethod = btn.dataset.method;
    });
});

// ── Main Design Function ───────────────────────
async function runDesign() {
    const btn = document.getElementById('designBtn');
    const loadingBar = document.getElementById('loadingBar');
    
    btn.disabled = true;
    btn.querySelector('span').textContent = 'Optimizing...';
    loadingBar.classList.add('active');
    
    const payload = {
        mach: parseFloat(document.getElementById('mach').value),
        alpha: parseFloat(document.getElementById('alpha').value),
        method: selectedMethod,
        weight_CL: parseFloat(document.getElementById('weightCL').value),
        weight_CD: parseFloat(document.getElementById('weightCD').value),
        weight_CM: parseFloat(document.getElementById('weightCM').value)
    };
    
    if (document.getElementById('clToggle').checked) {
        payload.target_CL = parseFloat(document.getElementById('targetCL').value);
    }
    if (document.getElementById('cdToggle').checked) {
        payload.target_CD = parseFloat(document.getElementById('targetCD').value);
    }
    if (document.getElementById('cmToggle').checked) {
        payload.target_CM = parseFloat(document.getElementById('targetCM').value);
    }
    
    try {
        const response = await fetch('/api/inverse_design', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Design failed');
        }
        
        currentResult = await response.json();
        
        // Update all visualizations
        drawAirfoil(currentResult.x_coords, currentResult.y_coords);
        updateResults(currentResult, payload);
        updatePlots(currentResult.alpha_sweep);
        updateCST(currentResult.cst_upper, currentResult.cst_lower);
        
        document.getElementById('airfoilEmpty').style.display = 'none';
        document.getElementById('resultsGrid').style.display = 'grid';
        document.getElementById('cstSection').style.display = 'block';
        
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.querySelector('span').textContent = 'Run Inverse Design';
        loadingBar.classList.remove('active');
    }
}

// ── Airfoil Drawing ────────────────────────────
function drawAirfoil(xCoords, yCoords) {
    const canvas = document.getElementById('airfoilCanvas');
    const ctx = canvas.getContext('2d');
    
    // High DPI
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    
    const W = rect.width;
    const H = rect.height;
    
    // Clear
    ctx.fillStyle = '#1a2236';
    ctx.fillRect(0, 0, W, H);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(42, 54, 84, 0.5)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 10; i++) {
        const x = 60 + (W - 100) * i / 10;
        ctx.beginPath(); ctx.moveTo(x, 30); ctx.lineTo(x, H - 40); ctx.stroke();
    }
    for (let i = 0; i <= 6; i++) {
        const y = 30 + (H - 70) * i / 6;
        ctx.beginPath(); ctx.moveTo(60, y); ctx.lineTo(W - 40, y); ctx.stroke();
    }
    
    // Transform coordinates
    const padding = { left: 80, right: 40, top: 50, bottom: 50 };
    const plotW = W - padding.left - padding.right;
    const plotH = H - padding.top - padding.bottom;
    
    const xMin = Math.min(...xCoords);
    const xMax = Math.max(...xCoords);
    const yMin = Math.min(...yCoords);
    const yMax = Math.max(...yCoords);
    
    const yRange = Math.max(yMax - yMin, 0.01);
    const xRange = xMax - xMin;
    const scale = Math.min(plotW / xRange, plotH / yRange);
    
    const offsetX = padding.left + (plotW - xRange * scale) / 2;
    const offsetY = padding.top + plotH / 2;
    
    function tx(x) { return offsetX + (x - xMin) * scale; }
    function ty(y) { return offsetY - y * scale; }
    
    // Chord line
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(tx(0), ty(0));
    ctx.lineTo(tx(1), ty(0));
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Fill airfoil
    const gradient = ctx.createLinearGradient(tx(0), ty(yMax), tx(1), ty(yMin));
    gradient.addColorStop(0, 'rgba(129, 140, 248, 0.12)');
    gradient.addColorStop(0.5, 'rgba(52, 211, 153, 0.08)');
    gradient.addColorStop(1, 'rgba(129, 140, 248, 0.12)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(tx(xCoords[0]), ty(yCoords[0]));
    for (let i = 1; i < xCoords.length; i++) {
        ctx.lineTo(tx(xCoords[i]), ty(yCoords[i]));
    }
    ctx.closePath();
    ctx.fill();
    
    // Draw airfoil outline
    const outlineGrad = ctx.createLinearGradient(tx(0), 0, tx(1), 0);
    outlineGrad.addColorStop(0, '#818CF8');
    outlineGrad.addColorStop(0.5, '#34D399');
    outlineGrad.addColorStop(1, '#818CF8');
    
    ctx.strokeStyle = outlineGrad;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(tx(xCoords[0]), ty(yCoords[0]));
    for (let i = 1; i < xCoords.length; i++) {
        ctx.lineTo(tx(xCoords[i]), ty(yCoords[i]));
    }
    ctx.closePath();
    ctx.stroke();
    
    // Axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) {
        ctx.fillText((i / 10).toFixed(1), tx(i / 10), H - 28);
    }
    ctx.fillText('x/c', W / 2, H - 10);
    
    ctx.textAlign = 'right';
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
        const yVal = yMin + (yMax - yMin) * (yTicks - i) / yTicks;
        ctx.fillText(yVal.toFixed(3), padding.left - 8, ty(yVal) + 4);
    }
    ctx.save();
    ctx.translate(14, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('y/c', 0, 0);
    ctx.restore();
}

// ── Update Results ─────────────────────────────
function updateResults(result, targets) {
    const pred = result.predicted;
    
    document.getElementById('resCL').textContent = pred.CL.toFixed(4);
    document.getElementById('resCD').textContent = pred.CD.toFixed(6);
    document.getElementById('resCM').textContent = pred.CM.toFixed(4);
    document.getElementById('resLD').textContent = (pred.CL / pred.CD).toFixed(1);
    
    // Target comparison
    const clTarget = document.getElementById('resCLTarget');
    const cdTarget = document.getElementById('resCDTarget');
    const cmTarget = document.getElementById('resCMTarget');
    
    if (targets.target_CL !== undefined) {
        const err = Math.abs(pred.CL - targets.target_CL);
        clTarget.textContent = `Target: ${targets.target_CL} (Δ=${err.toFixed(4)})`;
    } else { clTarget.textContent = ''; }
    
    if (targets.target_CD !== undefined) {
        const err = Math.abs(pred.CD - targets.target_CD);
        cdTarget.textContent = `Target: ${targets.target_CD} (Δ=${err.toFixed(6)})`;
    } else { cdTarget.textContent = ''; }
    
    if (targets.target_CM !== undefined) {
        const err = Math.abs(pred.CM - targets.target_CM);
        cmTarget.textContent = `Target: ${targets.target_CM} (Δ=${err.toFixed(4)})`;
    } else { cmTarget.textContent = ''; }
    
    // Animate in
    document.querySelectorAll('.result-card').forEach((card, i) => {
        card.style.animation = 'none';
        card.offsetHeight;
        card.style.animation = `fadeIn 0.4s ease-out ${i * 0.1}s both`;
    });
}

// ── Performance Plots ──────────────────────────
function drawPlot(canvasId, xData, yData, xLabel, yLabel, color, markerAlpha) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const W = rect.width - 24;
    const H = 160;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    
    const pad = { left: 45, right: 15, top: 10, bottom: 25 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    
    ctx.fillStyle = '#1a2236';
    ctx.fillRect(0, 0, W, H);
    
    if (!xData || xData.length === 0) return;
    
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const yMin = Math.min(...yData);
    const yMax = Math.max(...yData);
    const yr = yMax - yMin || 1;
    const xr = xMax - xMin || 1;
    
    function tx(x) { return pad.left + (x - xMin) / xr * plotW; }
    function ty(y) { return pad.top + plotH - (y - yMin) / yr * plotH; }
    
    // Grid
    ctx.strokeStyle = 'rgba(42, 54, 84, 0.5)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + plotH * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    }
    
    // Line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(tx(xData[0]), ty(yData[0]));
    for (let i = 1; i < xData.length; i++) {
        ctx.lineTo(tx(xData[i]), ty(yData[i]));
    }
    ctx.stroke();
    
    // Fill under
    ctx.fillStyle = color.replace(')', ', 0.08)').replace('rgb', 'rgba');
    ctx.beginPath();
    ctx.moveTo(tx(xData[0]), ty(yMin));
    for (let i = 0; i < xData.length; i++) {
        ctx.lineTo(tx(xData[i]), ty(yData[i]));
    }
    ctx.lineTo(tx(xData[xData.length - 1]), ty(yMin));
    ctx.closePath();
    ctx.fill();
    
    // Axes labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, W / 2, H - 4);
    
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const val = yMin + yr * (4 - i) / 4;
        ctx.fillText(val.toFixed(val < 0.1 ? 4 : 2), pad.left - 4, pad.top + plotH * i / 4 + 4);
    }
}

function updatePlots(sweep) {
    if (!sweep || sweep.length === 0) return;
    
    const alphas = sweep.map(s => s.alpha);
    const cls = sweep.map(s => s.CL);
    const cds = sweep.map(s => s.CD);
    const lds = sweep.map(s => s.CL / Math.max(s.CD, 1e-6));
    
    drawPlot('clAlphaPlot', alphas, cls, 'α (°)', 'CL', 'rgb(129, 140, 248)');
    drawPlot('cdAlphaPlot', alphas, cds, 'α (°)', 'CD', 'rgb(52, 211, 153)');
    drawPlot('ldAlphaPlot', alphas, lds, 'α (°)', 'L/D', 'rgb(167, 139, 250)');
    drawPlot('dragPolar', cds, cls, 'CD', 'CL', 'rgb(251, 191, 36)');
}

// ── CST Display ────────────────────────────────
function updateCST(upper, lower) {
    const upperDiv = document.getElementById('cstUpper');
    const lowerDiv = document.getElementById('cstLower');
    
    upperDiv.innerHTML = upper.map((v, i) =>
        `<div class="cst-val"><span class="cst-val-label">w${i}</span><span>${v.toFixed(6)}</span></div>`
    ).join('');
    
    lowerDiv.innerHTML = lower.map((v, i) =>
        `<div class="cst-val"><span class="cst-val-label">w${i}</span><span>${v.toFixed(6)}</span></div>`
    ).join('');
}

// ── Alpha Sweep Update ─────────────────────────
async function updateSweep() {
    if (!currentResult) return;
    
    const mach = parseFloat(document.getElementById('plotMach').value);
    
    try {
        const response = await fetch('/api/alpha_sweep', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cst_upper: currentResult.cst_upper,
                cst_lower: currentResult.cst_lower,
                mach: mach
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            updatePlots(data.sweep);
        }
    } catch (e) {
        console.error('Sweep error:', e);
    }
}

// ── Download Dropdown ───────────────────────────
function toggleDownloadMenu(e) {
    e.stopPropagation();
    if (!currentResult) { alert('No design to export yet. Run Inverse Design first.'); return; }
    const menu = document.getElementById('downloadMenu');
    menu.classList.toggle('open');
}

// Close dropdown when clicking outside
document.addEventListener('click', () => {
    document.getElementById('downloadMenu')?.classList.remove('open');
});

// ── Ramer-Douglas-Peucker (RDP) Simplification ──
/**
 * RDP algorithm for polyline simplification.
 * Recursively removes points that deviate less than `epsilon` from the
 * straight line connecting the segment endpoints.
 * Ideal for airfoil coordinates because it always keeps leading-edge
 * and trailing-edge points and retains high curvature regions automatically.
 *
 * @param {number[]} xs - x coordinates
 * @param {number[]} ys - y coordinates
 * @param {number} epsilon - maximum allowed perpendicular deviation (in chord units)
 * @returns {boolean[]} mask - true = keep this point
 */
function rdpSimplify(xs, ys, epsilon) {
    const n = xs.length;
    if (n <= 2) return new Array(n).fill(true);

    const mask = new Array(n).fill(false);
    mask[0] = true;
    mask[n - 1] = true;

    function perpendicularDist(px, py, ax, ay, bx, by) {
        const dx = bx - ax, dy = by - ay;
        const lenSq = dx * dx + dy * dy;
        if (lenSq === 0) return Math.hypot(px - ax, py - ay);
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
        return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    }

    function rdp(start, end) {
        if (end - start <= 1) return;
        let maxDist = 0, maxIdx = start;
        for (let i = start + 1; i < end; i++) {
            const d = perpendicularDist(xs[i], ys[i], xs[start], ys[start], xs[end], ys[end]);
            if (d > maxDist) { maxDist = d; maxIdx = i; }
        }
        if (maxDist > epsilon) {
            mask[maxIdx] = true;
            rdp(start, maxIdx);
            rdp(maxIdx, end);
        }
    }

    rdp(0, n - 1);
    return mask;
}

/**
 * Airfoil-aware coordinate reduction.
 * Splits the airfoil at the leading edge, applies a tighter RDP epsilon
 * to the high-curvature leading-edge region (front 15% chord) and a
 * looser epsilon to the flatter mid/aft sections, then recombines.
 */
function reduceAirfoilCoords(xs, ys) {
    const n = xs.length;

    // Find the leading edge index (minimum x)
    let leIdx = 0;
    for (let i = 1; i < n; i++) {
        if (xs[i] < xs[leIdx]) leIdx = i;
    }

    // Upper surface: from TE (index 0) → LE
    const xU = xs.slice(0, leIdx + 1);
    const yU = ys.slice(0, leIdx + 1);

    // Lower surface: from LE → TE (end)
    const xL = xs.slice(leIdx);
    const yL = ys.slice(leIdx);

    function adaptiveRDP(xArr, yArr) {
        const m = xArr.length;
        // Points within first 15% chord get tighter tolerance
        const leThreshold = 0.15;
        const epsTight = 0.0002;   // ~0.02% chord — preserves leading-edge curvature
        const epsLoose = 0.0008;   // ~0.08% chord — safe for flat aft sections

        // Run RDP twice on two zones and merge masks
        const leEnd = xArr.findIndex(x => x > leThreshold);
        const splitIdx = (leEnd > 2) ? leEnd : m;

        const maskA = rdpSimplify(xArr.slice(0, splitIdx), yArr.slice(0, splitIdx), epsTight);
        const maskB = rdpSimplify(xArr.slice(splitIdx - 1), yArr.slice(splitIdx - 1), epsLoose);

        // Combine — overlap point at splitIdx-1 is already kept in both
        const fullMask = [...maskA, ...maskB.slice(1)];
        return fullMask;
    }

    const maskU = adaptiveRDP(xU, yU);
    const maskL = adaptiveRDP(xL, yL);

    // Reconstruct: upper TE→LE, then lower LE→TE (skip shared LE point)
    let xOut = [], yOut = [];
    for (let i = 0; i < xU.length; i++) if (maskU[i]) { xOut.push(xU[i]); yOut.push(yU[i]); }
    for (let i = 1; i < xL.length; i++) if (maskL[i]) { xOut.push(xL[i]); yOut.push(yL[i]); }

    return { x: xOut, y: yOut };
}

// ── Export Coordinates ──────────────────────────
function exportCoords(mode) {
    document.getElementById('downloadMenu').classList.remove('open');
    if (!currentResult) { alert('No design to export.'); return; }

    let xs, ys, filename, header;

    if (mode === 'reduced') {
        const reduced = reduceAirfoilCoords(currentResult.x_coords, currentResult.y_coords);
        xs = reduced.x;
        ys = reduced.y;
        filename = 'designed_airfoil_reduced.csv';
        header = `# AeroInverse Designed Airfoil — Geometry-Accurate Reduced Coordinates\n` +
                 `# Points: ${xs.length} (RDP-simplified from ${currentResult.x_coords.length}, adaptive e=0.02-0.08% chord)\n` +
                 `x,y\n`;
    } else {
        xs = currentResult.x_coords;
        ys = currentResult.y_coords;
        filename = 'designed_airfoil_full.csv';
        header = `# AeroInverse Designed Airfoil — Full CST Coordinates\n` +
                 `# Points: ${xs.length} (cosine-spaced, 200 pts/surface)\n` +
                 `x,y\n`;
    }

    let content = header;
    for (let i = 0; i < xs.length; i++) {
        content += `${xs[i].toFixed(8)},${ys[i].toFixed(8)}\n`;
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

// ── Window Resize ──────────────────────────────
window.addEventListener('resize', () => {
    if (currentResult) {
        drawAirfoil(currentResult.x_coords, currentResult.y_coords);
        if (currentResult.alpha_sweep) {
            updatePlots(currentResult.alpha_sweep);
        }
    }
});

// ── Init ───────────────────────────────────────
console.log('AeroInverse initialized');
