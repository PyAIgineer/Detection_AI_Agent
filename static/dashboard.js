// ==================== DASHBOARD.JS ====================

let currentTaskId      = null;
let uploadedVideoPath  = null;
let eventSource        = null;
let modelUsageChart    = null;
let statusPollInterval = null;

// RTSP state: camera_id → { isRunning, pollInterval }
const rtspState    = {};
let availableModels = {};   // from /rtsp/models

// ==================== DOM Elements ====================
const uploadZone          = document.getElementById('uploadZone');
const fileInput           = document.getElementById('fileInput');
const uploadText          = document.getElementById('uploadText');
const predictBtn          = document.getElementById('predictBtn');
const stopBtn             = document.getElementById('stopBtn');
const logsContainer       = document.getElementById('logsContainer');
const taskIdSpan          = document.getElementById('taskId');
const taskStatusSpan      = document.getElementById('taskStatus');
const detectedItemsSpan   = document.getElementById('detectedItems');
const selectedModelsSpan  = document.getElementById('selectedModels');
const classifierResultDiv = document.getElementById('classifierResult');
const outputSection       = document.getElementById('outputSection');
const outputVideo         = document.getElementById('outputVideo');
const downloadBtn         = document.getElementById('downloadBtn');
const historyBody         = document.getElementById('historyBody');
const totalDetectionsEl   = document.getElementById('totalDetections');
const successRateEl       = document.getElementById('successRate');
const activeTasksEl       = document.getElementById('activeTasks');


// ==================== INIT ====================
document.addEventListener('DOMContentLoaded', async () => {
    await loadAvailableModels();   // fetch model list first
    await loadCameraGrid();        // build tiles from /rtsp/cameras
    loadAnalytics();
    loadHistory();
    setInterval(loadAnalytics, 10000);
    setInterval(pollAllRtspStatuses, 3000);   // refresh tile stats
});


// ==================== RTSP CAMERA GRID ====================

async function loadAvailableModels() {
    try {
        const res  = await fetch('/rtsp/models');
        availableModels = await res.json();
    } catch (e) {
        console.error('Failed to load models', e);
    }
}

async function loadCameraGrid() {
    try {
        const res     = await fetch('/rtsp/cameras');
        const cameras = await res.json();

        const grid    = document.getElementById('feedGrid');
        const upload  = document.getElementById('uploadTile');

        // Remove existing camera tiles (keep upload tile)
        grid.querySelectorAll('.cam-tile').forEach(t => t.remove());

        // Rebuild sidebar list
        const camList = document.getElementById('camListEl');
        camList.innerHTML = '<div class="grplabel">RTSP Cameras</div>';

        cameras.forEach(cam => {
            // ── Build tile ──────────────────────────────────────
            const tile = buildCameraTile(cam);
            grid.insertBefore(tile, upload);

            // ── Build sidebar item ───────────────────────────────
            const li = document.createElement('div');
            li.className = 'citem';
            li.id = `citem-${cam.camera_id}`;
            const dotCls = cam.configured ? (cam.is_running ? 'on' : 'wn') : 'off';
            li.innerHTML = `<div class="dot ${dotCls}" id="cdot-${cam.camera_id}"></div>
                            <span class="cname">${cam.label}</span>`;
            li.onclick = () => {
                document.querySelectorAll('.citem').forEach(c => c.classList.remove('active'));
                li.classList.add('active');
            };
            camList.appendChild(li);

            // ── Initial state ────────────────────────────────────
            rtspState[cam.camera_id] = { isRunning: cam.is_running };
        });

    } catch (e) {
        console.error('Failed to load camera grid', e);
    }
}


function buildCameraTile(cam) {
    const tile = document.createElement('div');
    tile.className = 'cam-tile';
    tile.id = `tile-${cam.camera_id}`;

    const configured = cam.configured;
    const isRunning  = cam.is_running;

    // ── Placeholder or stream image ──────────────────────────────
    const feedOrPlaceholder = isRunning
        ? `<img class="cam-feed" id="feed-${cam.camera_id}"
               src="/rtsp/stream/${cam.camera_id}"
               onerror="onFeedError('${cam.camera_id}')">`
        : `<div class="cam-placeholder" id="placeholder-${cam.camera_id}">
               <div class="big">${configured ? '📹' : '🔴'}</div>
               <p>${cam.label} — ${configured ? 'Ready' : 'Not Configured'}</p>
           </div>`;

    // ── Detection stats bar (visible when running) ───────────────
    const statsBar = `
        <div class="det-stats" id="stats-${cam.camera_id}" style="${isRunning ? '' : 'display:none'}">
            <span class="dstat green" id="fps-${cam.camera_id}">0 fps</span>
            <span class="dstat" id="det-${cam.camera_id}">0 det</span>
        </div>`;

    // ── Bottom bar ────────────────────────────────────────────────
    const badgeHtml = isRunning
        ? `<span class="live-badge lb-running">RUNNING</span>`
        : configured
            ? `<span class="live-badge lb-warn">IDLE</span>`
            : `<span class="live-badge lb-offline">OFFLINE</span>`;

    const dotCls = isRunning ? 'on' : configured ? 'wn' : 'off';

    tile.innerHTML = `
        ${feedOrPlaceholder}
        ${statsBar}

        <!-- Detection panel (gear opens this) -->
        <div class="det-panel" id="panel-${cam.camera_id}">
            <div class="dp-head">
                <span class="dp-title">⚙ ${cam.label} — Detection</span>
                <button class="dp-close" onclick="togglePanel('${cam.camera_id}', false)">✕</button>
            </div>
            <div class="dp-label">Select Models</div>
            <div class="model-list" id="modellist-${cam.camera_id}">
                ${buildModelCheckboxes(cam.camera_id, cam.model_keys)}
            </div>
            <div class="dp-footer">
                <button class="dp-btn dp-btn-start" id="startbtn-${cam.camera_id}"
                    onclick="startRtsp('${cam.camera_id}')"
                    ${!configured || isRunning ? 'disabled' : ''}>
                    ▶ Start
                </button>
                <button class="dp-btn dp-btn-stop" id="stopbtn-${cam.camera_id}"
                    onclick="stopRtsp('${cam.camera_id}')"
                    ${!isRunning ? 'disabled' : ''}>
                    ⏹ Stop
                </button>
            </div>
            <div class="dp-status" id="panelstatus-${cam.camera_id}">
                ${!configured ? '⚠ No RTSP URL set in config.py' : ''}
            </div>
        </div>

        <div class="cam-bar">
            <span class="cam-label" id="cbar-${cam.camera_id}">
                <span class="dot ${dotCls}" id="bardot-${cam.camera_id}"></span>
                ${cam.label}
                <span id="badge-${cam.camera_id}">${badgeHtml}</span>
            </span>
            <span class="gear-btn" onclick="togglePanel('${cam.camera_id}', true)" title="Configure Detection">⚙</span>
        </div>`;

    return tile;
}


function buildModelCheckboxes(camera_id, activeKeys = []) {
    return Object.entries(availableModels).map(([key, info]) => {
        const checked  = activeKeys.includes(key);
        const classes  = info.target_classes.join(', ');
        const boundary = info.has_boundary ? ' 🔷' : '';
        return `
            <label class="model-item ${checked ? 'checked' : ''}" id="mi-${camera_id}-${key}">
                <input type="checkbox" value="${key}" ${checked ? 'checked' : ''}
                    onchange="onModelCheck('${camera_id}', '${key}', this.checked)">
                <div class="model-item-text">
                    <div class="model-name">${key}${boundary}</div>
                    <div class="model-classes">${classes}</div>
                </div>
            </label>`;
    }).join('');
}


function onModelCheck(camera_id, key, checked) {
    const mi = document.getElementById(`mi-${camera_id}-${key}`);
    if (mi) mi.classList.toggle('checked', checked);
}


function togglePanel(camera_id, open) {
    const panel = document.getElementById(`panel-${camera_id}`);
    if (!panel) return;
    panel.classList.toggle('open', open);
}


// ==================== RTSP START / STOP ====================

async function startRtsp(camera_id) {
    const startBtn = document.getElementById(`startbtn-${camera_id}`);
    const stopBtn2 = document.getElementById(`stopbtn-${camera_id}`);
    const statusEl = document.getElementById(`panelstatus-${camera_id}`);

    // Collect checked model keys
    const checkboxes = document.querySelectorAll(`#modellist-${camera_id} input[type=checkbox]:checked`);
    const model_keys = Array.from(checkboxes).map(c => c.value);

    if (model_keys.length === 0) {
        setTileStatus(camera_id, '⚠ Select at least one model', 'err');
        return;
    }

    startBtn.disabled = true;
    setTileStatus(camera_id, '⏳ Starting...', '');

    try {
        const res = await fetch(`/rtsp/start/${camera_id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_keys })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Start failed');
        }

        setTileRunning(camera_id, true, model_keys);
        setTileStatus(camera_id, `✔ Running: ${model_keys.join(', ')}`, 'ok');
        togglePanel(camera_id, false);

    } catch (e) {
        setTileStatus(camera_id, `✖ ${e.message}`, 'err');
        startBtn.disabled = false;
    }
}


async function stopRtsp(camera_id) {
    const stopBtn2 = document.getElementById(`stopbtn-${camera_id}`);
    stopBtn2.disabled = true;
    setTileStatus(camera_id, '⏳ Stopping...', '');

    try {
        await fetch(`/rtsp/stop/${camera_id}`, { method: 'POST' });
        setTileRunning(camera_id, false, []);
        setTileStatus(camera_id, '⏹ Stopped', '');
    } catch (e) {
        setTileStatus(camera_id, `✖ ${e.message}`, 'err');
        stopBtn2.disabled = false;
    }
}


function setTileRunning(camera_id, running, model_keys) {
    rtspState[camera_id].isRunning = running;

    const tile        = document.getElementById(`tile-${camera_id}`);
    const startBtn    = document.getElementById(`startbtn-${camera_id}`);
    const stopBtn2    = document.getElementById(`stopbtn-${camera_id}`);
    const badge       = document.getElementById(`badge-${camera_id}`);
    const bardot      = document.getElementById(`bardot-${camera_id}`);
    const statsBar    = document.getElementById(`stats-${camera_id}`);
    const placeholder = document.getElementById(`placeholder-${camera_id}`);

    if (running) {
        // Swap placeholder for stream image
        if (placeholder) {
            const img = document.createElement('img');
            img.className = 'cam-feed';
            img.id        = `feed-${camera_id}`;
            img.src       = `/rtsp/stream/${camera_id}`;
            img.onerror   = () => onFeedError(camera_id);
            placeholder.replaceWith(img);
        } else {
            const img = document.getElementById(`feed-${camera_id}`);
            if (img) img.src = `/rtsp/stream/${camera_id}`;
        }

        if (badge)   badge.innerHTML   = '<span class="live-badge lb-running">RUNNING</span>';
        if (bardot)  { bardot.className = 'dot on'; }
        if (statsBar) statsBar.style.display = '';
        if (startBtn) startBtn.disabled = true;
        if (stopBtn2) stopBtn2.disabled = false;

        // Update sidebar dot
        const cdot = document.getElementById(`cdot-${camera_id}`);
        if (cdot) cdot.className = 'dot on';

    } else {
        // Swap stream image back to placeholder
        const img = document.getElementById(`feed-${camera_id}`);
        if (img) {
            const ph = document.createElement('div');
            ph.className = 'cam-placeholder';
            ph.id        = `placeholder-${camera_id}`;
            ph.innerHTML = '<div class="big">📹</div><p>Detection stopped</p>';
            img.replaceWith(ph);
        }

        if (badge)    badge.innerHTML  = '<span class="live-badge lb-warn">IDLE</span>';
        if (bardot)   { bardot.className = 'dot wn'; }
        if (statsBar) statsBar.style.display = 'none';
        if (startBtn) startBtn.disabled = false;
        if (stopBtn2) stopBtn2.disabled = true;

        const cdot = document.getElementById(`cdot-${camera_id}`);
        if (cdot) cdot.className = 'dot wn';
    }
}


function setTileStatus(camera_id, msg, type) {
    const el = document.getElementById(`panelstatus-${camera_id}`);
    if (!el) return;
    el.textContent = msg;
    el.className   = `dp-status ${type || ''}`;
}


function onFeedError(camera_id) {
    // Stream dropped — revert to placeholder
    setTileRunning(camera_id, false, []);
    setTileStatus(camera_id, '⚠ Stream lost', 'err');
    togglePanel(camera_id, true);
}


// ==================== RTSP STATUS POLLING ====================
async function pollAllRtspStatuses() {
    for (const camera_id of Object.keys(rtspState)) {
        try {
            const res    = await fetch(`/rtsp/status/${camera_id}`);
            const status = await res.json();

            const wasRunning = rtspState[camera_id].isRunning;

            // ── Camera became running (started via dashboard or externally) ──
            if (!wasRunning && status.is_running) {
                setTileRunning(camera_id, true, status.model_key ? [status.model_key] : []);
                setTileStatus(camera_id, `● ${status.model_key || 'Running'}`, '');
            }

            // ── Camera stopped or errored while UI thought it was running ──
            if (wasRunning && !status.is_running) {
                const errMsg = status.error ? `⚠ ${status.error}` : '⏹ Stopped';
                setTileRunning(camera_id, false, []);
                setTileStatus(camera_id, errMsg, status.error ? 'err' : '');
            }

        } catch (e) { /* silently skip */ }
    }
}


// ==================== VIDEO UPLOAD ====================
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    document.getElementById('uploadTile').classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
    document.getElementById('uploadTile').classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    document.getElementById('uploadTile').classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
});

async function handleFileSelect(file) {
    if (!file.type.startsWith('video/')) { alert('Please select a video file'); return; }
    uploadText.innerHTML = `<strong>${file.name}</strong><br>Uploading...`;
    predictBtn.disabled  = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data     = await response.json();
        uploadedVideoPath = data.path;
        uploadText.innerHTML = `<strong>✅ ${file.name}</strong><br><span style="font-size:0.85em;color:#00a891;">Ready for detection</span>`;
        predictBtn.disabled  = false;
    } catch (error) {
        uploadText.innerHTML = `<strong>❌ Upload failed</strong><br>${error.message}`;
        predictBtn.disabled  = true;
    }
}


// ==================== PREDICTION ====================
predictBtn.addEventListener('click', startPrediction);
stopBtn.addEventListener('click', stopDetection);

async function startPrediction() {
    if (!uploadedVideoPath) { alert('Please upload a video first'); return; }
    predictBtn.disabled = true;
    stopBtn.disabled    = false;
    clearLogs();
    addLog('🚀 Starting detection pipeline...', 'suc');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_path: uploadedVideoPath })
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Prediction failed to start');
        }
        const data    = await response.json();
        currentTaskId = data.task_id;
        taskIdSpan.textContent = currentTaskId;
        updateTaskStatus('running');
        activeTasksEl.textContent = '1';
        startLogStreaming(currentTaskId);
        pollTaskStatus(currentTaskId);
    } catch (error) {
        addLog(`❌ Error: ${error.message}`, 'e2');
        predictBtn.disabled = false;
        stopBtn.disabled    = true;
        currentTaskId       = null;
    }
}

async function stopDetection() {
    if (!currentTaskId) return;
    stopBtn.disabled = true;
    addLog('⏹ Stop requested — finishing current frame...', 'w');
    try {
        const response = await fetch(`/stop/${currentTaskId}`, { method: 'POST' });
        const data     = await response.json();
        addLog(`⏹ ${data.message}`, 'w');
    } catch (error) {
        addLog(`❌ Stop request failed: ${error.message}`, 'e2');
        stopBtn.disabled = false;
    }
}

function resetControls() {
    predictBtn.disabled = false;
    stopBtn.disabled    = true;
    activeTasksEl.textContent = '0';
}


// ==================== LOG STREAMING ====================
function startLogStreaming(taskId) {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`/logs/${taskId}`);

    eventSource.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.log) {
            const { text, type } = classifyLog(data.log);
            addLog(text, type);
        }
        if (data.done) {
            eventSource.close();
            const label = data.status === 'stopped' ? '⏹ Detection stopped by user' : '✅ Detection pipeline complete';
            addLog(label, data.status === 'stopped' ? 'w' : 'suc');
        }
    };
    eventSource.onerror = () => eventSource.close();
}

function classifyLog(line) {
    if (/\[CLASSIFIER\]/.test(line))           return { text: line, type: 'c'   };
    if (/\[AGENT-BRAIN\]/.test(line))          return { text: line, type: 'ab'  };
    if (/\[AGENT\]/.test(line))                return { text: line, type: 'a'   };
    if (/\[DETECTION\]/.test(line))            return { text: line, type: 'd'   };
    if (/\[PIPELINE\]/.test(line))             return { text: line, type: 'p'   };
    if (/STEP \d/.test(line))                  return { text: line, type: 's'   };
    if (/ERROR|✖|failed|Failed/.test(line))    return { text: line, type: 'e2'  };
    if (/✔|complete|selected|done/.test(line)) return { text: line, type: 'suc' };
    if (/⏹|Stop|stop|WARNING/.test(line))      return { text: line, type: 'w'   };
    if (/={5,}/.test(line))                    return { text: line, type: 'div' };
    return { text: line, type: 'n' };
}

const LOG_CLASS = {
    n:'ln', c:'lc', ab:'lab', a:'la', d:'ld', p:'lp',
    s:'ls', e2:'le2', suc:'lsuc', w:'lw', div:'ldiv'
};

function clearLogs() { logsContainer.innerHTML = ''; }
function addLog(message, type = 'n') {
    const entry    = document.createElement('div');
    entry.className = `le ${LOG_CLASS[type] || 'ln'}`;
    entry.textContent = message;
    logsContainer.appendChild(entry);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}


// ==================== STATUS POLLING ====================
function pollTaskStatus(taskId) {
    if (statusPollInterval) clearInterval(statusPollInterval);
    statusPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/status/${taskId}`);
            const data     = await response.json();
            updateTaskStatus(data.status);

            if (data.classifier_output) {
                classifierResultDiv.classList.remove('hidden');
                const detected = data.classifier_output.detected || [];
                detectedItemsSpan.innerHTML = detected.map(i => `<span class="dtag">${i}</span>`).join('');
            }
            if (data.selected_models) {
                selectedModelsSpan.innerHTML = data.selected_models.map(m => `<span class="mtag">${m}</span>`).join('');
            }
            if (data.status === 'completed') {
                clearInterval(statusPollInterval);
                resetControls();
                if (data.output_video) showOutputVideo(taskId);
                loadAnalytics();
                loadHistory();
            }
            if (data.status === 'stopped') {
                clearInterval(statusPollInterval);
                resetControls();
                addLog('⏹ Task stopped — no output video saved.', 'w');
                loadAnalytics();
                loadHistory();
            }
            if (data.status === 'failed') {
                clearInterval(statusPollInterval);
                resetControls();
                addLog(`❌ Task failed: ${data.error}`, 'e2');
                loadHistory();
            }
        } catch (error) {
            clearInterval(statusPollInterval);
        }
    }, 1000);
}

function updateTaskStatus(status) {
    const label = status ? status.toUpperCase() : 'UNKNOWN';
    taskStatusSpan.innerHTML = `<span class="sbadge s-${status || 'pending'}">${label}</span>`;
}


// ==================== OUTPUT VIDEO ====================
function showOutputVideo(taskId) {
    outputSection.classList.remove('hidden');
    outputVideo.src   = `/download/${taskId}`;
    downloadBtn.onclick = () => window.open(`/download/${taskId}`, '_blank');
}


// ==================== ANALYTICS ====================
async function loadAnalytics() {
    try {
        const response = await fetch('/analytics');
        const data     = await response.json();
        totalDetectionsEl.textContent = data.total_detections;
        const done  = data.status_counts.completed || 0;
        const total = done + (data.status_counts.failed || 0) + (data.status_counts.stopped || 0);
        successRateEl.textContent = total > 0 ? `${Math.round((done / total) * 100)}%` : '0%';
        updateModelUsageChart(data.model_usage);
    } catch (error) { console.error('Analytics error:', error); }
}

function updateModelUsageChart(modelUsage) {
    const ctx = document.getElementById('modelUsageChart').getContext('2d');
    if (modelUsageChart) modelUsageChart.destroy();
    const labels = Object.keys(modelUsage);
    const values = Object.values(modelUsage);
    modelUsageChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Times Used',
                data: values,
                backgroundColor: [
                    'rgba(61,90,241,0.72)', 'rgba(0,194,168,0.72)',
                    'rgba(247,99,77,0.72)', 'rgba(245,166,35,0.72)',
                    'rgba(118,75,162,0.72)'
                ],
                borderColor: [
                    'rgba(61,90,241,1)', 'rgba(0,194,168,1)',
                    'rgba(247,99,77,1)', 'rgba(245,166,35,1)',
                    'rgba(118,75,162,1)'
                ],
                borderWidth: 2, borderRadius: 7
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, ticks: { stepSize:1, color:'#6b7a99' }, grid: { color:'#e2e8f8' } },
                x: { ticks: { color:'#6b7a99' }, grid: { display: false } }
            }
        }
    });
}


// ==================== HISTORY ====================
async function loadHistory() {
    try {
        const response = await fetch('/history');
        const history  = await response.json();

        if (!history.length) {
            historyBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#6b7a99;padding:28px;">No detection history yet</td></tr>';
            return;
        }
        historyBody.innerHTML = history.slice(-10).reverse().map(record => {
            const detected    = record.classifier_output?.detected || [];
            const models      = record.selected_models || [];
            const completedAt = record.completed_at ? new Date(record.completed_at).toLocaleString() : '—';
            return `<tr>
                <td><code>${record.task_id}</code></td>
                <td>${record.video_name}</td>
                <td><span class="sbadge s-${record.status}">${record.status.toUpperCase()}</span></td>
                <td>${detected.map(d => `<span class="dtag">${d}</span>`).join('') || '—'}</td>
                <td>${models.map(m => `<span class="mtag">${m}</span>`).join('') || '—'}</td>
                <td>${completedAt}</td>
            </tr>`;
        }).join('');
    } catch (error) { console.error('History error:', error); }
}

document.getElementById('clearHistoryBtn').addEventListener('click', async () => {
    if (!confirm('Clear all detection history?')) return;
    await fetch('/history', { method: 'DELETE' });
    loadHistory();
    loadAnalytics();
});