// ==================== DASHBOARD.JS ====================

let currentTaskId     = null;
let uploadedVideoPath = null;
let eventSource       = null;
let modelUsageChart   = null;
let statusPollInterval = null;

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


// ==================== UPLOAD ====================
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
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();
        uploadedVideoPath = data.path;
        uploadText.innerHTML = `<strong>✅ ${file.name}</strong><br><span style="font-size:0.85em;color:#00a891;">Ready for detection</span>`;
        predictBtn.disabled = false;
    } catch (error) {
        uploadText.innerHTML = `<strong>❌ Upload failed</strong><br>${error.message}`;
        predictBtn.disabled = true;
    }
}


// ==================== PREDICTION ====================
predictBtn.addEventListener('click', startPrediction);
stopBtn.addEventListener('click', stopDetection);

async function startPrediction() {
    if (!uploadedVideoPath) { alert('Please upload a video first'); return; }

    predictBtn.disabled = true;
    stopBtn.disabled = false;
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

        const data = await response.json();
        currentTaskId = data.task_id;

        taskIdSpan.textContent = currentTaskId;
        updateTaskStatus('running');
        activeTasksEl.textContent = '1';

        startLogStreaming(currentTaskId);
        pollTaskStatus(currentTaskId);

    } catch (error) {
        addLog(`❌ Error: ${error.message}`, 'e2');
        predictBtn.disabled = false;
        stopBtn.disabled = true;
        currentTaskId = null;
    }
}

async function stopDetection() {
    if (!currentTaskId) return;
    stopBtn.disabled = true;
    addLog('⏹ Stop requested — finishing current frame...', 'w');
    try {
        const response = await fetch(`/stop/${currentTaskId}`, { method: 'POST' });
        const data = await response.json();
        addLog(`⏹ ${data.message}`, 'w');
    } catch (error) {
        addLog(`❌ Stop request failed: ${error.message}`, 'e2');
        stopBtn.disabled = false;
    }
}

function resetControls() {
    predictBtn.disabled = false;
    stopBtn.disabled = true;
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
    if (/\[CLASSIFIER\]/.test(line))           return { text: line, type: 'c'  };
    if (/\[AGENT-BRAIN\]/.test(line))          return { text: line, type: 'ab' };
    if (/\[AGENT\]/.test(line))                return { text: line, type: 'a'  };
    if (/\[DETECTION\]/.test(line))            return { text: line, type: 'd'  };
    if (/\[PIPELINE\]/.test(line))             return { text: line, type: 'p'  };
    if (/STEP \d/.test(line))                  return { text: line, type: 's'  };
    if (/ERROR|✖|failed|Failed/.test(line))    return { text: line, type: 'e2' };
    if (/✔|complete|selected|done/.test(line)) return { text: line, type: 'suc'};
    if (/⏹|Stop|stop|WARNING/.test(line))      return { text: line, type: 'w'  };
    if (/={5,}/.test(line))                    return { text: line, type: 'div'};
    return { text: line, type: 'n' };
}

// Map short type codes to CSS classes
const LOG_CLASS = {
    n:'ln', c:'lc', ab:'lab', a:'la', d:'ld', p:'lp',
    s:'ls', e2:'le2', suc:'lsuc', w:'lw', div:'ldiv'
};

function clearLogs() { logsContainer.innerHTML = ''; }

function addLog(message, type = 'n') {
    const entry = document.createElement('div');
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
            const data = await response.json();

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
    outputVideo.src = `/download/${taskId}`;
    downloadBtn.onclick = () => window.open(`/download/${taskId}`, '_blank');
}


// ==================== ANALYTICS ====================
async function loadAnalytics() {
    try {
        const response = await fetch('/analytics');
        const data = await response.json();

        totalDetectionsEl.textContent = data.total_detections;

        const done  = data.status_counts.completed || 0;
        const total = done + (data.status_counts.failed || 0) + (data.status_counts.stopped || 0);
        successRateEl.textContent = total > 0 ? `${Math.round((done / total) * 100)}%` : '0%';

        updateModelUsageChart(data.model_usage);
    } catch (error) {
        console.error('Analytics error:', error);
    }
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
                borderWidth: 2,
                borderRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
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
            return `
                <tr>
                    <td><code>${record.task_id}</code></td>
                    <td>${record.video_name}</td>
                    <td><span class="sbadge s-${record.status}">${record.status.toUpperCase()}</span></td>
                    <td>${detected.map(d => `<span class="dtag">${d}</span>`).join('') || '—'}</td>
                    <td>${models.map(m => `<span class="mtag">${m}</span>`).join('') || '—'}</td>
                    <td>${completedAt}</td>
                </tr>`;
        }).join('');

    } catch (error) {
        console.error('History error:', error);
    }
}


// ==================== INIT ====================
document.addEventListener('DOMContentLoaded', () => {
    loadAnalytics();
    loadHistory();
    setInterval(loadAnalytics, 10000);
});

document.getElementById('clearHistoryBtn').addEventListener('click', async () => {
    if (!confirm('Clear all detection history?')) return;
    await fetch('/history', { method: 'DELETE' });
    loadHistory();
    loadAnalytics();
});