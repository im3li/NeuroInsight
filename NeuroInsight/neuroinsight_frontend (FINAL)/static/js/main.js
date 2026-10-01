const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadInner = document.getElementById('upload-inner');
const previewImg = document.getElementById('preview-img');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const resetBtn = document.getElementById('reset-btn');

let selectedFile = null;

const SEVERITY_COLORS = {
  low: 'var(--severity-low)',
  moderate: 'var(--severity-moderate)',
  high: 'var(--severity-high)',
};

// --- Upload interactions ---

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    uploadInner.hidden = true;
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// --- Analyze ---

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  loadingOverlay.hidden = false;
  document.getElementById('loading-text').textContent = 'Analyzing scan…';

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      alert('Error: ' + data.error);
      loadingOverlay.hidden = true;
      return;
    }

    renderResults(data);
    document.getElementById('upload-section').hidden = true;
    resultsSection.hidden = false;
  } catch (err) {
    alert('Could not reach the server. Is app.py running?');
  } finally {
    loadingOverlay.hidden = true;
  }
});

function updateConfidenceRing(percent) {
  const ring = document.getElementById('confidence-ring');
  const circumference = 2 * Math.PI * 54; // r=54
  const offset = circumference - (percent / 100) * circumference;
  ring.style.strokeDasharray = circumference;
  ring.style.strokeDashoffset = offset;
  // color gradient based on value
  if (percent < 50) {
    ring.style.stroke = '#F0563D'; // red
  } else if (percent < 75) {
    ring.style.stroke = '#F5A623'; // orange
  } else {
    ring.style.stroke = 'var(--accent)'; // teal
  }
}

function animateCount(el, target, duration = 700) {
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = eased * target;
    el.textContent = Math.round(current) + '%';
    updateConfidenceRing(current);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function renderResults(data) {
  document.getElementById('result-original').src = 'data:image/png;base64,' + data.original_image;
  document.getElementById('result-gradcam').src = 'data:image/png;base64,' + data.gradcam_image;

  document.getElementById('prediction-name').textContent = data.display_name;

  const confidenceEl = document.getElementById('confidence-value');
  animateCount(confidenceEl, data.confidence);

  const dot = document.getElementById('severity-dot');
  dot.style.background = SEVERITY_COLORS[data.severity] || 'var(--text-muted)';

  const verdictBlock = document.getElementById('verdict-block');
  let badge = document.getElementById('severity-badge');
  if (!badge) {
    badge = document.createElement('span');
    badge.id = 'severity-badge';
    badge.className = 'severity-badge';
    verdictBlock.insertBefore(badge, dot.nextSibling);
  }
  badge.className = 'severity-badge ' + data.severity;
  const severityLabel = { low: 'Low Risk', moderate: 'Elevated Risk', high: 'Severe Risk' };
  badge.textContent = severityLabel[data.severity] || data.severity;

  document.getElementById('report-description').textContent = data.description;
  document.getElementById('report-recommendation').textContent = data.recommendation;
  document.getElementById('report-id').textContent = 'REF-' + Date.now().toString().slice(-8);

  const thresholdPct = (data.nondemented_threshold * 100).toFixed(1);
  document.getElementById('uncertainty-text').textContent =
    `This model outputs a probability for each of the 3 classes; the highest one is normally selected. ` +
    `One exception: the Non-Demented class uses an adjusted decision threshold (${thresholdPct}%, tuned during ` +
    `evaluation to improve reliability) rather than a simple highest-probability rule. Probabilities shown ` +
    `above are the model's raw output before that adjustment is applied.`;

  window._lastResult = data;

  const probBars = document.getElementById('prob-bars');
  probBars.innerHTML = '';
  const sorted = [...data.class_probabilities].sort((a, b) => b.probability - a.probability);
  sorted.forEach((cls, i) => {
    const row = document.createElement('div');
    row.className = 'prob-bar-row' + (cls.class === data.prediction ? ' is-winner' : '');
    row.innerHTML = `
      <div class="prob-bar-label"><span>${cls.display_name}</span><span>${cls.probability.toFixed(1)}%</span></div>
      <div class="prob-bar-track"><div class="prob-bar-fill" style="width:0%"></div></div>
    `;
    probBars.appendChild(row);
    const fill = row.querySelector('.prob-bar-fill');
    setTimeout(() => { fill.style.width = cls.probability + '%'; }, 50 + i * 80);
  });

  initSlider();
}

// --- Compare slider (drag to reveal Grad-CAM) ---

function initSlider() {
  const slider = document.getElementById('compare-slider');
  const overlayWrap = document.getElementById('compare-overlay-wrap');
  const handle = document.getElementById('compare-handle');
  let dragging = false;

  function setPosition(clientX) {
    const rect = slider.getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    overlayWrap.style.width = pct + '%';
    handle.style.left = pct + '%';
  }

  slider.onmousedown = (e) => { dragging = true; setPosition(e.clientX); };
  window.onmousemove = (e) => { if (dragging) setPosition(e.clientX); };
  window.onmouseup = () => { dragging = false; };

  slider.ontouchstart = (e) => { dragging = true; setPosition(e.touches[0].clientX); };
  slider.ontouchmove = (e) => { if (dragging) setPosition(e.touches[0].clientX); };
  slider.ontouchend = () => { dragging = false; };

  // Set default to 50%
  const rect = slider.getBoundingClientRect();
  setPosition(rect.left + rect.width / 2);
}

// --- Reset ---

resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.hidden = true;
  uploadInner.hidden = false;
  analyzeBtn.disabled = true;
  resultsSection.hidden = true;
  document.getElementById('upload-section').hidden = false;
  document.getElementById('intensity-slider').value = 100;
  document.getElementById('intensity-value').textContent = '100%';
  document.getElementById('result-gradcam').style.opacity = 1;
  // reset confidence ring
  const ring = document.getElementById('confidence-ring');
  ring.style.strokeDashoffset = 0;
  ring.style.stroke = 'var(--accent)';
  document.getElementById('confidence-value').textContent = '0%';
});

// --- Overlay intensity slider ---

const intensitySlider = document.getElementById('intensity-slider');
const intensityValue = document.getElementById('intensity-value');

intensitySlider.addEventListener('input', () => {
  const val = intensitySlider.value;
  intensityValue.textContent = val + '%';
  document.getElementById('result-gradcam').style.opacity = val / 100;
});

// --- Copy summary ---

document.getElementById('copy-btn').addEventListener('click', () => {
  const data = window._lastResult;
  if (!data) return;
  const summary =
    `NeuroInsight Analysis Summary (${document.getElementById('report-id').textContent})\n` +
    `Classification: ${data.display_name}\n` +
    `Confidence: ${data.confidence.toFixed(1)}%\n` +
    `Risk Level: ${data.severity.charAt(0).toUpperCase() + data.severity.slice(1)}\n\n` +
    `Finding: ${data.description}\n\n` +
    `Recommendation: ${data.recommendation}\n\n` +
    `Probability breakdown:\n` +
    data.class_probabilities.map(c => `  - ${c.display_name}: ${c.probability.toFixed(1)}%`).join('\n') +
    `\n\nNote: This is a research/educational prototype. Not a diagnostic device.`;

  navigator.clipboard.writeText(summary).then(() => {
    const confirmEl = document.getElementById('copy-confirm');
    confirmEl.hidden = false;
    setTimeout(() => { confirmEl.hidden = true; }, 2000);
  });
});