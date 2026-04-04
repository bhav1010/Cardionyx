/* ============================================================
   CARDIONYX — FRONTEND JAVASCRIPT
   ============================================================ */

/* ──────────────────────────────────────────
   BMI AUTO-CALCULATION
   ────────────────────────────────────────── */
const weightInput = document.getElementById('weight');
const heightInput = document.getElementById('height');
const bmiInput    = document.getElementById('bmi');

function calcBMI() {
  const w = parseFloat(weightInput.value);
  const h = parseFloat(heightInput.value) / 100; // cm → m
  if (w > 0 && h > 0) {
    bmiInput.value = (w / (h * h)).toFixed(2);
  } else {
    bmiInput.value = '';
  }
}

weightInput.addEventListener('input', calcBMI);
heightInput.addEventListener('input', calcBMI);

/* ──────────────────────────────────────────
   TOGGLE BUTTONS (Yes / No)
   ────────────────────────────────────────── */
document.querySelectorAll('.toggle-group').forEach(group => {
  const btns = group.querySelectorAll('.toggle-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => { b.classList.remove('active', 'yes-active'); });
      btn.classList.add('active');

      const fieldName = btn.dataset.field;
      const value     = btn.dataset.value;

      // Mark yes-variants differently for style
      const isYes = (value === '1' || value === 'Y');
      if (isYes) btn.classList.add('yes-active');

      // Update the hidden input
      const hiddenInputId = fieldToId(fieldName);
      const hidden = document.getElementById(hiddenInputId);
      if (hidden) hidden.value = value;
    });
  });
});

/** Map field name → hidden input id (same heuristic as HTML) */
function fieldToId(name) {
  // Direct matches first
  const map = {
    'DM': 'DM', 'HTN': 'HTN', 'FH': 'FH',
    'Obesity': 'Obesity', 'CRF': 'CRF', 'CVA': 'CVA',
    'CHF': 'CHF', 'DLP': 'DLP', 'Edema': 'Edema',
    'Dyspnea': 'Dyspnea', 'Atypical': 'Atypical', 'Nonanginal': 'Nonanginal',
    'Current Smoker': 'current-smoker',
    'EX-Smoker': 'ex-smoker',
    'Airway disease': 'airway-disease',
    'Thyroid Disease': 'thyroid-disease',
    'Weak Peripheral Pulse': 'weak-peripheral-pulse',
    'Lung rales': 'lung-rales',
    'Systolic Murmur': 'systolic-murmur',
    'Diastolic Murmur': 'diastolic-murmur',
    'Typical Chest Pain': 'typical-chest-pain',
    'Exertional CP': 'exertional-cp',
    'LowTH Ang': 'lowth-ang',
  };
  return map[name] || name.toLowerCase().replace(/\s+/g, '-');
}

/* ──────────────────────────────────────────
   ECG IMAGE UPLOAD
   ────────────────────────────────────────── */
const ecgInput     = document.getElementById('ecgImage');
const dropZone     = document.getElementById('ecgDropZone');
const uploadContent = document.getElementById('uploadContent');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg   = document.getElementById('previewImg');
const previewName  = document.getElementById('previewName');
const removeBtn    = document.getElementById('removeImage');

ecgInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearECGImage();
});

// Drag-and-drop
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].type.startsWith('image/')) {
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    ecgInput.files = dt.files;
    showPreview(files[0]);
  }
});

function handleFileSelect() {
  const file = ecgInput.files[0];
  if (file) showPreview(file);
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    uploadContent.style.display = 'none';
    uploadPreview.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

function clearECGImage() {
  ecgInput.value = '';
  previewImg.src = '';
  uploadContent.style.display = '';
  uploadPreview.style.display = 'none';
}

/* ──────────────────────────────────────────
   FORM SUBMISSION
   ────────────────────────────────────────── */
const form        = document.getElementById('predictionForm');
const submitBtn   = document.getElementById('submitBtn');
const btnSpinner  = document.getElementById('btnSpinner');
const btnText     = submitBtn.querySelector('.btn-text');
const resultPanel = document.getElementById('resultPanel');
const errorBanner = document.getElementById('errorBanner');
const errorMsg    = document.getElementById('errorMsg');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Basic validation
  const requiredInputs = form.querySelectorAll('[required]');
  let valid = true;
  requiredInputs.forEach(inp => {
    if (!inp.value.trim()) {
      inp.classList.add('invalid');
      valid = false;
    } else {
      inp.classList.remove('invalid');
    }
  });
  if (!valid) {
    showError('Please fill in all required fields (Age, Weight, Height, Sex).');
    return;
  }

  setLoading(true);
  hideError();
  resultPanel.style.display = 'none';

  try {
    const formData = new FormData(form);
    // Append BMI if auto-calculated
    if (bmiInput.value) formData.set('BMI', bmiInput.value);

    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || 'Prediction failed. Please check your inputs.');
      return;
    }

    displayResult(data);
  } catch (err) {
    showError('Network error — make sure the server is running.');
  } finally {
    setLoading(false);
  }
});

/* ──────────────────────────────────────────
   DISPLAY RESULT
   ────────────────────────────────────────── */
function displayResult(data) {
  const { probability, risk_level, prediction, ecg_detections, confidence_metadata } = data;
  const pct  = Math.round(probability * 100);
  const risk = risk_level.toLowerCase();  // low / moderate / high

  // Gauge
  const gaugeCircle = document.getElementById('gaugeCircle');
  const gaugePct    = document.getElementById('gaugePct');
  gaugeCircle.className = `gauge-circle risk-${risk}`;
  gaugePct.textContent  = `${pct}%`;

  // Risk badge
  const riskBadge = document.getElementById('riskBadge');
  riskBadge.className = `result-risk-badge ${risk}`;
  riskBadge.textContent = `${risk_level} RISK`;

  // Verdict
  const verdict = document.getElementById('resultVerdict');
  verdict.textContent = prediction === 1
    ? '🫀 CAD Likely Detected'
    : '✅ Normal — CAD Unlikely';
  verdict.style.color = prediction === 1 ? 'var(--danger)' : 'var(--success)';

  // ECG findings
  const findingsGrid = document.getElementById('findingsGrid');
  findingsGrid.innerHTML = '';
  for (const [feat, val] of Object.entries(ecg_detections || {})) {
    const conf  = (confidence_metadata || {})[feat] || 0;
    const detected = val === 1;
    const item  = document.createElement('div');
    item.className = `finding-item ${detected ? 'detected' : 'not-detected'}`;
    item.innerHTML = `
      <span class="finding-dot"></span>
      <span>${feat}${detected ? ` <small>(${(conf * 100).toFixed(0)}%)</small>` : ''}</span>
    `;
    findingsGrid.appendChild(item);
  }

  // Show panel
  resultPanel.style.display = 'block';
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ──────────────────────────────────────────
   CLOSE RESULT PANEL
   ────────────────────────────────────────── */
document.getElementById('closeResult').addEventListener('click', () => {
  resultPanel.style.display = 'none';
});

/* ──────────────────────────────────────────
   HELPERS
   ────────────────────────────────────────── */
function setLoading(on) {
  submitBtn.disabled = on;
  btnText.textContent = on ? 'Analysing…' : 'Run CAD Risk Prediction';
  btnSpinner.style.display = on ? 'inline-flex' : 'none';
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.style.display = 'flex';
  errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
  errorBanner.style.display = 'none';
}

// Remove invalid class on input
document.querySelectorAll('.field-input').forEach(inp => {
  inp.addEventListener('input', () => inp.classList.remove('invalid'));
});
