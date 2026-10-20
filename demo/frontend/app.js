// Plain JavaScript: pick an image, send it to the Python server, show the result,
// and browse the frozen thesis results. No build step or external library.
const $ = (id) => document.getElementById(id);
let busy = false;
let previewUrl = null;
let current = null;        // { file: File, sample: sample-object-or-null }
let resultRows = [];
let activeRq = 'All';
// Plain names for the HAM10000 diagnosis codes used by the examples.
const DX_NAMES = { nv: 'common mole', bkl: 'benign keratosis', mel: 'melanoma' };

/* ---------- Status pill: shows whether the verified model is loaded ---------- */
async function refreshStatus() {
  try {
    const status = await (await fetch('/api/status')).json();
    const device = status.device === 'cuda' ? 'GPU' : 'CPU';
    $('status').className = 'status' + (status.loaded ? ' ready' : '');
    $('status-text').textContent = status.loaded ? `AI ready · ${device}` : `Starting AI · ${device}`;
    if (!status.loaded) setTimeout(refreshStatus, 2000);
  } catch {
    $('status').className = 'status error';
    $('status-text').textContent = 'Server not reachable';
  }
}

/* ---------- Tabs ---------- */
// Only visible tabs take part; the hidden research-results tab stays in the code.
const tabs = [...document.querySelectorAll('[role="tab"]')].filter((t) => !t.hidden);
function selectTab(tab) {
  tabs.forEach((t) => {
    const on = t === tab;
    t.setAttribute('aria-selected', on);
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute('aria-controls')).hidden = !on;
  });
  if (tab.id === 'tab-results' && !resultRows.length) loadResults();
}
tabs.forEach((tab, i) => {
  tab.onclick = () => selectTab(tab);
  tab.onkeydown = (e) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
    selectTab(next);
    next.focus();
  };
});

/* ---------- Choosing an image ---------- */
function setProgress(text, isError = false) {
  $('progress').textContent = text;
  $('progress').classList.toggle('error', isError);
}

function useImage(file, sample) {
  current = file ? { file, sample } : null;
  // Release the previous preview from browser memory.
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : null;
  $('input-preview').hidden = !file;
  if (previewUrl) $('input-preview').src = previewUrl;
  else $('input-preview').removeAttribute('src');
  document.querySelectorAll('#samples button').forEach((b) => b.setAttribute('aria-pressed', sample && b.dataset.id === sample.id));
  $('file-name').textContent = sample ? 'JPEG or PNG, up to 10 MB' : (file ? file.name : 'JPEG or PNG, up to 10 MB');
  $('predict').disabled = !file || busy;
  $('prediction').hidden = true;
  $('empty').hidden = false;
  setProgress('');
}

// A file the browser cannot display is usually not a valid image; hide the broken icon.
$('input-preview').onerror = () => {
  if (!$('input-preview').getAttribute('src')) return;
  $('input-preview').hidden = true;
  setProgress('This file cannot be previewed. It may not be a valid JPEG or PNG.', true);
};

$('image-file').onchange = () => {
  useImage($('image-file').files[0] || null, null);
};

async function loadSamples() {
  try {
    const { samples } = await (await fetch('/api/samples')).json();
    for (const sample of samples) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.id = sample.id;
      button.setAttribute('aria-pressed', 'false');
      button.title = `Example ${sample.id} (${sample.image})`;
      button.innerHTML = `<img src="${sample.url}" alt="Example ${sample.id}">`;
      button.onclick = async () => {
        const blob = await (await fetch(sample.url)).blob();
        $('image-file').value = '';
        useImage(new File([blob], `${sample.image}.jpg`, { type: 'image/jpeg' }), sample);
      };
      $('samples').append(button);
    }
  } catch {
    $('samples').textContent = 'Examples are unavailable.';
  }
}

/* ---------- Running the model ---------- */
$('predict').onclick = async () => {
  if (!current || busy) return;
  const { file, sample } = current;
  if (file.size > 10 * 1024 * 1024) return setProgress('Please choose an image no larger than 10 MB.', true);
  busy = true;
  $('predict').disabled = true;
  $('image-file').disabled = true;
  $('prediction').hidden = true;
  $('empty').hidden = false;
  setProgress('Checking image…');
  try {
    // Upload raw bytes; the server does not save the image to disk.
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: file
    });
    const result = await response.json().catch(() => {
      throw new Error('The server returned an invalid response. Please try again.');
    });
    if (!response.ok) throw new Error(result.error || 'Unable to process this image.');
    showResult(result, sample);
    setProgress('');
  } catch (error) {
    setProgress(`Error: ${error instanceof TypeError ? 'Cannot reach the server. Make sure the demo is running and try again.' : error.message}`, true);
  } finally {
    busy = false;
    $('predict').disabled = !current;
    $('image-file').disabled = false;
    refreshStatus();
  }
};

function showResult(result, sample) {
  const mel = result.positive;
  $('decision').textContent = mel ? 'Looks like melanoma' : 'Does not look like melanoma';
  $('decision').className = 'decision ' + (mel ? 'mel' : 'non');
  $('score').textContent = result.score.toFixed(4);
  $('threshold').textContent = result.threshold.toFixed(4);
  $('meter-fill').style.width = `${(result.score * 100).toFixed(1)}%`;
  $('meter-fill').classList.toggle('mel', mel);
  $('meter-threshold').style.left = `calc(${(result.threshold * 100).toFixed(1)}% - 1px)`;
  $('truth-row').hidden = !sample;
  if (sample) {
    const agrees = (sample.label === 'Melanoma') === mel;
    const truth = sample.label === 'Melanoma' ? 'Melanoma' : `Not melanoma (${DX_NAMES[sample.dx] || sample.dx})`;
    const verdict = agrees ? 'AI was right' : (mel ? 'AI was wrong: false alarm' : 'AI was wrong: missed melanoma');
    $('truth').textContent = `${truth} — ${verdict}`;
    $('truth').className = agrees ? 'right' : 'wrong';
  }
  $('model-input').src = result.preview;
  $('overlay').hidden = !result.cam_available;
  $('cam-empty').hidden = result.cam_available;
  $('legend').hidden = !result.cam_available;
  if (result.overlay) $('overlay').src = result.overlay;
  else $('overlay').removeAttribute('src');
  $('empty').hidden = true;
  $('prediction').hidden = false;
  // On narrow screens the output sits below the input; bring it into view.
  if (window.matchMedia('(max-width: 700px)').matches) $('output-heading').scrollIntoView({ behavior: 'smooth' });
}

/* ---------- Research results tab ---------- */
const num = (v) => (v === '' || v == null || Number.isNaN(Number(v)) ? null : Number(v));

function fmt(value, units) {
  const x = num(value);
  if (x === null) return value || '—';
  if (units === 'pp' || units === '%') return x.toFixed(1);
  return Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(3);
}

function fmtEstimate(row) {
  const est = fmt(row.estimate, row.units);
  const unit = row.units === 'pp' ? ' pp' : row.units === '%' ? '%' : '';
  const ci = num(row.ci95_lo) !== null && num(row.ci95_hi) !== null
    ? ` [${fmt(row.ci95_lo, row.units)}, ${fmt(row.ci95_hi, row.units)}]` : '';
  return est + unit + ci;
}

function findRow(match) {
  return resultRows.find((r) => Object.entries(match).every(([k, v]) => r[k] === v));
}

function signed(text) {
  return text.startsWith('-') ? '−' + text.slice(1) : '+' + text;
}

function renderHighlights() {
  const cards = [];
  const gap = findRow({ endpoint: 'E1.2 sensitivity difference', cohort: 'Derm7pt PRIMARY', role: 'primary estimate' });
  const derm = findRow({ endpoint: 'E1.2 sensitivity (dermoscopy)', cohort: 'Derm7pt PRIMARY' });
  const clin = findRow({ endpoint: 'E1.2 sensitivity (clinical)', cohort: 'Derm7pt PRIMARY' });
  const s1 = findRow({ endpoint: 'E1.1/E1.2 seed spread', seed: '1' });
  const s2 = findRow({ endpoint: 'E1.1/E1.2 seed spread', seed: '2' });
  if (gap && derm && clin) cards.push({
    tag: 'RQ1 · sensitivity gap', value: signed(fmt(gap.estimate, 'pp')) + ' pp',
    ci: `[${fmt(gap.ci95_lo, 'pp')}, ${fmt(gap.ci95_hi, 'pp')}]`,
    desc: `Melanoma found in ${fmt(derm.estimate, '%')}% of dermoscopy images vs ${fmt(clin.estimate, '%')}% of clinical images of the same 249 lesions.` +
      (s1 && s2 ? ` Other seeds: ${signed(fmt(s1.estimate, 'pp'))} and ${signed(fmt(s2.estimate, 'pp'))} pp.` : '')
  });
  const auc = findRow({ endpoint: 'E1.3 paired AUROC difference', cohort: 'Derm7pt PRIMARY' });
  const aucD = findRow({ endpoint: 'E1.3 AUROC (dermoscopy)', cohort: 'Derm7pt PRIMARY' });
  const aucC = findRow({ endpoint: 'E1.3 AUROC (clinical)', cohort: 'Derm7pt PRIMARY' });
  if (auc && aucD && aucC) cards.push({
    tag: 'RQ1 · ranking (AUROC)', value: signed(fmt(auc.estimate)),
    ci: `[${fmt(auc.ci95_lo)}, ${fmt(auc.ci95_hi)}]`,
    desc: `AUROC ${fmt(aucD.estimate)} on dermoscopy vs ${fmt(aucC.estimate)} on clinical images (1,002 lesion pairs).`
  });
  const xai = findRow({ endpoint: 'E4.6 Delta AUROC M1-M0', cohort: 'primary' });
  if (xai) cards.push({
    tag: 'RQ4 · XAI for error prediction', value: signed(fmt(xai.estimate)),
    ci: `[${fmt(xai.ci95_lo)}, ${fmt(xai.ci95_hi)}]`,
    desc: 'Adding explanation features to a probability-only error predictor: no improvement was demonstrated.'
  });
  const rq5 = findRow({ endpoint: 'adapted - base auroc (PAD eval)' });
  if (rq5) cards.push({
    tag: 'RQ5 · smartphone fine-tuning', value: signed(fmt(rq5.estimate)),
    ci: `[${fmt(rq5.ci95_lo)}, ${fmt(rq5.ci95_hi)}]`,
    desc: 'AUROC change after fine-tuning on PAD-UFES-20 smartphone images. One recipe, few validation melanomas.'
  });
  $('highlights').innerHTML = '';
  for (const c of cards) {
    const div = document.createElement('div');
    div.className = 'stat';
    div.innerHTML = '<p class="tag"></p><p class="value"></p><p class="ci"></p><p class="desc"></p>';
    div.querySelector('.tag').textContent = c.tag;
    div.querySelector('.value').textContent = c.value;
    div.querySelector('.ci').textContent = '95% CI ' + c.ci;
    div.querySelector('.desc').textContent = c.desc;
    $('highlights').append(div);
  }
}

function renderTable() {
  const query = $('search').value.trim().toLowerCase();
  const rows = resultRows.filter((r) =>
    (activeRq === 'All' || r.rq === activeRq) &&
    (!query || [r.endpoint, r.model, r.cohort, r.estimand, r.role].join(' ').toLowerCase().includes(query)));
  const body = $('results-body');
  body.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    const cells = [
      [r.rq], [r.endpoint, r.estimand], [r.model, `seed ${r.seed}`], [r.cohort],
      [r.n, r.clusters && r.clusters !== r.n ? `${r.clusters} clusters` : ''], [fmtEstimate(r), r.units && r.units !== 'pp' && r.units !== '%' ? r.units : ''], [r.role]
    ];
    cells.forEach(([main, sub], i) => {
      const td = document.createElement('td');
      if (i === 4 || i === 5) td.className = 'num';
      td.textContent = main;
      if (sub) { const s = document.createElement('small'); s.textContent = sub; td.append(s); }
      tr.append(td);
    });
    body.append(tr);
  }
  $('results-count').textContent = `${rows.length} of ${resultRows.length} rows`;
}

async function loadResults() {
  $('results-count').textContent = 'Loading results…';
  try {
    const data = await (await fetch('/api/results')).json();
    resultRows = data.rows;
    $('results-source').textContent = `Source: ${data.source} (SHA-256 ${data.sha256.slice(0, 12)}…). Read directly; nothing is recomputed.`;
    const rqs = ['All', ...new Set(resultRows.map((r) => r.rq))];
    $('rq-filter').innerHTML = '';
    for (const rq of rqs) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = rq;
      b.setAttribute('aria-pressed', rq === activeRq);
      b.onclick = () => {
        activeRq = rq;
        document.querySelectorAll('#rq-filter button').forEach((x) => x.setAttribute('aria-pressed', x === b));
        renderTable();
      };
      $('rq-filter').append(b);
    }
    renderHighlights();
    renderTable();
  } catch {
    $('results-count').textContent = 'Results could not be loaded. Make sure the demo server is running.';
  }
}
$('search').oninput = renderTable;

refreshStatus();
loadSamples();
