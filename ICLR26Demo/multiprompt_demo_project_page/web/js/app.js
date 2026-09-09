/* VGGT-Omega promptable segmentation demo -- frontend controller. */
(() => {
'use strict';

const $ = (id) => document.getElementById(id);
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 300) || r.statusText);
  return r;
};
const json = async (url, body) => (await api(url, body ? {
  method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
} : {})).json();

/* Matches object_colors_bgr() on the backend so chip colours line up with
   the rendered masks. */
const objectColor = (i, n) => `hsl(${Math.round(i * 180 / Math.max(n, 1)) * 2}, 78%, 55%)`;

const state = {
  sourceId: null,
  sid: null,
  session: null,
  refView: 0,
  objects: [],          // [{points: [{x,y,label}]}]
  activeObj: 0,
  polarity: 1,
  promptMode: 'points',
  sceneMode: 'dynamic',
  maskUrl: null,
  showMask: true,
  poll: null,
};

/* ── source selection ─────────────────────────────────────────── */
async function loadGallery() {
  const {items} = await json('/api/gallery');
  const box = $('gallery');
  const render = (filter = '') => {
    box.innerHTML = '';
    const groups = new Map();
    for (const it of items) {
      const hay = `${it.name} ${it.group}`.toLowerCase();
      if (filter && !hay.includes(filter)) continue;
      if (!groups.has(it.group)) groups.set(it.group, []);
      groups.get(it.group).push(it);
    }
    for (const [group, list] of groups) {
      const label = document.createElement('div');
      label.className = 'gitem group-label';
      label.textContent = group;
      box.appendChild(label);
      for (const it of list) {
        const b = document.createElement('button');
        b.className = 'gitem';
        b.dataset.id = it.id;
        b.innerHTML = `<b></b><span>${it.kind}</span>`;
        b.querySelector('b').textContent = it.name;
        b.onclick = () => selectSource(it, b);
        box.appendChild(b);
      }
    }
    if (!box.children.length) box.innerHTML = '<div class="gitem group-label">no matches</div>';
  };
  render();
  $('gallery-search').oninput = (e) => render(e.target.value.trim().toLowerCase());
}

function selectSource(item, el) {
  state.sourceId = item.id;
  document.querySelectorAll('.gitem').forEach((n) => n.classList.remove('selected'));
  if (el) el.classList.add('selected');
  if (item.suggested_mode) setSceneMode(item.suggested_mode);
  startSession();
}

async function startSession() {
  if (!state.sourceId) return;
  resetPrompts();
  $('results-card').hidden = true;
  const s = await json('/api/session', {
    source: state.sourceId,
    scene_mode: state.sceneMode,
    n_frames: parseInt($('n-frames').value, 10),
  });
  state.sid = s.sid;
  state.session = s;
  setState(s.state);
  beginPolling();
}

/* ── polling ──────────────────────────────────────────────────── */
function beginPolling() {
  clearInterval(state.poll);
  state.poll = setInterval(refresh, 900);
  refresh();
}

async function refresh() {
  if (!state.sid) return;
  let s;
  try { s = await json(`/api/session/${state.sid}`); } catch { return; }
  const prevState = state.session ? state.session.state : null;
  state.session = s;
  setState(s.state);
  $('pill-frames').textContent = s.n_frames ? `${s.n_frames} frames` : 'frames —';
  $('pill-geometry').textContent = s.geometry ? `${s.geometry} geometry` : 'geometry —';
  $('log').textContent = (s.log || []).join('\n');
  $('log').scrollTop = $('log').scrollHeight;
  $('progress-line').textContent = s.progress || '';

  if (s.state === 'ready' && prevState !== 'ready') onSessionReady(s);
  if (s.state === 'done' && prevState !== 'done') onRunDone(s);
  const busy = s.state === 'preparing' || s.state === 'running';
  $('run-btn').disabled = busy || s.state === 'new' || !hasPrompt();
  $('preview-btn').disabled = busy || s.state === 'new' || !hasPrompt();
  if (!busy && s.state !== 'new') clearInterval(state.poll);
}

function setState(st) {
  const pill = $('pill-state');
  pill.textContent = st;
  pill.dataset.state = st;
}

/* ── reference frame + filmstrip ──────────────────────────────── */
function onSessionReady(s) {
  $('canvas-placeholder').hidden = true;
  $('frame-img').hidden = false;
  $('point-canvas').hidden = false;
  showFrame(0);
  buildFilmstrip(s.n_frames);
}

function showFrame(i) {
  state.refView = i;
  const img = $('frame-img');
  img.onload = () => { sizeCanvas(); drawPoints(); };
  img.src = `/api/session/${state.sid}/frame/${i}?w=960&t=${Date.now()}`;
  document.querySelectorAll('.filmstrip img').forEach((n, k) =>
    n.classList.toggle('active', k === i));
  clearMaskPreview();
}

function buildFilmstrip(n) {
  const strip = $('filmstrip');
  strip.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const t = document.createElement('img');
    t.src = `/api/session/${state.sid}/frame/${i}?w=120`;
    t.title = `frame ${i}`;
    t.className = i === 0 ? 'active' : '';
    t.onclick = () => { resetPrompts(); showFrame(i); };
    strip.appendChild(t);
  }
}

function sizeCanvas() {
  const img = $('frame-img'), cv = $('point-canvas'), mk = $('mask-img');
  const r = img.getBoundingClientRect();
  cv.width = r.width; cv.height = r.height;
  cv.style.width = `${r.width}px`; cv.style.height = `${r.height}px`;
  mk.style.width = `${r.width}px`; mk.style.height = `${r.height}px`;
}

/* ── point prompting ──────────────────────────────────────────── */
function ensureObject() {
  if (!state.objects.length) { state.objects.push({points: []}); state.activeObj = 0; }
}

function canvasToImage(ev) {
  const r = $('frame-img').getBoundingClientRect();
  // Normalised, because the preview is served at a display resolution that has
  // nothing to do with the working resolution the backend indexes.
  return {
    x: (ev.clientX - r.left) / r.width,
    y: (ev.clientY - r.top) / r.height,
  };
}

$('point-canvas').addEventListener('click', (ev) => {
  if (state.promptMode !== 'points' || !state.session || state.session.state !== 'ready') return;
  ensureObject();
  const p = canvasToImage(ev);
  const label = ev.shiftKey ? 0 : state.polarity;
  state.objects[state.activeObj].points.push({x: p.x, y: p.y, label});
  renderObjects(); drawPoints(); refreshButtons();
});

function drawPoints() {
  const cv = $('point-canvas');
  if (!cv.width) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  state.objects.forEach((obj, oi) => {
    const col = objectColor(oi, state.objects.length);
    obj.points.forEach((p) => {
      const x = p.x * cv.width, y = p.y * cv.height;
      ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fillStyle = p.label ? col : '#0b0d12';
      ctx.strokeStyle = p.label ? '#fff' : col;
      ctx.lineWidth = 2.5; ctx.fill(); ctx.stroke();
      if (!p.label) {                    // minus sign for exclusions
        ctx.beginPath(); ctx.moveTo(x - 3.5, y); ctx.lineTo(x + 3.5, y);
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
      }
      if (oi !== state.activeObj) { ctx.globalAlpha = 1; }
    });
  });
}

function renderObjects() {
  const box = $('objects');
  box.innerHTML = '';
  state.objects.forEach((obj, i) => {
    const el = document.createElement('div');
    el.className = 'obj' + (i === state.activeObj ? ' active' : '');
    el.innerHTML = `<span class="dot"></span><span class="name"></span>
                    <span class="count"></span><button class="del" title="remove">&times;</button>`;
    el.querySelector('.dot').style.background = objectColor(i, state.objects.length);
    el.querySelector('.name').textContent = `Object ${i + 1}`;
    el.querySelector('.count').textContent =
      `${obj.points.length} pt${obj.points.length === 1 ? '' : 's'}`;
    el.onclick = (e) => {
      if (e.target.classList.contains('del')) {
        state.objects.splice(i, 1);
        state.activeObj = Math.max(0, Math.min(state.activeObj, state.objects.length - 1));
      } else {
        state.activeObj = i;
      }
      renderObjects(); drawPoints(); refreshButtons();
    };
    box.appendChild(el);
  });
}

function resetPrompts() {
  state.objects = []; state.activeObj = 0;
  renderObjects(); clearMaskPreview(); refreshButtons();
  const cv = $('point-canvas');
  if (cv.width) cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
}

function hasPrompt() {
  return state.promptMode === 'dense'
    || state.objects.some((o) => o.points.length > 0);
}

function refreshButtons() {
  const ready = state.session && state.session.state === 'ready';
  $('preview-btn').disabled = !ready || !hasPrompt();
  $('run-btn').disabled = !ready || !hasPrompt();
}

/* ── mask preview ─────────────────────────────────────────────── */
function promptPayload() {
  return {
    mode: state.promptMode,
    objects: state.objects,
    ref_view: state.refView,
    top_k: parseInt($('top-k').value, 10),
    points_per_side: parseInt($('grid-density').value, 10),
  };
}

function clearMaskPreview() {
  if (state.maskUrl) URL.revokeObjectURL(state.maskUrl);
  state.maskUrl = null;
  $('mask-img').hidden = true;
  $('toggle-preview').hidden = true;
}

$('preview-btn').onclick = async () => {
  const btn = $('preview-btn');
  btn.disabled = true; btn.textContent = 'Running SAM…';
  try {
    const r = await api(`/api/session/${state.sid}/prompt`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(promptPayload()),
    });
    const n = parseInt(r.headers.get('X-Object-Count') || '0', 10);
    clearMaskPreview();
    state.maskUrl = URL.createObjectURL(await r.blob());
    const mk = $('mask-img');
    mk.src = state.maskUrl; mk.hidden = false;
    state.showMask = true;
    $('toggle-preview').hidden = false;
    $('toggle-preview').textContent = 'Hide masks';
    sizeCanvas();
    if (state.promptMode === 'dense') {
      // Dense mode discovers its own objects; mirror them as chips so the
      // colours in the preview have a legend.
      state.objects = Array.from({length: n}, () => ({points: []}));
      renderObjects();
    }
  } catch (e) {
    alert(`Prompt failed: ${e.message}`);
  } finally {
    btn.textContent = 'Preview masks'; refreshButtons();
  }
};

$('toggle-preview').onclick = () => {
  state.showMask = !state.showMask;
  $('mask-img').hidden = !state.showMask;
  $('toggle-preview').textContent = state.showMask ? 'Hide masks' : 'Show masks';
};

$('undo-point').onclick = () => {
  const obj = state.objects[state.activeObj];
  if (obj && obj.points.length) { obj.points.pop(); renderObjects(); drawPoints(); refreshButtons(); }
};
$('clear-points').onclick = resetPrompts;
$('add-object').onclick = () => {
  state.objects.push({points: []});
  state.activeObj = state.objects.length - 1;
  renderObjects(); refreshButtons();
};

/* ── run ──────────────────────────────────────────────────────── */
$('run-btn').onclick = async () => {
  $('run-btn').disabled = true;
  await json(`/api/session/${state.sid}/run`, {
    ...promptPayload(),
    render: $('opt-render').checked,
    exclusive: $('opt-exclusive').checked,
    geo_cc: $('opt-geocc').checked,
  });
  $('console-card').scrollIntoView({behavior: 'smooth', block: 'nearest'});
  beginPolling();
};

function onRunDone(s) {
  const m = s.manifest || {};
  const file = (n) => `/api/session/${state.sid}/file/${n}`;
  $('results-card').hidden = false;

  if (m.pip_video) { $('pip-video').src = file(m.pip_video); selectResultTab('pip'); }
  else selectResultTab('overlay');
  if (m.overlay_video) $('overlay-video').src = file(m.overlay_video);

  const meta = $('result-meta');
  meta.innerHTML = '';
  const chips = [
    `${m.n_objects} object${m.n_objects === 1 ? '' : 's'}`,
    `${m.n_frames} frames`,
    `${m.scene_mode} mode`,
    `${m.geometry} geometry`,
    `${m.elapsed_s}s`,
  ];
  if (m.render_error) chips.push(`render failed: ${m.render_error}`);
  for (const c of chips) {
    const sp = document.createElement('span'); sp.textContent = c; meta.appendChild(sp);
  }
  for (const [label, key] of [['masks .npz', 'masks_npz'], ['cloud .npz', 'cloud_npz'],
                              ['follow .mp4', 'follow_video'], ['BEV .mp4', 'birdeye_video']]) {
    if (!m[key]) continue;
    const sp = document.createElement('span');
    const a = document.createElement('a');
    a.href = file(m[key]); a.download = m[key]; a.textContent = label;
    sp.appendChild(a); meta.appendChild(sp);
  }
  $('results-card').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

/* ── viser ────────────────────────────────────────────────────── */
$('viser-start').onclick = async () => {
  const btn = $('viser-start');
  btn.disabled = true; btn.textContent = 'Starting…';
  try {
    const {port} = await json(`/api/session/${state.sid}/viser`, {});
    const f = $('viser-frame');
    f.src = `${location.protocol}//${location.hostname}:${port}`;
    f.hidden = false;
    $('viser-hint').textContent = `Running on port ${port}.`;
    btn.textContent = 'Restart viewer';
  } catch (e) {
    $('viser-hint').textContent = `Failed: ${e.message}`;
    btn.textContent = 'Launch viewer';
  } finally {
    btn.disabled = false;
  }
};

/* ── tabs, toggles, misc ──────────────────────────────────────── */
function wireTabs(rootId, onPick) {
  const root = $(rootId);
  root.querySelectorAll('button').forEach((b) => {
    b.onclick = () => {
      root.querySelectorAll('button').forEach((n) => n.classList.remove('active'));
      b.classList.add('active');
      onPick(b.dataset.tab || b.dataset.label);
    };
  });
}

wireTabs('source-tabs', (t) => {
  $('tab-gallery').hidden = t !== 'gallery';
  $('tab-upload').hidden = t !== 'upload';
});
wireTabs('prompt-tabs', (t) => {
  state.promptMode = t;
  $('tab-points').hidden = t !== 'points';
  $('tab-dense').hidden = t !== 'dense';
  clearMaskPreview(); refreshButtons();
});
wireTabs('point-polarity', (l) => { state.polarity = parseInt(l, 10); });

function selectResultTab(tab) {
  const root = $('result-tabs');
  root.querySelectorAll('button').forEach((b) =>
    b.classList.toggle('active', b.dataset.tab === tab));
  $('res-pip').hidden = tab !== 'pip';
  $('res-overlay').hidden = tab !== 'overlay';
  $('res-viser').hidden = tab !== 'viser';
}
wireTabs('result-tabs', selectResultTab);

function setSceneMode(mode) {
  state.sceneMode = mode;
  document.querySelectorAll('#scene-mode .choice-card').forEach((card) => {
    const on = card.querySelector('input').value === mode;
    card.classList.toggle('selected', on);
    card.querySelector('input').checked = on;
  });
  // The 3D-connectivity filter only makes sense on a rigid scene.
  const wrap = $('opt-geocc-wrap');
  const isStatic = mode === 'static';
  $('opt-geocc').disabled = !isStatic;
  $('opt-geocc').checked = isStatic;
  wrap.classList.toggle('disabled', !isStatic);
}

document.querySelectorAll('#scene-mode input').forEach((r) => {
  r.onchange = () => { setSceneMode(r.value); if (state.sourceId) startSession(); };
});

$('n-frames').oninput = (e) => { $('n-frames-val').textContent = e.target.value; };
$('n-frames').onchange = () => { if (state.sourceId) startSession(); };
$('grid-density').oninput = (e) => {
  document.querySelectorAll('#grid-density-val, #grid-density-val + b')
    .forEach((n) => { n.textContent = e.target.value; });
};
$('top-k').oninput = (e) => { $('top-k-val').textContent = e.target.value; };

/* upload */
const dz = $('dropzone'), fi = $('file-input');
dz.onclick = () => fi.click();
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('hot'); };
dz.ondragleave = () => dz.classList.remove('hot');
dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove('hot'); upload(e.dataTransfer.files); };
fi.onchange = () => upload(fi.files);

async function upload(fileList) {
  const files = [...(fileList || [])];
  if (!files.length) return;
  const label = files.length === 1 ? files[0].name : `${files.length} images`;
  dz.querySelector('strong').textContent = `Uploading ${label}…`;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  try {
    const item = await (await api('/api/upload', {method: 'POST', body: fd})).json();
    dz.querySelector('strong').textContent = label;
    selectSource({...item, suggested_mode: null}, null);
  } catch (e) {
    dz.querySelector('strong').textContent = `Upload failed: ${e.message}`;
  }
}

window.addEventListener('resize', () => { if (!$('frame-img').hidden) { sizeCanvas(); drawPoints(); } });

setSceneMode('dynamic');
renderObjects();
loadGallery().catch((e) => { $('gallery').textContent = `gallery failed: ${e.message}`; });
})();
