const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const state = {
  clips: [],
  presets: [],
  clip: null,
  job: null,
  view: 'compare',
  mode: 'side',
  stream: 'output',
  frame: 0,
  frames: 0,
  playing: false,
  rate: 1,
  timer: null,
  diag: null,
  wipe: 0.5,
  motionKind: 'motion',
};

const fmt = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '—' :
  (typeof v === 'number' ? (Math.abs(v) > 1000 ? v.toFixed(0) : v.toFixed(n)) : v);

function badge(el, text, kind) {
  el.textContent = text;
  el.className = 'badge badge-' + (kind || 'idle');
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const text = await r.text();
  let j = {};
  try {
    j = text ? JSON.parse(text) : {};
  } catch (e) {
    throw new Error('服务端返回了无法解析的数据 (HTTP ' + r.status + ')');
  }
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

/* ---------------------------------------------------------------- library */
async function loadLibrary() {
  const { clips, presets } = await api('/api/library');
  state.clips = clips;
  state.presets = presets;
  const list = $('#clip-list');
  list.innerHTML = '';
  if (!clips.length) {
    list.innerHTML = '<div class="empty">media/clips 下没有片段。先运行 tools/make_clips.py。</div>';
  }
  clips.forEach((c, i) => {
    const el = document.createElement('div');
    el.className = 'clip' + (i === 0 ? ' sel' : '');
    const warn = [];
    if ((c.dropped_frames || []).length) warn.push('缺帧');
    if ((c.scene_cuts || []).length) warn.push('切镜');
    el.innerHTML = `<div class="t"><strong>${c.name}</strong></div>
      <div class="n">${c.width}×${c.height} · ${c.frames}f
      ${warn.map(w => `<span class="chip warn">${w}</span>`).join('')}</div>`;
    el.onclick = () => {
      $$('.clip').forEach(x => x.classList.remove('sel'));
      el.classList.add('sel');
      selectClip(c);
    };
    list.appendChild(el);
  });
  const prow = $('#presets');
  prow.innerHTML = '';
  presets.forEach(p => {
    const b = document.createElement('button');
    b.textContent = p.name;
    b.onclick = () => applyParams(p.params);
    prow.appendChild(b);
  });
  if (clips.length) selectClip(clips[0]);
}

function selectClip(c) {
  state.clip = c;
  $('#clip-meta').textContent = `${c.name} · ${c.width}×${c.height} · ${c.fps.toFixed(2)} fps`;
  const n = c.noise || {};
  $('#noise-box').innerHTML = `
    <div class="row"><span>模型</span><b>${n.model || '—'}</b></div>
    <div class="row"><span>σ (R,G,B)</span><b>${(n.sigma_rgb || []).map(x => x.toFixed(3)).join(', ')}</b></div>
    <div class="row"><span>来源</span><b class="origin">${n.origin || '—'}</b></div>
    <div style="margin-top:6px;color:var(--muted);font-size:10px;line-height:1.5">${n.detail || ''}</div>`;
}

/* ---------------------------------------------------------------- params */
function applyParams(p) {
  for (const [k, v] of Object.entries(p)) {
    const el = $('#' + ({ temporal_radius: 'radius', temporal_strength: 'temporal',
      spatial_strength: 'spatial', max_residual: 'maxres', min_confidence: 'minconf',
      motion_model: 'motion', sigma_scale: 'sigscale' })[k]);
    if (!el) continue;
    el.value = v;
    el.dispatchEvent(new Event('input'));
  }
}
function collectParams() {
  return {
    temporal_radius: +$('#radius').value,
    temporal_strength: +$('#temporal').value,
    spatial_strength: +$('#spatial').value,
    max_residual: +$('#maxres').value,
    min_confidence: +$('#minconf').value,
    motion_model: $('#motion').value,
    sigma_scale: +$('#sigscale').value,
  };
}
['radius', ['temporal'], ['spatial'], ['maxres'], ['minconf'], ['sigscale']].flat()
  .forEach(id => {
    const key = Array.isArray(id) ? id[0] : id;
    const el = $('#' + key);
    const out = $('#' + key + '-val');
    const sync = () => { if (out) out.textContent = el.value; };
    el.addEventListener('input', sync); sync();
  });

/* ---------------------------------------------------------------- job */
async function startJob() {
  if (!state.clip) return;
  $('#run').disabled = true;
  $('#cancel').disabled = false;
  $('#progress-text').textContent = '排队 · 0/0 帧';
  $('#progress-bar').style.width = '0%';
  badge($('#engine-badge'), '内核运行中', 'run');
  try {
    const r = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        clip_id: state.clip.id,
        params: collectParams(),
        auto_cuts: $('#autocuts').checked,
        scene_cuts: state.clip.scene_cuts || [],
      }),
    });
    state.job = { id: r.job_id };
    poll();
  } catch (e) {
    finishWithError(e.message);
  }
}

let pollSeq = 0;
async function poll() {
  if (!state.job) return;
  const seq = ++pollSeq;
  try {
    const j = await api('/api/jobs/' + state.job.id);
    // Ignore a stale response that arrived after a newer one (and never
    // regress the status text to an older snapshot).
    if (seq !== pollSeq) return;
    if (!j || j.status === undefined) {
      // A malformed/empty response must not silently end the job: keep polling
      // and show that the reply was unusable.
      badge($('#engine-badge'), '响应异常，重试中', 'run');
      setTimeout(poll, 1200);
      return;
    }
    if (state.job && state.job.status === 'done' && j.status !== 'done') return;
    state.job = Object.assign({}, state.job, j);
    const p = j.progress || { done: 0, total: 0 };
    $('#progress-bar').style.width = (p.total ? (100 * p.done / p.total) : (j.status === 'done' ? 100 : 0)) + '%';
    const stages = {
      queued: '排队', loading: '读取素材', denoising: '时序融合',
      rendering: '编码输出', done: '完成', error: '失败', cancelled: '已取消',
    };
    $('#progress-text').textContent = `${stages[j.status] || j.status} · ${p.done}/${p.total} 帧`;
    if (j.status === 'error') { finishWithError(j.error || '未知错误'); return; }
    if (j.status === 'cancelled') {
      $('#run').disabled = false; $('#cancel').disabled = true;
      badge($('#engine-badge'), '已取消', 'bad'); return;
    }
    if (j.status === 'done') { await onDone(j); return; }
    setTimeout(poll, 450);
  } catch (e) {
    // Surface the real failure rather than looping forever with a stale panel.
    console.error('poll failed', e);
    badge($('#engine-badge'), '轮询失败: ' + e.message, 'bad');
    setTimeout(poll, 1200);
  }
}

function finishWithError(msg) {
  $('#run').disabled = false; $('#cancel').disabled = true;
  badge($('#engine-badge'), '失败: ' + msg, 'bad');
  $('#progress-text').textContent = '失败：' + msg;
  state.job = null;
}

async function onDone(j) {
  $('#run').disabled = false; $('#cancel').disabled = true;
  badge($('#engine-badge'), '完成', 'ok');
  $('#progress-text').textContent = `完成 · ${j.summary.frames_out} 帧 · ${j.summary.elapsed_s}s`;
  state.frames = (j.preview_frames || []).length;
  state.frame = 0;
  $('#scrub').max = Math.max(0, state.frames - 1);
  $('#scrub').value = 0;
  // Keep the merged job object (it carries params/previews/downloads used by
  // the viewer) and add the timeline fallback.
  j.timeline = j.timeline || (j.summary && j.summary.timeline) || [];
  state.job = Object.assign({}, state.job, j);
  state.diag = j.diags || null;
  try {
    renderStamps(j);
    renderMetrics(j);
    renderDiags(j);
    renderDownloads(j);
    await showFrame(0);
  } catch (e) {
    console.error('render failed', e);
    $('#progress-text').textContent = '渲染界面失败: ' + e.message;
  }
  // Preload neighbours around the current frame.
  for (let k = 1; k < Math.min(state.frames, 6); k++) preload(k);
}

/* ---------------------------------------------------------------- viewer */
const imgCache = new Map();
function frameUrl(kind, k) { return `/api/jobs/${state.job.id}/frames/${k}?kind=${kind}`; }

function preload(k) {
  if (!state.job || !state.job.preview_frames) return;
  if (k < 0 || k >= state.frames) return;
  ['source', 'output'].forEach(kind => {
    const u = frameUrl(kind, k);
    if (!imgCache.has(u)) { const im = new Image(); im.src = u; imgCache.set(u, im); }
  });
}

async function showFrame(k) {
  if (!state.job || !state.job.preview_frames) return;
  const n = state.frames;
  k = Math.max(0, Math.min(n - 1, k));
  state.frame = k;
  $('#scrub').value = k;
  const tl = state.job.timeline || [];
  const ts = tl[k];
  $('#frame-label').textContent = ts ? `#${ts.index} · t=${ts.t.toFixed(3)}s` : '--';

  const a = $('#img-a'), b = $('#img-b');
  const stream = state.stream; // output | input | truth
  if (state.mode === 'diff') {
    a.src = await diffDataUrl(k);
    $('#tag-a').textContent = '差异 ×6（输出 vs 源）；亮色=差异大';
    b.src = a.src;
  } else {
    a.src = frameUrl(stream === 'truth' ? 'truth' : 'source', k);
    $('#tag-a').textContent = stream === 'truth' ? '清晰真值' : '原始输入';
    b.src = frameUrl(stream === 'truth' ? 'truth' : 'output', k);
    $('#tag-b').textContent = stream === 'truth' ? '清晰真值' : '降噪输出';
  }
  if (state.view === 'motion') refreshMotion();
  highlightStamp(k);
  preload(k - 1); preload(k + 1);
}

async function diffDataUrl(k) {
  const [sa, sb] = await Promise.all([
    loadImage(frameUrl('source', k)), loadImage(frameUrl('output', k)),
  ]);
  const c = document.createElement('canvas');
  c.width = sa.naturalWidth; c.height = sa.naturalHeight;
  const x = c.getContext('2d');
  x.drawImage(sa, 0, 0); const A = x.getImageData(0, 0, c.width, c.height);
  x.clearRect(0, 0, c.width, c.height); x.drawImage(sb, 0, 0);
  const B = x.getImageData(0, 0, c.width, c.height);
  const D = x.createImageData(c.width, c.height);
  for (let i = 0; i < A.data.length; i += 4) {
    const d = 6 * Math.abs(A.data[i] - B.data[i]);
    D.data[i] = Math.min(255, d * 2); D.data[i + 1] = Math.min(255, 60 + d);
    D.data[i + 2] = Math.min(255, 100 + d); D.data[i + 3] = 255;
  }
  x.putImageData(D, 0, 0);
  return c.toDataURL('image/png');
}
function loadImage(url) {
  return new Promise((res, rej) => {
    const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url;
  });
}

function highlightStamp(k) {
  $$('.stamp').forEach((s, i) => s.classList.toggle('cur', i === k));
}

function renderStamps(j) {
  const box = $('#stamps'); box.innerHTML = '';
  const tl = j.timeline || (j.summary && j.summary.timeline) || [];
  const cuts = new Set(j.summary.detected_cuts || []);
  const fb = new Set((j.diags || []).filter(d => d.fallback !== 'none').map(d => d.frame_index));
  const srcIdx = tl.map(t => t.index);
  tl.forEach((t, k) => {
    const s = document.createElement('div');
    s.className = 'stamp';
    if (cuts.has(t.index)) s.classList.add('cut');
    else if (fb.has(t.index)) s.classList.add('fallback');
    s.title = `#${t.index} t=${t.t.toFixed(3)}`;
    s.onclick = () => showFrame(k);
    box.appendChild(s);
  });
  // Mark source frames that are missing between two timeline entries as empty
  // slots, preserving their expected cadence (they are not silently filled).
  for (let k = 1; k < srcIdx.length; k++) {
    const gap = srcIdx[k] - srcIdx[k - 1];
    if (gap > 1) {
      for (let m = 1; m < gap; m++) {
        const s = document.createElement('div');
        s.className = 'stamp missing';
        s.title = `缺帧槽位 #${srcIdx[k - 1] + m}（输出保留空档）`;
        box.insertBefore(s, box.children[k + m - 1] || null);
      }
    }
  }
}

/* ---------------------------------------------------------------- motion */
async function refreshMotion() {
  if (!state.job || !state.job.preview_frames) return;
  const k = state.frame;
  const d = (state.diag || [])[k];
  const tl0 = state.job.timeline || [];
  const tm = tl0[k];
  $('#m-frame').textContent = tm ? `帧 #${tm.index} · t=${tm.t.toFixed(3)}s` : '帧 --';
  const fb = d ? d.fallback : '-';
  badge($('#m-fallback'), '回退: ' + (d ? fallbackLabel(fb) : '-'),
        d && fb !== 'none' ? 'bad' : 'ok');
  $('#m-motion').src = frameUrl('motion', k);
  $('#m-valid').src = frameUrl('valid', k);
  $('#m-weights').src = frameUrl('weights', k);
  $('#m-residual').src = frameUrl('residual', k);
  const table = $('#cand-table');
  table.innerHTML = `<tr><th>邻帧</th><th>类型</th><th>平均权重</th><th>覆盖</th><th>残差中位</th><th>说明</th></tr>`;
  const kinds = { dense: '稠密光流', translation: '全局平移',
                  fallback_identity: '无补偿（残差门）', dropped: '丢弃' };
  const notes = [
    ['outside clip window', '窗口越界'],
    ['source frame unusable/missing', '源帧缺失'],
    ['registration dropped', '配准失败'],
    ['compensation failed; residual gate only, no blind average', '补偿失败：只做残差门，不盲目平均'],
    ['clip boundary: shorter temporal window', '片头/片尾：时窗收缩'],
  ];
  const trNote = (s) => s ? notes.reduce((acc, [k, v]) => acc.replace(k, v), s) : '';
  (d?.candidates || []).forEach(c => {
    const tr = document.createElement('tr');
    tr.className = c.weight_mean > 0 ? '' : 'dropped';
    tr.innerHTML = `<td>${c.offset > 0 ? '+' : ''}${c.offset}</td>
      <td>${kinds[c.kind] || c.kind}</td><td>${c.weight_mean.toFixed(3)}</td>
      <td>${(c.overlap_ratio * 100).toFixed(0)}%</td>
      <td>${fmt(c.residual_median)}</td><td class="dim">${trNote(c.note)}</td>`;
    table.appendChild(tr);
  });
  $('#m-note').textContent = d
    ? `采用邻帧: ${(d.candidates || []).filter(c => c.weight_mean > 0).map(c => (c.offset > 0 ? '+' : '') + c.offset).join(', ') || '无（回退到当前帧）'}`
    : '';
}

/* ---------------------------------------------------------------- metrics */
function renderMetrics(j) {
  const body = $('#metrics-body'), empty = $('#metrics-empty');
  const m = j.summary?.metrics;
  if (!m) {
    body.classList.add('hidden'); empty.classList.remove('hidden');
    empty.textContent = '该片段没有清晰真值，无法计算客观指标（真实拍摄素材不冒充有真值）。';
    return;
  }
  empty.classList.add('hidden'); body.classList.remove('hidden');
  const cards = [
    ['PSNR 均值', fmt(m.psnr_out, 2), `输入 ${fmt(m.psnr_noisy, 2)} dB`, m.psnr_out > m.psnr_noisy ? 'good' : 'bad'],
    ['SSIM', fmt(m.ssim_out, 3), `输入 ${fmt(m.ssim_noisy, 3)}`, m.ssim_out > m.ssim_noisy ? 'good' : 'bad'],
    ['平坦区 σ', fmt(m.sigma_out_flat, 4), `输入 ${fmt(m.sigma_noisy_flat, 4)} · 真值 ${fmt(m.sigma_clean_flat, 4)}`, m.sigma_out_flat < m.sigma_noisy_flat ? 'good' : 'bad'],
    ['噪声降低', fmt(m.noise_reduction_ratio, 3), '输出/输入（越小越好）', m.noise_reduction_ratio < 0.75 ? 'good' : ''],
    ['细线细节保留', fmt(m.detail_retention_ratio, 3), '输出/输入对比度（越接近 1 越好）', m.detail_retention_ratio > 0.9 ? 'good' : (m.detail_retention_ratio < 0.8 ? 'bad' : '')],
    ['时域闪烁', fmt(m.flicker_out, 4), `输入 ${fmt(m.flicker_noisy, 4)} · 真值 ${fmt(m.flicker_clean, 4)}`, m.flicker_out < m.flicker_noisy ? 'good' : 'bad'],
  ];
  $('#metric-cards').innerHTML = cards.map(([k, v, d, cls]) =>
    `<div class="mcard ${cls}"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`
  ).join('');
  drawChart(j);
}

async function drawChart(j) {
  const cv = $('#metric-chart'); if (!cv) return;
  const data = await api(`/api/jobs/${j.id}/metrics`).catch(() => null);
  if (!data) return;
  const frames = data.frames || [];
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 600, h = 220;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + 'px'; cv.style.height = h + 'px';
  const ctx = cv.getContext('2d'); ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 44, r: 10, t: 14, b: 24 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const series = [
    ['psnr_noisy', '#7a93a0', '输入 PSNR'], ['psnr_out', '#25d0c0', '输出 PSNR'],
  ];
  const vals = frames.flatMap(f => [f.psnr_noisy, f.psnr_out]).filter(Number.isFinite);
  const lo = Math.min(...vals) - 1, hi = Math.max(...vals) + 1;
  const x = i => pad.l + pw * (i / Math.max(1, frames.length - 1));
  const y = v => pad.t + ph * (1 - (v - lo) / Math.max(1e-6, hi - lo));
  ctx.strokeStyle = '#1c3b4a'; ctx.fillStyle = '#7fa3b0'; ctx.font = '10px "NotoSC", ui-monospace, monospace';
  for (let g = 0; g <= 4; g++) {
    const vy = pad.t + ph * g / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, vy); ctx.lineTo(w - pad.r, vy); ctx.stroke();
    const v = hi - (hi - lo) * g / 4;
    ctx.fillText(v.toFixed(1) + ' dB', 4, vy + 3);
  }
  for (const [key, color] of series) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    frames.forEach((f, i) => {
      const v = f[key]; if (!Number.isFinite(v)) return;
      i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v));
    });
    ctx.stroke();
  }
  ctx.fillStyle = '#7fa3b0'; ctx.font = '10px "NotoSC", ui-monospace, monospace';
  ctx.fillText('帧', w - pad.r - 8, h - 6);
  ctx.fillStyle = '#25d0c0'; ctx.fillRect(w - 160, 6, 8, 8);
  ctx.fillStyle = '#7fa3b0'; ctx.fillText('输出 PSNR', w - 148, 14);
  ctx.fillStyle = '#7a93a0'; ctx.fillRect(w - 62, 6, 8, 8);
  ctx.fillStyle = '#7fa3b0'; ctx.fillText('输入', w - 50, 14);
}

/* ---------------------------------------------------------------- diag */
// Kernel fallback strings -> what they mean for the viewer.
const FALLBACK_LABELS = [
  ['no temporal evidence', '无时域证据：回退到本帧'],
  ['shortened window at clip boundary', '片头/片尾：时窗收缩'],
  ['motion compensation failed', '运动补偿失败：仅残差门'],
  ['source frame unusable/missing', '缺帧：不参与融合'],
  ['identity fallback', '退化为不平移'],
];

function fallbackLabel(s) {
  if (s === 'none' || !s) return '无（已采用邻帧）';
  const parts = s.split(';').map(x => x.trim()).filter(Boolean);
  return parts.map(p => {
    const hit = FALLBACK_LABELS.find(([k]) => p.includes(k));
    return hit ? hit[1] : p;
  }).join('；');
}

function renderDiags(j) {
  const warns = $('#warn-list'); warns.innerHTML = '';
  (j.summary?.warnings || []).forEach(w => {
    const el = document.createElement('div'); el.className = 'warn'; el.textContent = w;
    warns.appendChild(el);
  });
  const tbl = $('#diag-table');
  tbl.innerHTML = `<tr><th>#</th><th>t (s)</th><th>回退状态</th><th>时间证据</th><th>有效区</th>
    <th>采用邻帧</th><th>σ</th><th>说明</th></tr>`;
  (j.diags || []).forEach(d => {
    const tr = document.createElement('tr');
    const used = (d.candidates || []).filter(c => c.weight_mean > 0)
      .map(c => (c.offset > 0 ? '+' : '') + c.offset).join(',') || '—';
    const fbk = d.fallback !== 'none';
    const note = (d.notes || []).map(n => n
      .replace('outside clip window', '窗口越界（片头/片尾）')
      .replace('source frame unusable/missing', '缺帧：不参与融合')
      .replace(/residual/gi, '残差')
      .replace('exceeds gate', '超过门限')
      .replace('registration failed', '配准失败')
    ).join(' · ');
    tr.innerHTML = `<td>${d.frame_index}</td><td>${d.timestamp.toFixed(3)}</td>
      <td class="${fbk ? 'warn' : 'ok'}">${fallbackLabel(d.fallback)}</td>
      <td>${d.temporal_evidence.toFixed(2)}</td><td>${(d.valid_ratio * 100).toFixed(0)}%</td>
      <td>${used}</td><td>${d.noise_sigma.toFixed(3)}</td>
      <td class="dim">${note}</td>`;
    tbl.appendChild(tr);
  });
  $('#view-diag').dataset.frames = j.diags?.length || 0;
}

function renderDownloads(j) {
  const d = j.downloads || {};
  const box = $('#downloads'); box.innerHTML = '';
  const add = (href, label) => {
    if (!href) return;
    const a = document.createElement('a'); a.href = href; a.textContent = label;
    box.appendChild(a);
  };
  add(d.mp4, '播放版 MP4');
  add(d.lossless, '无损 MKV');
  add(d.input, '输入代理');
  add(d.truth, '真值代理');
  add(d.metrics, '逐帧指标 JSON');
}

/* ---------------------------------------------------------------- tabs */
$$('.tab').forEach(t => {
  t.onclick = () => {
    $$('.tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    state.view = t.dataset.view;
    $$('.view').forEach(v => v.classList.remove('active'));
    $('#view-' + state.view).classList.add('active');
    if (state.view === 'motion') refreshMotion();
  };
});
$$('.segbtn[data-mode]').forEach(b => b.onclick = () => {
  $$('.segbtn[data-mode]').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  state.mode = b.dataset.mode;
  const stage = $('#stage');
  stage.classList.toggle('swipe', state.mode === 'swipe');
  stage.classList.toggle('diff', state.mode === 'diff');
  ensureHandle();
  showFrame(state.frame);
});
$$('.segbtn[data-stream]').forEach(b => b.onclick = () => {
  $$('.segbtn[data-stream]').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  state.stream = b.dataset.stream;
  showFrame(state.frame);
});

function ensureHandle() {
  const stage = $('#stage');
  let h = $('#swipe-handle');
  if (state.mode === 'swipe' && !h) {
    h = document.createElement('div'); h.id = 'swipe-handle'; stage.appendChild(h);
    let down = false;
    const move = (e) => {
      if (!down) return;
      const r = stage.getBoundingClientRect();
      let f = (e.clientX - r.left) / r.width; f = Math.max(0, Math.min(1, f));
      state.wipe = f;
      $('#img-b').style.clipPath = `inset(0 0 0 ${f * 100}%)`;
      h.style.left = (f * 100) + '%';
      const tag = $('.stage.swipe .pane:nth-child(2) .tag');
      if (tag) tag.style.left = `calc(${f * 100}% + 8px)`;
    };
    h.onpointerdown = (e) => { down = true; h.setPointerCapture(e.pointerId); };
    h.onpointermove = move;
    h.onpointerup = () => down = false;
  }
  if (h) {
    h.style.left = (state.wipe * 100) + '%';
    $('#img-b').style.clipPath = `inset(0 0 0 ${state.wipe * 100}%)`;
  }
}

/* ---------------------------------------------------------------- transport */
function togglePlay() {
  if (state.playing) { stopPlay(); return; }
  if (!state.frames) return;
  state.playing = true; $('#play').textContent = '❚❚';
  const fps = (state.clip?.fps || 24) * state.rate;
  state.timer = setInterval(() => {
    let k = state.frame + 1;
    if (k >= state.frames) k = 0;
    showFrame(k);
  }, 1000 / fps);
}
function stopPlay() {
  state.playing = false; $('#play').textContent = '▶';
  if (state.timer) clearInterval(state.timer); state.timer = null;
}
$('#play').onclick = togglePlay;
$('#scrub').oninput = (e) => { stopPlay(); showFrame(+e.target.value); };
$('#fps-select').onchange = (e) => {
  state.rate = +e.target.value;
  if (state.playing) { stopPlay(); togglePlay(); }
};
$('#run').onclick = () => { stopPlay(); startJob(); };
$('#cancel').onclick = async () => {
  if (!state.job) return;
  await fetch(`/api/jobs/${state.job.id}/cancel`, { method: 'POST' }).catch(() => {});
};
document.addEventListener('keydown', (e) => {
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  if (e.key === 'ArrowRight') showFrame(state.frame + 1);
  if (e.key === 'ArrowLeft') showFrame(state.frame - 1);
});

loadLibrary().catch(e => {
  $('#clip-list').innerHTML = `<div class="empty">无法载入素材库：${e.message}</div>`;
});
