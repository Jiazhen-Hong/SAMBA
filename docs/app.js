(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const observer = new IntersectionObserver((entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add('in-view')), { threshold: .1 });
  $$('.reveal').forEach((el) => observer.observe(el));

  function canvasContext(canvas) {
    const ratio = Math.min(devicePixelRatio || 1, 2), rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width * ratio)), h = Math.max(1, Math.floor(rect.height * ratio));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  const hero = $('#heroCanvas');
  function drawHero(time = 0) {
    const { ctx, w, h } = canvasContext(hero); ctx.clearRect(0, 0, w, h);
    const phase = time * .00032, centerY = h * .49;
    const glow = ctx.createRadialGradient(w * .72, centerY, 10, w * .72, centerY, Math.max(w, h) * .46);
    glow.addColorStop(0, '#62d1c515'); glow.addColorStop(.45, '#f1c56f0d'); glow.addColorStop(1, '#10182000');
    ctx.fillStyle = glow; ctx.fillRect(0, 0, w, h);

    for (let row = 0; row < 18; row++) {
      ctx.beginPath();
      const offset = (row - 8.5) * Math.max(8, h * .027);
      for (let x = -20; x <= w + 20; x += 3) {
        const nx = x / Math.max(1, w), t = nx * 8.2 + phase + row * .11;
        const ribbon = Math.sin(t) * h * .12 + Math.sin(t * 2.05 + row * .17) * h * .025;
        const fold = Math.sin(nx * Math.PI) * offset;
        const y = centerY + ribbon + fold;
        x > -20 ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      const g = ctx.createLinearGradient(0, 0, w, 0);
      g.addColorStop(0, '#72839118'); g.addColorStop(.28, '#62d1c543'); g.addColorStop(.66, '#f1c56f66'); g.addColorStop(1, '#ff746160');
      ctx.strokeStyle = g; ctx.lineWidth = row % 5 === 0 ? 1.35 : .72; ctx.stroke();
    }

    const headX = w * .78, headY = h * .49, radius = Math.min(w, h) * .18;
    ctx.save(); ctx.globalAlpha = .78; ctx.strokeStyle = '#90a4ae55'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(headX, headY, radius, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(headX - radius * .14, headY - radius * .98); ctx.lineTo(headX, headY - radius * 1.18); ctx.lineTo(headX + radius * .14, headY - radius * .98); ctx.stroke();
    for (let ring = 0; ring < 5; ring++) {
      const count = 7 + ring * 3, rr = radius * (.18 + ring * .17);
      for (let i = 0; i < count; i++) {
        const angle = -Math.PI * .92 + i / Math.max(1, count - 1) * Math.PI * 1.84;
        const x = headX + Math.cos(angle) * rr, y = headY + Math.sin(angle) * rr * .9;
        const pulse = .65 + .35 * Math.sin(phase * 5 + i * .7 + ring);
        ctx.beginPath(); ctx.arc(x, y, Math.max(1.5, radius * .022) * (1 + pulse * .35), 0, Math.PI * 2);
        ctx.fillStyle = ring < 2 ? `rgba(98,209,197,${.24 + pulse * .3})` : `rgba(255,116,97,${.18 + pulse * .38})`; ctx.fill();
      }
    }
    ctx.restore();
    if (!reduced) requestAnimationFrame(drawHero);
  }
  drawHero();

  const state = { seq: 120, mask: 50, heads: 4, channels: 14, seed: 3, selected: 0 };
  const montageByChannels = { 2: 'mn8', 14: 'epocx', 32: 'standard32', 64: 'physionet64', 128: 'grosse128', 256: 'standard256' };
  const montages = window.SAMBA_MONTAGES || {};
  function currentMontage() { return montages[montageByChannels[state.channels]]; }
  function rand(i) { const x = Math.sin((i + state.seed * 31) * 12.9898) * 43758.5453; return x - Math.floor(x); }
  function signal(t, row) { return Math.sin(t * (.72 + row * .018) + row) * .48 + Math.sin(t * 2.2 + row * .3) * .2 + Math.sin(t * 6.2) * .07; }

  function setFill(input) { const p = (+input.value - +input.min) / (+input.max - +input.min) * 100; input.style.setProperty('--value', `${p}%`); }
  function maskBlocks(count) {
    const target = Math.round(count * state.mask / 100), masked = new Set(); let guard = 0;
    while (masked.size < target && guard++ < 1000) {
      const start = Math.floor(rand(guard * 7) * count), length = 1 + Math.floor(rand(guard * 11) * Math.max(2, count / 10));
      for (let i = start; i < Math.min(count, start + length) && masked.size < target; i++) masked.add(i);
    }
    return masked;
  }

  function drawSequence() {
    const canvas = $('#sequenceCanvas'), { ctx, w, h } = canvasContext(canvas); ctx.clearRect(0, 0, w, h);
    const rows = 6, panelH = h / 2, blocks = Math.max(16, Math.round(state.seq / 2)), masked = maskBlocks(blocks);
    ctx.font = '8px DM Mono, monospace'; ctx.fillStyle = '#647f85'; ctx.fillText('MASKED INPUT', 7, 13); ctx.fillText('RECONSTRUCTION', 7, panelH + 13);
    ctx.strokeStyle = '#17333d'; ctx.beginPath(); ctx.moveTo(0, panelH); ctx.lineTo(w, panelH); ctx.stroke();
    for (let row = 0; row < rows; row++) {
      for (let panel = 0; panel < 2; panel++) {
        const y0 = 30 + row * (panelH - 38) / rows + panel * panelH; ctx.beginPath();
        for (let x = 0; x <= w; x += 1.5) {
          const t = x / w * state.seq * .9; let v = signal(t, row);
          if (panel === 1) v = v * .96 + Math.sin(t * 2.5 + 1.2) * (state.mask / 100) * .045;
          const y = y0 - v * 13; x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.strokeStyle = panel ? '#ff6f61' : '#aeb8c4'; ctx.lineWidth = 1; ctx.stroke();
      }
    }
    masked.forEach((i) => { const x = i * w / blocks, bw = w / blocks + .5; ctx.fillStyle = '#17353be0'; ctx.fillRect(x, 19, bw, panelH - 25); ctx.strokeStyle = '#55c7d4'; ctx.setLineDash([3, 3]); ctx.strokeRect(x, 19, bw, panelH - 25); ctx.setLineDash([]); });
  }

  let differentialFrame = 0, displayedNoiseLevel = null;
  function differentialNoiseTarget() { return 1 - (state.heads - 2) / 6 * .9; }
  function drawDifferentialSignal(noiseLevel = differentialNoiseTarget(), phase = 0) {
    const canvas = $('#diffCanvas'), { ctx, w, h } = canvasContext(canvas); ctx.clearRect(0, 0, w, h);
    const pad = Math.max(20, w * .025), plotLeft = pad, plotRight = w - pad;
    const top = 34, bottom = h - 24, baseline = h * .56, amplitude = Math.min(82, h * .29), duration = 24;
    const neural = (t) => Math.sin(t * .47 - .7) * .38 + Math.sin(t * 1.16 + .5) * .18 + Math.sin(t * 2.7 - .2) * .075 + Math.sin(t * .13) * .12;
    const hash = (index, layer) => {
      const value = Math.sin((index + 1) * 12.9898 + (layer + 1) * 78.233 + state.seed * 19.19 + phase * 2.7) * 43758.5453;
      return (value - Math.floor(value)) * 2 - 1;
    };
    const noise = (index, layer = 0) => {
      const white = hash(index, layer) * .66 + hash(index + 17, layer + 3) * .24;
      const burstSeed = Math.abs(hash(Math.floor(index / 19), layer + 8));
      const burst = burstSeed > .82 ? hash(index + 41, layer + 11) * (burstSeed - .82) * 3.6 : 0;
      return white + burst + Math.sin(index * .34 + layer * .8 + phase * 4) * .16;
    };
    const trace = (color, width, alpha, valueAt) => {
      ctx.beginPath();
      let sample = 0;
      for (let x = plotLeft; x <= plotRight; x += 1.4, sample++) {
        const t = (x - plotLeft) / (plotRight - plotLeft) * duration, y = baseline - valueAt(t, sample) * amplitude;
        x > plotLeft ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.restore();
    };

    ctx.fillStyle = '#111b23'; ctx.fillRect(.5, .5, w - 1, h - 1);
    ctx.strokeStyle = '#31414c'; ctx.lineWidth = 1; ctx.strokeRect(.5, .5, w - 1, h - 1);
    for (let grid = 1; grid < 8; grid++) {
      const x = plotLeft + (plotRight - plotLeft) * grid / 8;
      ctx.strokeStyle = '#31414c55'; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    }
    for (let grid = 1; grid < 4; grid++) {
      const y = top + (bottom - top) * grid / 4;
      ctx.strokeStyle = '#31414c55'; ctx.beginPath(); ctx.moveTo(plotLeft, y); ctx.lineTo(plotRight, y); ctx.stroke();
    }
    ctx.strokeStyle = '#52616b88'; ctx.beginPath(); ctx.moveTo(plotLeft, baseline); ctx.lineTo(plotRight, baseline); ctx.stroke();
    ctx.font = '8px DM Mono, monospace'; ctx.fillStyle = '#8ea2ac';
    ctx.fillText('EXAMPLE CHANNEL · AF3', plotLeft, 18);
    ctx.fillStyle = '#ff8c7b'; ctx.fillText(`NOISE AMPLITUDE · ${Math.round(noiseLevel * 100)}%`, plotRight - 148, 18);

    for (let layer = 1; layer <= 4; layer++) {
      trace('#ff7461', .8, .08 + noiseLevel * .06, (t, sample) => neural(t) + noise(sample, layer) * noiseLevel * .62);
    }
    trace('#ff7461', 1.15, .72, (t, sample) => neural(t) + noise(sample) * noiseLevel * .7);
    trace('#62d1c5', 1.8, .78, neural);

    displayedNoiseLevel = noiseLevel;
    $('#headPairCount').textContent = `${state.heads} PAIRED HEADS · ${state.heads * 2} MAMBA BRANCHES`;
    $('#noiseResidual').textContent = `${Math.round(noiseLevel * 100)}% illustrative noise amplitude`;
  }

  function animateDifferentialSignal() {
    cancelAnimationFrame(differentialFrame);
    const target = differentialNoiseTarget(), start = displayedNoiseLevel === null ? target : displayedNoiseLevel;
    if (reduced || Math.abs(target - start) < .001) { drawDifferentialSignal(target); return; }
    const started = performance.now(), duration = 460;
    const frame = (now) => {
      const progress = Math.min(1, (now - started) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      drawDifferentialSignal(start + (target - start) * eased, progress);
      if (progress < 1) differentialFrame = requestAnimationFrame(frame);
    };
    differentialFrame = requestAnimationFrame(frame);
  }

  function renderDifferentialHeads() {
    const container = $('#diffHeads'); container.innerHTML = '';
    for (let index = 0; index < state.heads; index++) {
      const head = document.createElement('div'); head.className = 'diff-head-card';
      head.innerHTML = `<span class="mono">H${index + 1}</span><div><i>A</i><i>B</i></div><b>− λ</b><strong>Δ</strong>`;
      container.appendChild(head);
    }
  }

  function updateDifferentialDemo() { renderDifferentialHeads(); animateDifferentialSignal(); }

  function updateGrowth() {
    const baselineSeconds = 120;
    const ratio = state.seq / baselineSeconds, quad = ratio * ratio;
    const maxRatio = +$('#seq').max / baselineSeconds, maxGrowth = maxRatio * maxRatio;
    const sharedLogScale = (value) => Math.log1p(value) / Math.log1p(maxGrowth) * 100;
    $('#tokenCount').textContent = `${(state.seq * 128).toLocaleString()} time steps`;
    $('#transformerGrowth').textContent = `×${quad.toFixed(1)}`; $('#mambaGrowth').textContent = `×${ratio.toFixed(1)}`;
    $('#transformerBar').style.width = `${Math.max(2, sharedLogScale(quad))}%`;
    $('#mambaBar').style.width = `${Math.max(2, sharedLogScale(ratio))}%`;
  }

  const benchmark = {
    sampleRate: 128,
    measuredMax: 2000,
    maxSteps: 600 * 128,
    memory: { points: [[200, 207.20], [2000, 1437.54]], unit: 'MB', decimals: 0 },
    inference: { points: [[200, .28], [2000, .38]], unit: 'ms', decimals: 2 }
  };

  function affineEstimate(points, steps) {
    const [[x1, y1], [x2, y2]] = points;
    const slope = (y2 - y1) / (x2 - x1);
    return Math.max(0, y1 + (steps - x1) * slope);
  }

  function drawResourceChart(canvas, series, currentSteps) {
    const { ctx, w, h } = canvasContext(canvas);
    ctx.clearRect(0, 0, w, h);
    const pad = { left: 39, right: 12, top: 19, bottom: 28 };
    const plotW = Math.max(1, w - pad.left - pad.right), plotH = Math.max(1, h - pad.top - pad.bottom);
    const x = (steps) => pad.left + steps / benchmark.maxSteps * plotW;
    const maxValue = affineEstimate(series.points, benchmark.maxSteps) * 1.12;
    const y = (value) => pad.top + plotH - value / maxValue * plotH;
    const boundaryX = x(benchmark.measuredMax);

    ctx.font = '7px DM Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#617b81';
    ctx.strokeStyle = '#17343d';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const value = maxValue * i / 3, yy = y(value);
      ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(w - pad.right, yy); ctx.stroke();
      const label = series.unit === 'MB' ? `${Math.round(value / 100) * 100}` : value.toFixed(1);
      ctx.fillText(label, pad.left - 6, yy + 3);
    }

    ctx.textAlign = 'center';
    [0, 120, 240, 360, 480, 600].forEach((seconds) => {
      const xx = x(seconds * benchmark.sampleRate);
      ctx.fillStyle = '#617b81'; ctx.fillText(`${seconds}s`, xx, h - 8);
    });

    ctx.fillStyle = '#171b24';
    ctx.fillRect(boundaryX, pad.top, w - pad.right - boundaryX, plotH);
    ctx.save();
    ctx.setLineDash([4, 4]); ctx.strokeStyle = '#ffb05caa';
    ctx.beginPath(); ctx.moveTo(boundaryX, pad.top); ctx.lineTo(boundaryX, pad.top + plotH); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = '#ffc17f'; ctx.font = '7px DM Mono, monospace'; ctx.textAlign = 'left';
    ctx.fillText('REPORTED TO 2,000', Math.min(boundaryX + 5, w - 105), pad.top + 9);

    const start = series.points[0], end = series.points[1];
    ctx.strokeStyle = '#55c7d4'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x(start[0]), y(start[1])); ctx.lineTo(x(end[0]), y(end[1])); ctx.stroke();

    ctx.save();
    ctx.setLineDash([7, 5]); ctx.strokeStyle = '#ffb05c';
    ctx.beginPath(); ctx.moveTo(x(end[0]), y(end[1])); ctx.lineTo(x(benchmark.maxSteps), y(affineEstimate(series.points, benchmark.maxSteps))); ctx.stroke();
    ctx.restore();

    series.points.forEach(([steps, value]) => {
      ctx.beginPath(); ctx.arc(x(steps), y(value), 3.5, 0, Math.PI * 2); ctx.fillStyle = '#55c7d4'; ctx.fill();
      ctx.strokeStyle = '#0e1017'; ctx.lineWidth = 1; ctx.stroke();
    });

    const currentValue = affineEstimate(series.points, currentSteps);
    ctx.beginPath(); ctx.arc(x(currentSteps), y(currentValue), 5, 0, Math.PI * 2);
    ctx.fillStyle = currentSteps > benchmark.measuredMax ? '#ffb05c' : '#fff2ef'; ctx.fill();
    ctx.strokeStyle = '#0e1017'; ctx.lineWidth = 2; ctx.stroke();
  }

  function updateResourceForecast() {
    const steps = state.seq * benchmark.sampleRate;
    const memory = affineEstimate(benchmark.memory.points, steps);
    const inference = affineEstimate(benchmark.inference.points, steps);
    $('#memoryEstimate').textContent = `${Math.round(memory).toLocaleString()} MB`;
    $('#inferenceEstimate').textContent = `${inference.toFixed(2)} ms`;
    const estimated = steps > benchmark.measuredMax;
    const status = $('#estimateStatus');
    status.classList.toggle('estimated', estimated);
    status.classList.toggle('measured', !estimated);
    status.querySelector('span').textContent = estimated ? 'ESTIMATED' : 'MEASURED RANGE';
    status.querySelector('small').textContent = estimated ? 'Past reported resource points' : 'Between reported resource points';
    drawResourceChart($('#memoryChart'), benchmark.memory, steps);
    drawResourceChart($('#inferenceChart'), benchmark.inference, steps);
  }

  function updatePlay() {
    const montage = currentMontage();
    $('#seqValue').textContent = `${state.seq} s`; $('#maskValue').textContent = `${state.mask}%`; $('#headValue').textContent = state.heads; $('#montageValue').textContent = montage ? `${montage.label} · ${montage.channels} ch` : `${state.channels} ch`; $('#visibleValue').textContent = `${100 - state.mask}%`;
    ['seq','mask','heads'].forEach((id) => setFill($(`#${id}`))); drawSequence(); updateDifferentialDemo(); updateGrowth(); updateResourceForecast(); renderScalps();
  }
  $('#seq').addEventListener('input', (e) => { state.seq = +e.target.value; updatePlay(); });
  $('#mask').addEventListener('input', (e) => { state.mask = +e.target.value; updatePlay(); });
  $('#heads').addEventListener('input', (e) => { state.heads = +e.target.value; updatePlay(); });
  $$('#montageButtons button').forEach((b) => b.addEventListener('click', () => { state.channels = +b.dataset.ch; $$('#montageButtons button').forEach((x) => x.classList.toggle('active', x === b)); state.selected = 0; updatePlay(); }));
  $('#resample').addEventListener('click', () => { state.seed++; drawSequence(); });
  $('#reset').addEventListener('click', () => { Object.assign(state, { seq: 120, mask: 50, heads: 4, channels: 14, seed: 3, selected: 0 }); $('#seq').value = 120; $('#mask').value = 50; $('#heads').value = 4; $$('#montageButtons button').forEach((b) => b.classList.toggle('active', b.dataset.ch === '14')); updatePlay(); });

  const method = {
    tsr: ['SELF-SUPERVISION','Hide EEG spans','Hide part of the signal, then rebuild it.','Learn the larger pattern.',['hidden EEG','context','rebuilt EEG']],
    cdse: ['SHARED EEG SPACE','Unify headset layouts','3D electrode positions map any headset into one pretraining space.','Train across different layouts.',['input headset','3D map','shared space']],
    mamba: ['DENOISING','Reduce shared noise','Paired paths subtract repeated responses.','Keep useful changes.',['path A','difference','path B']],
    decoder: ['RECONSTRUCTION','Rebuild the signal','Restore hidden EEG at several scales.','Use short and long context.',['compress','combine','rebuild']]
  };
  $$('.method-tabs button').forEach((button) => button.addEventListener('click', () => {
    $$('.method-tabs button').forEach((b) => b.classList.toggle('active', b === button)); const d = method[button.dataset.step];
    $('#methodKicker').textContent = d[0]; $('#methodTitle').textContent = d[1]; $('#methodText').textContent = d[2]; $('#methodNote').textContent = d[3]; $('#methodVisual').dataset.step = button.dataset.step;
    ['tag1','tag2','tag3'].forEach((c, i) => $(`.${c}`).textContent = d[4][i]);
  }));

  const svgNS = 'http://www.w3.org/2000/svg';
  const headOutline = '<path d="M48 135c-18 0-22 15-18 31 3 13 12 21 23 19M252 135c18 0 22 15 18 31-3 13-12 21-23 19M137 49l13-21 13 21" fill="none" stroke="#485563" stroke-width="2"/><circle cx="150" cy="150" r="105" fill="#111821" stroke="#485563"/>';

  function projectPoint(point) {
    return { x: 150 + point.x / .105 * 102, y: 150 - point.y / .12 * 100 };
  }

  const methodHeadOutline = '<path d="M48 135c-18 0-22 15-18 31 3 13 12 21 23 19M252 135c18 0 22 15 18 31-3 13-12 21-23 19M137 49l13-21 13 21" fill="none" stroke="#58717d" stroke-width="2"/><circle cx="150" cy="150" r="105" fill="#17232d" stroke="#58717d" stroke-width="2"/>';

  function renderMethodTopo(svg, montage, fill) {
    if (!svg || !montage) return;
    svg.innerHTML = methodHeadOutline;
    const radius = montage.points.length <= 14 ? 6.5 : montage.points.length <= 64 ? 4.1 : montage.points.length <= 128 ? 3 : 2.1;
    montage.points.forEach((point) => {
      const projected = projectPoint(point), circle = document.createElementNS(svgNS, 'circle');
      circle.setAttribute('cx', projected.x); circle.setAttribute('cy', projected.y); circle.setAttribute('r', radius);
      circle.setAttribute('fill', fill); circle.setAttribute('opacity', '.82');
      circle.setAttribute('stroke', '#f7f1ef'); circle.setAttribute('stroke-width', '.55');
      svg.appendChild(circle);
    });
  }

  function renderMethodCdse(montage, target) {
    renderMethodTopo($('#methodInputTopo'), montage, '#3f9daa');
    renderMethodTopo($('#methodTargetTopo'), target, '#e35e51');
    $('#methodInputLabel').textContent = `${montage.label} · ${montage.channels} ch`;
    $('#methodTargetLabel').textContent = `${target.label} · ${target.channels} ch`;
  }

  function electrodeRadius(count) {
    if (count <= 14) return 7;
    if (count <= 64) return 4.3;
    if (count <= 128) return 3.1;
    return 2.25;
  }

  function proxyWeights(inputPoints, targetPoints) {
    const sigma = inputPoints.length <= 2 ? .08 : inputPoints.length <= 14 ? .052 : .038;
    const sigma2 = 2 * sigma * sigma;
    return targetPoints.map((target, targetIndex) => {
      const logits = inputPoints.map((input) => {
        const dx = target.x - input.x, dy = target.y - input.y, dz = target.z - input.z;
        return -(dx * dx + dy * dy + dz * dz) / sigma2;
      });
      const maxLogit = Math.max(...logits);
      const exp = logits.map((value) => Math.exp(value - maxLogit));
      const denominator = exp.reduce((sum, value) => sum + value, 0);
      return { targetIndex, value: exp[state.selected] / denominator };
    });
  }

  function relativeWeights(weights) {
    const values = weights.map((weight) => weight.value), min = Math.min(...values), max = Math.max(...values);
    return weights.map((weight) => ({ ...weight, relative: max === min ? 1 : (weight.value - min) / (max - min) }));
  }

  function weightColor(relative) {
    const hue = 34 - 26 * Math.pow(relative, .78);
    const light = 55 + 9 * relative;
    return `hsl(${hue.toFixed(0)} 88% ${light.toFixed(0)}%)`;
  }

  function makeTitle(text) {
    const title = document.createElementNS(svgNS, 'title');
    title.textContent = text;
    return title;
  }

  function renderInputScalp(svg, montage) {
    svg.innerHTML = headOutline;
    const radius = electrodeRadius(montage.points.length);
    montage.points.forEach((point, index) => {
      const projected = projectPoint(point), selected = index === state.selected;
      const group = document.createElementNS(svgNS, 'g');
      group.setAttribute('class', `electrode input-electrode${selected ? ' active' : ''}`);
      group.setAttribute('data-index', index);
      group.setAttribute('role', 'button');
      group.setAttribute('tabindex', '0');
      group.setAttribute('aria-label', `Select ${point.name}`);
      group.appendChild(makeTitle(`${point.name} · (${point.x.toFixed(3)}, ${point.y.toFixed(3)}, ${point.z.toFixed(3)}) m`));
      const circle = document.createElementNS(svgNS, 'circle');
      circle.setAttribute('cx', projected.x); circle.setAttribute('cy', projected.y); circle.setAttribute('r', selected ? radius + 1.6 : radius);
      circle.setAttribute('fill', selected ? '#ff6f61' : '#55c7d4'); circle.setAttribute('opacity', selected ? '1' : '.72');
      circle.setAttribute('stroke', selected ? '#fff2ef' : '#0e1017'); circle.setAttribute('stroke-width', selected ? '1.4' : '.6');
      group.appendChild(circle);
      const pick = () => { state.selected = index; renderScalps(); };
      group.addEventListener('click', pick);
      group.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); pick(); } });
      svg.appendChild(group);
    });
  }

  function renderTargetScalp(svg, targetMontage, weights, selectedName) {
    svg.innerHTML = headOutline;
    weights.forEach((weight) => {
      const point = targetMontage.points[weight.targetIndex], projected = projectPoint(point);
      const visual = .2 + .8 * Math.pow(weight.relative, .45);
      const group = document.createElementNS(svgNS, 'g');
      group.setAttribute('class', 'electrode target-electrode');
      group.setAttribute('data-index', weight.targetIndex);
      group.appendChild(makeTitle(`${point.name}: ${(weight.value * 100).toFixed(2)}% contribution from ${selectedName}`));
      const circle = document.createElementNS(svgNS, 'circle');
      circle.setAttribute('cx', projected.x); circle.setAttribute('cy', projected.y);
      circle.setAttribute('r', 3.5 + visual * 4.1); circle.setAttribute('fill', weightColor(weight.relative));
      circle.setAttribute('opacity', .45 + visual * .5); circle.setAttribute('stroke', '#16090c'); circle.setAttribute('stroke-width', '.65');
      group.appendChild(circle); svg.appendChild(group);
    });
  }

  function renderProjectionBars(weights) {
    const box = $('#projectionLines'); box.innerHTML = '';
    [...weights].sort((a, b) => b.value - a.value).slice(0, 8).forEach((weight, index) => {
      const line = document.createElement('i'); line.style.top = `${24 + index * 7}%`; line.style.width = `${48 + 42 * weight.relative}%`;
      line.style.opacity = `${.28 + .62 * weight.relative}`; line.style.transform = `rotate(${(index - 3.5) * 3.2}deg)`; line.style.background = `linear-gradient(90deg,transparent,${weightColor(weight.relative)},transparent)`; box.appendChild(line);
    });
  }

  function renderMappingOverlay(weights) {
    const overlay = $('#mappingOverlay'), lab = $('.spatial-lab'), selected = $('#inputScalp .input-electrode.active circle');
    if (!overlay || !lab || !selected) return;
    const labRect = lab.getBoundingClientRect(), sourceRect = selected.getBoundingClientRect();
    overlay.setAttribute('viewBox', `0 0 ${labRect.width} ${labRect.height}`); overlay.innerHTML = '';
    const sourceX = sourceRect.left + sourceRect.width / 2 - labRect.left, sourceY = sourceRect.top + sourceRect.height / 2 - labRect.top;
    [...weights].sort((a, b) => a.value - b.value).forEach((weight) => {
      const target = $(`#targetScalp .target-electrode[data-index="${weight.targetIndex}"] circle`); if (!target) return;
      const targetRect = target.getBoundingClientRect(), targetX = targetRect.left + targetRect.width / 2 - labRect.left, targetY = targetRect.top + targetRect.height / 2 - labRect.top;
      const path = document.createElementNS(svgNS, 'path');
      const bend = Math.max(45, Math.abs(targetX - sourceX) * .32);
      path.setAttribute('d', `M ${sourceX} ${sourceY} C ${sourceX + bend} ${sourceY}, ${targetX - bend} ${targetY}, ${targetX} ${targetY}`);
      const emphasis = Math.pow(weight.relative, .85);
      path.setAttribute('fill', 'none'); path.setAttribute('stroke', weightColor(weight.relative)); path.setAttribute('stroke-width', .45 + emphasis * 2.5); path.setAttribute('opacity', .035 + emphasis * .48);
      overlay.appendChild(path);
    });
  }

  function renderScalps() {
    const montage = currentMontage(), target = montages.target64;
    if (!montage || !target) return;
    state.selected = Math.min(state.selected, montage.points.length - 1);
    const selected = montage.points[state.selected], weights = relativeWeights(proxyWeights(montage.points, target.points));
    renderInputScalp($('#inputScalp'), montage); renderTargetScalp($('#targetScalp'), target, weights, selected.name); renderProjectionBars(weights); renderMethodCdse(montage, target);
    $('#inputMapTitle').textContent = `${montage.label} · ${montage.channels} channels${montage.schematic ? ' · schematic' : ''}`;
    $('#selectedChannel').textContent = selected.name;
    const strongest = [...weights].sort((a, b) => b.value - a.value)[0];
    $('#weightSummary').textContent = `Peak: ${target.points[strongest.targetIndex].name} · ${(strongest.value * 100).toFixed(1)}%`;
    requestAnimationFrame(() => renderMappingOverlay(weights));
  }

  const tasks = {
    eyes:['assets/topography/eyes-state.gif','FRONTAL REGION','Eyes open / closed','Focuses on frontal signals linked to eye closure.'],
    driver:['assets/topography/driver-distraction.gif','LEFT TEMPORAL REGION','Driver distraction','Focuses on left temporal signals during distraction.'],
    workload:['assets/topography/workload.gif','LEFT FRONTAL REGION','Mental workload','Focuses on left frontal signals during workload.'],
    tuab:['assets/topography/tuab.gif','OCCIPITAL REGION','Abnormal EEG','Focuses on occipital signals for abnormal EEG.'],
    motor:['assets/topography/motor-imagery.gif','MOTOR CORTEX','Motor imagery','Focuses near sensorimotor regions during imagined movement.']
  };
  $$('.task-list button').forEach((button)=>button.addEventListener('click',()=>{ $$('.task-list button').forEach((b)=>b.classList.toggle('active',b===button)); const d=tasks[button.dataset.task]; $('#topoImage').src=d[0]; $('#topoImage').alt=`Animated SAMBA spatial weights for ${d[2]}`; $('#topoRegion').textContent=d[1]; $('#topoTitle').textContent=d[2]; $('#topoText').textContent=d[3]; }));

  function copy(button,text,label){navigator.clipboard.writeText(text).then(()=>{button.textContent='Copied';setTimeout(()=>button.textContent=label,1400)}).catch(()=>button.textContent='Select text')}
  if ($('#copyCode')) $('#copyCode').addEventListener('click',()=>copy($('#copyCode'),$('.code-card code').innerText,'Copy'));
  const bib=`@inproceedings{hong2026samba,\n  title={{SAMBA}: Toward an {EEG} Foundation Model via Differential Mamba and Coordinate-Driven Spatial Embedding},\n  author={Hong, Jiazhen and Mackellar, Geoffrey and Ghane, Soheila},\n  booktitle={NeurReps 2026: Symmetry and Geometry in Neural Representations},\n  year={2026},\n  note={Proceedings Track, to appear in Proceedings of Machine Learning Research}\n}`;
  $('#copyCitation').addEventListener('click',()=>copy($('#copyCitation'),bib,'Copy BibTeX'));
  addEventListener('resize',()=>{drawSequence();drawDifferentialSignal();updateResourceForecast();renderScalps();if(reduced)drawHero()});
  updatePlay();
})();
