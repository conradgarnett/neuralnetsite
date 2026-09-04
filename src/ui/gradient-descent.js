// Tab 4 — Gradient Descent.
//
// The update rule with live numbers, the learning-rate knob, the step controls,
// and the loss surface with the descent trajectory drawn on it.
//
// The loss surface is an honest 2-D *slice*: two chosen parameters are varied
// over a grid while the other fifteen are held at their current values, and the
// mean loss over the dataset is evaluated at every grid point. That is why the
// surface reshapes as training proceeds — the slice itself is moving.

import { el, fmt, num, pd, COLORS, lossColor, signedColor, classColor } from './dom.js';
import { Plot, marchingSquares } from './plots.js';
import {
  PARAM_SPECS,
  PARAM_BY_ID,
  getParam,
  setParam,
  getGrad,
  cloneNetwork,
  batchLossFlat,
  flattenDataset,
  batchGradients,
  gradientDescentStep,
  forward,
} from '../math/network.js';

const LR_MIN = 0.001;
const LR_MAX = 30;
const PRESETS = [
  { lr: 0.005, label: 'far too small', tone: 'cold' },
  { lr: 0.5, label: 'a good default', tone: 'good' },
  { lr: 5, label: 'too large', tone: 'warm' },
  { lr: 25, label: 'wildly too large', tone: 'hot' },
];

/** Slider position (0..1) <-> learning rate, on a log scale. */
const posToLr = (p) => LR_MIN * Math.pow(LR_MAX / LR_MIN, p);
const lrToPos = (lr) => Math.log(lr / LR_MIN) / Math.log(LR_MAX / LR_MIN);

export function createGradientDescentTab(host, store) {
  const refs = {};
  const surface = { key: null, canvas: null, win: null, lo: 0, hi: 1, stamp: 0 };

  host.replaceChildren(build());
  wire();

  function build() {
    return el('div', { class: 'stack' }, [
      // ---- the rule itself --------------------------------------------------
      el('div', { class: 'card' }, [
        el('h3', { text: 'The update rule' }),
        el('p', {
          class: 'blurb tight',
          html:
            'One line is the whole of gradient descent. The gradient says which way the loss increases; ' +
            'subtracting a multiple of it walks the other way. η is the learning rate — the only thing you tune.',
        }),
        refs.ruleBig = el('div', { class: 'rule-big' }),
        refs.ruleLive = el('div', { class: 'work' }),
      ]),

      // ---- controls ---------------------------------------------------------
      el('div', { class: 'card' }, [
        el('div', { class: 'card-head' }, [
          el('h3', { text: 'Controls' }),
          refs.phase = el('span', { class: 'pill' }),
        ]),
        el('div', { class: 'lr-row' }, [
          el('label', { class: 'slider grow' }, [
            el('span', { class: 'slabel', text: 'learning rate η' }),
            refs.lr = el('input', { type: 'range', min: 0, max: 1, step: 0.0005 }),
            refs.lrOut = el('span', { class: 'sval wide' }),
          ]),
          refs.presets = el('div', { class: 'row gap' }),
        ]),
        refs.lrHint = el('div', { class: 'lr-hint' }),
        el('div', { class: 'row gap controls' }, [
          refs.stepF = el('button', { class: 'btn', text: '1 · Step forward' }),
          refs.stepB = el('button', { class: 'btn', text: '2 · Step backward' }),
          refs.update = el('button', { class: 'btn', text: '3 · Update weights' }),
          el('span', { class: 'divider' }),
          refs.epoch = el('button', { class: 'btn primary', text: 'Run one epoch' }),
          refs.run = el('button', { class: 'btn primary', text: '▶ Run training' }),
          el('span', { class: 'divider' }),
          refs.reset = el('button', { class: 'btn ghost', text: '⟲ Reset weights' }),
        ]),
        el('div', { class: 'row gap sub-controls' }, [
          el('label', { class: 'slider' }, [
            el('span', { class: 'slabel', text: 'gradient from' }),
            refs.mode = el('select', { class: 'select' }, [
              el('option', { value: 'batch', text: 'all 200 points (full batch)' }),
              el('option', { value: 'sgd', text: 'the selected point only (SGD)' }),
            ]),
          ]),
          el('label', { class: 'slider' }, [
            el('span', { class: 'slabel', text: 'epochs / frame' }),
            refs.speed = el('input', { type: 'range', min: 1, max: 25, step: 1 }),
            refs.speedOut = el('span', { class: 'sval narrow' }),
          ]),
          refs.stats = el('span', { class: 'stats' }),
        ]),
        refs.diagnosis = el('div', { class: 'diagnosis' }),
      ]),

      // ---- loss surface -----------------------------------------------------
      el('div', { class: 'card' }, [
        el('div', { class: 'card-head' }, [
          el('h3', { text: 'Descent on the loss surface' }),
          el('div', { class: 'row gap' }, [
            el('label', { class: 'slider sm' }, [
              el('span', { class: 'slabel', text: 'x-axis' }),
              refs.axisX = el('select', { class: 'select' }),
            ]),
            el('label', { class: 'slider sm' }, [
              el('span', { class: 'slabel', text: 'y-axis' }),
              refs.axisY = el('select', { class: 'select' }),
            ]),
            el('label', { class: 'slider sm' }, [
              el('span', { class: 'slabel', text: 'zoom' }),
              refs.zoom = el('input', { type: 'range', min: 0.4, max: 6, step: 0.1, value: 2.5 }),
            ]),
          ]),
        ]),
        el('p', {
          class: 'blurb tight',
          html:
            'A 2-D slice through the 17-dimensional loss surface: these two parameters vary, the other fifteen stay ' +
            'where they currently are. Brighter is higher loss. The white trail is where training has been, the green ' +
            'arrow is the step it is about to take, and the dashed line previews the next few steps at the current η.',
        }),
        el('div', { class: 'two-col equal' }, [
          el('div', { class: 'plot-box' }, [
            refs.surfaceCanvas = el('canvas', { class: 'canvas h360' }),
            refs.surfaceNote = el('div', { class: 'legend' }),
          ]),
          el('div', { class: 'stack' }, [
            el('div', { class: 'plot-box' }, [
              el('div', { class: 'mini-label', text: 'loss vs epoch' }),
              refs.lossCanvas = el('canvas', { class: 'canvas h170' }),
            ]),
            el('div', { class: 'plot-box' }, [
              el('div', { class: 'mini-label', text: 'decision boundary' }),
              refs.boundaryCanvas = el('canvas', { class: 'canvas h170' }),
            ]),
          ]),
        ]),
      ]),

      // ---- every parameter --------------------------------------------------
      el('div', { class: 'card' }, [
        el('h3', { text: 'Every parameter, this update' }),
        el('p', { class: 'blurb tight', html: 'Move the learning-rate slider and watch the whole table recompute. Click a row to inspect its derivation on the Backpropagation tab.' }),
        refs.table = el('div', { class: 'ptable-wrap' }),
      ]),
    ]);
  }

  function wire() {
    refs.lr.value = lrToPos(store.state.learningRate);
    refs.speed.value = store.state.speed;

    refs.lr.addEventListener('input', () => {
      store.setLearningRate(round3(posToLr(parseFloat(refs.lr.value))));
    });
    refs.speed.addEventListener('input', () => {
      store.set({ speed: parseInt(refs.speed.value, 10) });
    });
    for (const p of PRESETS) {
      const b = el('button', { class: `btn sm preset ${p.tone}`, html: `η = ${p.lr}<span>${p.label}</span>` });
      b.addEventListener('click', () => {
        refs.lr.value = lrToPos(p.lr);
        store.setLearningRate(p.lr);
      });
      refs.presets.append(b);
    }

    refs.stepF.addEventListener('click', () => store.stepForward());
    refs.stepB.addEventListener('click', () => store.stepBackward());
    refs.update.addEventListener('click', () => store.updateWeights());
    refs.epoch.addEventListener('click', () => store.runEpoch());
    refs.run.addEventListener('click', () => store.toggleRunning());
    refs.reset.addEventListener('click', () => store.resetWeights());
    refs.mode.addEventListener('change', () => store.setMode(refs.mode.value));

    for (const sel of [refs.axisX, refs.axisY]) {
      for (const spec of PARAM_SPECS) {
        sel.append(el('option', { value: spec.id, text: spec.text }));
      }
      sel.addEventListener('change', () => {
        store.set({ surfaceX: refs.axisX.value, surfaceY: refs.axisY.value });
        surface.key = null;
        surface.win = null;
      });
    }
    refs.zoom.addEventListener('input', () => {
      surface.key = null;
      surface.win = null;
      render();
    });
  }

  const round3 = (v) => Number(v.toPrecision(3));

  // -------------------------------------------------------------------------
  // The update rule, with live numbers
  // -------------------------------------------------------------------------

  function renderRule(s) {
    const spec = PARAM_BY_ID[s.inspectedParam] ?? PARAM_SPECS[0];
    const grads = s.pendingGrads ?? trainingGradients(s);
    const w = getParam(s.net, spec);
    const g = getGrad(grads, spec);
    const stepSize = s.learningRate * g;
    const next = w - stepSize;

    refs.ruleBig.innerHTML = `
      <span class="rb-sym">w</span>
      <span class="rb-op">:=</span>
      <span class="rb-sym">w</span>
      <span class="rb-op">−</span>
      <span class="rb-eta">η</span>
      <span class="rb-op">·</span>
      <span class="rb-grad">${pd('L', 'w')}</span>`;

    refs.ruleLive.innerHTML = `
      <div class="wline big">
        <span class="lhs">${spec.html}</span><span class="eq">:=</span>
        <span class="sub">${fmt(w, 5)} <span class="op">−</span> ${fmt(s.learningRate, 4)} <span class="op">×</span> ${fmt(g, 5)}</span>
        <span class="eq">=</span>${num(next, 5)}
      </div>
      <div class="rule-parts">
        <span><b>w</b> current value ${num(w, 5)}</span>
        <span><b>η</b> learning rate ${fmt(s.learningRate, 4)}</span>
        <span><b>${pd('L', 'w')}</b> gradient ${num(g, 5)}
          <span class="dim">(${s.mode === 'sgd' ? 'point #' + s.selectedIndex : 'averaged over ' + s.data.X.length + ' points'})</span></span>
        <span><b>Δw</b> the step ${num(-stepSize, 5)}</span>
      </div>
      <div class="note">The gradient here is ${g > 0 ? '<b>positive</b>, so raising this weight would raise the loss — the rule lowers it' : g < 0 ? '<b>negative</b>, so raising this weight would lower the loss — the rule raises it' : '<b>zero</b>, so this weight does not move at all'}.
      The size of the move is |η · ∂L/∂w| = ${fmt(Math.abs(stepSize), 5)}: gradient descent takes bigger steps where the surface is steeper.</div>`;
  }

  function trainingGradients(s) {
    return s.mode === 'sgd'
      ? batchGradients(s.net, s.data.X, s.data.Y, s.activation, [s.selectedIndex])
      : s.batchGrads;
  }

  // -------------------------------------------------------------------------
  // Loss surface
  // -------------------------------------------------------------------------

  /** Evaluate the mean loss with two parameters overridden. */
  function sliceLoss(net, activation, specA, specB, a, b, xs, ys) {
    setParam(net, specA, a);
    setParam(net, specB, b);
    return batchLossFlat(net, xs, ys, activation);
  }

  /** Flattened copy of the dataset, rebuilt only when the data itself changes. */
  const flat = { key: null, xs: null, ys: null };
  function flatData(s, stride) {
    const key = `${s.datasetId}|${s.dataSeed}|${s.noise}|${s.nPoints}|${stride}`;
    if (flat.key !== key) {
      Object.assign(flat, flattenDataset(s.data.X, s.data.Y, stride), { key });
    }
    return flat;
  }

  function renderSurface(s) {
    const specA = PARAM_BY_ID[s.surfaceX];
    const specB = PARAM_BY_ID[s.surfaceY];
    const a0 = getParam(s.net, specA);
    const b0 = getParam(s.net, specB);
    const r = parseFloat(refs.zoom.value);

    const plot = new Plot(refs.surfaceCanvas, {
      xDomain: [a0 - r, a0 + r],
      yDomain: [b0 - r, b0 + r],
      pad: { left: 46, right: 12, top: 12, bottom: 30 },
    });

    // Both axes get the same units-per-pixel. Without this the slice is
    // anisotropically stretched and the negative-gradient arrow would not look
    // perpendicular to the contours -- which it mathematically always is.
    const ry = r * (plot.plotH / Math.max(1, plot.plotW));

    // Keep the window still while the point stays comfortably inside it, so the
    // trail does not slide around underfoot on every step.
    if (
      !surface.win ||
      surface.win.r !== r ||
      Math.abs(a0 - surface.win.cx) > r * 0.55 ||
      Math.abs(b0 - surface.win.cy) > ry * 0.55
    ) {
      surface.win = { cx: a0, cy: b0, r };
      surface.key = null;
    }

    const win = surface.win;
    plot.setDomain([win.cx - r, win.cx + r], [win.cy - ry, win.cy + ry]);

    // --- the field itself, cached and throttled ----------------------------
    // Recomputing this is by far the most expensive thing the app does, so the
    // pixels are cached and only rebuilt when the slice actually changed (or
    // every 180 ms while training runs).
    const otherKey = PARAM_SPECS.filter((sp) => sp.id !== specA.id && sp.id !== specB.id)
      .map((sp) => getParam(s.net, sp).toFixed(4))
      .join(',');
    const key = `${specA.id}|${specB.id}|${win.cx.toFixed(3)},${win.cy.toFixed(3)},${r}|${s.activation}|${s.datasetId}|${otherKey}`;
    const now = performance.now();
    if (surface.key !== key && (!s.running || now - surface.stamp > 180)) {
      computeSurface(s, plot, specA, specB);
      surface.key = key;
      surface.stamp = now;
    }
    if (surface.canvas) {
      plot.ctx.drawImage(
        surface.canvas,
        plot.pad.left,
        plot.pad.top,
        plot.plotW,
        plot.plotH
      );
      plot.clipPlot();
      drawContours(plot);
      plot.restore();
    }

    // --- trajectory ---------------------------------------------------------
    const ia = PARAM_SPECS.indexOf(specA);
    const ib = PARAM_SPECS.indexOf(specB);
    const path = s.path.map((p) => [p[ia], p[ib]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));

    plot.clipPlot();
    if (path.length > 1) {
      plot.line(path, { color: 'rgba(255,255,255,0.9)', width: 2 });
      const stride = Math.max(1, Math.floor(path.length / 60));
      for (let i = 0; i < path.length; i += stride) {
        plot.dot(path[i][0], path[i][1], { r: 2.2, fill: 'rgba(255,255,255,0.7)' });
      }
    }

    // --- the next few steps, previewed at the current learning rate ---------
    const preview = previewPath(s, specA, specB, 6);
    if (preview.length > 1) {
      plot.line(preview, { color: '#fde68a', width: 2, dash: [5, 4] });
      for (const [x, y] of preview.slice(1)) {
        plot.dot(x, y, { r: 2.6, fill: 'rgba(253,230,138,0.95)' });
      }
    }

    // --- gradient arrows ----------------------------------------------------
    const grads = s.pendingGrads ?? trainingGradients(s);
    const ga = getGrad(grads, specA);
    const gb = getGrad(grads, specB);
    const gmag = Math.hypot(ga, gb);
    if (gmag > 1e-12) {
      // Dashed grey: the pure direction, at a fixed on-screen length.
      const unit = (r * 0.42) / gmag;
      plot.arrow(a0, b0, a0 - ga * unit, b0 - gb * unit, { color: 'rgba(255,255,255,0.55)', width: 1.5, head: 7 });
      // Solid green: the actual step this η will take.
      plot.arrow(a0, b0, a0 - ga * s.learningRate, b0 - gb * s.learningRate, { color: '#34d399', width: 2.6, head: 9 });
    }
    plot.restore();

    plot.dot(a0, b0, { r: 6, fill: '#fff', stroke: '#111827', width: 2.5 });
    plot.axes({ xLabel: specA.text, yLabel: specB.text, grid: false });
    plot.frame('#94a3b8');

    refs.surfaceNote.innerHTML = `
      <span><i class="sw white"></i> path so far (${s.path.length} recorded)</span>
      <span><i class="sw green"></i> next step at η = ${fmt(s.learningRate, 4)}</span>
      <span><i class="sw amber"></i> next 6 steps previewed</span>
      <span class="dim">loss on this slice: ${fmt(surface.lo, 3)} – ${fmt(surface.hi, 3)}</span>`;
  }

  function computeSurface(s, plot, specA, specB) {
    // Every grid point is a full pass over the dataset, so the grid is bounded:
    // on a wide screen the sample spacing grows rather than the cost. The field
    // is blitted back up with smoothing, so a coarser grid mostly costs contour
    // crispness, not legibility.
    const MAX_CELLS = 3000;
    let step = 6;
    while ((plot.plotW / step) * (plot.plotH / step) > MAX_CELLS) step += 1;
    const cols = Math.max(2, Math.ceil(plot.plotW / step));
    const rows = Math.max(2, Math.ceil(plot.plotH / step));

    // While idle, evaluate the slice on the whole dataset so that the contours
    // and the gradient arrow describe exactly the same function -- the arrow is
    // then provably perpendicular to the contour it sits on. While training runs
    // the grid is subsampled instead, to keep the recompute inside a frame.
    const stride = s.running && s.data.X.length > 120 ? 2 : 1;
    const { xs, ys } = flatData(s, stride);

    const probe = cloneNetwork(s.net);
    const vals = new Float64Array(cols * rows);
    let lo = Infinity;
    let hi = -Infinity;

    for (let rIdx = 0; rIdx < rows; rIdx++) {
      const by = plot.iy(plot.pad.top + rIdx * step + step / 2);
      for (let cIdx = 0; cIdx < cols; cIdx++) {
        const ax = plot.ix(plot.pad.left + cIdx * step + step / 2);
        const v = sliceLoss(probe, s.activation, specA, specB, ax, by, xs, ys);
        vals[rIdx * cols + cIdx] = v;
        if (Number.isFinite(v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    surface.lo = lo;
    surface.hi = hi;

    const img = new ImageData(cols, rows);
    const span = Math.max(1e-9, hi - lo);
    for (let i = 0; i < vals.length; i++) {
      // sqrt keeps the low-loss basin from washing out into a single flat colour
      const t = Math.sqrt(Math.max(0, Math.min(1, (vals[i] - lo) / span)));
      const [R, G, B] = lossColor(t);
      img.data[i * 4] = R;
      img.data[i * 4 + 1] = G;
      img.data[i * 4 + 2] = B;
      img.data[i * 4 + 3] = 255;
    }

    const off = document.createElement('canvas');
    off.width = cols;
    off.height = rows;
    off.getContext('2d').putImageData(img, 0, 0);
    surface.canvas = off;

    // Keep the raw values so real iso-loss contour lines can be traced over the
    // field, along with the data-space extent of the grid's sample centres.
    surface.vals = vals;
    surface.cols = cols;
    surface.rows = rows;
    surface.extent = {
      x0: plot.ix(plot.pad.left + step / 2),
      x1: plot.ix(plot.pad.left + (cols - 1) * step + step / 2),
      y0: plot.iy(plot.pad.top + step / 2),
      y1: plot.iy(plot.pad.top + (rows - 1) * step + step / 2),
    };
  }

  /** Trace a handful of iso-loss levels over the cached slice. */
  function drawContours(plot) {
    if (!surface.vals) return;
    const { vals, cols, rows, extent, lo, hi } = surface;
    const ctx = plot.ctx;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.30)';
    ctx.lineWidth = 1;
    for (let k = 1; k < 8; k++) {
      // Levels evenly spaced in sqrt(loss), matching the colour ramp, so the
      // lines stay usefully spread out inside the low-loss basin.
      const level = lo + Math.pow(k / 8, 2) * (hi - lo);
      const shifted = new Float64Array(vals.length);
      for (let i = 0; i < vals.length; i++) shifted[i] = vals[i] - level;
      const segs = marchingSquares(shifted, rows, cols, extent.x0, extent.x1, extent.y0, extent.y1);
      ctx.beginPath();
      for (const [[ax, ay], [bx, by]] of segs) {
        ctx.moveTo(plot.px(ax), plot.py(ay));
        ctx.lineTo(plot.px(bx), plot.py(by));
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  /** Simulate the next few updates without committing them. */
  function previewPath(s, specA, specB, n) {
    let net = cloneNetwork(s.net);
    const out = [[getParam(net, specA), getParam(net, specB)]];
    for (let i = 0; i < n; i++) {
      const g =
        s.mode === 'sgd'
          ? batchGradients(net, s.data.X, s.data.Y, s.activation, [s.selectedIndex])
          : batchGradients(net, s.data.X, s.data.Y, s.activation);
      net = gradientDescentStep(net, g, s.learningRate);
      const a = getParam(net, specA);
      const b = getParam(net, specB);
      if (!Number.isFinite(a) || !Number.isFinite(b)) break;
      out.push([a, b]);
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // Small companion plots
  // -------------------------------------------------------------------------

  function renderLossCurve(s) {
    const h = s.lossHistory.filter((p) => Number.isFinite(p.loss));
    const maxLoss = Math.max(0.1, ...h.map((p) => p.loss));
    const maxEpoch = Math.max(1, s.lossHistory[s.lossHistory.length - 1]?.epoch ?? 1);
    const plot = new Plot(refs.lossCanvas, {
      xDomain: [0, maxEpoch],
      yDomain: [0, maxLoss * 1.08],
      pad: { left: 44, right: 10, top: 8, bottom: 26 },
    });
    plot.axes({ xLabel: 'epoch', grid: true, yTicks: 3 });
    plot.clipPlot();
    plot.line(h.map((p) => [p.epoch, p.loss]), { color: COLORS.accent, width: 2 });
    plot.restore();
    plot.frame();
  }

  function renderBoundary(s) {
    const { X, Y } = s.data;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [a, b] of X) {
      x0 = Math.min(x0, a); x1 = Math.max(x1, a);
      y0 = Math.min(y0, b); y1 = Math.max(y1, b);
    }
    const dx = (x1 - x0) * 0.15;
    const dy = (y1 - y0) * 0.15;
    const plot = new Plot(refs.boundaryCanvas, {
      xDomain: [x0 - dx, x1 + dx],
      yDomain: [y0 - dy, y1 + dy],
      pad: { left: 30, right: 8, top: 8, bottom: 22 },
    });
    plot.field((px, py) => {
      const p = forward(s.net, [px, py], s.activation).yhat;
      const c = classColor(p).match(/[\d.]+/g);
      const conf = 0.14 + 0.86 * Math.abs(p - 0.5) * 2;
      return [
        Math.round(255 + (Number(c[0]) - 255) * conf * 0.8),
        Math.round(255 + (Number(c[1]) - 255) * conf * 0.8),
        Math.round(255 + (Number(c[2]) - 255) * conf * 0.8),
      ];
    }, { step: 4 });
    X.forEach(([x, y], i) => {
      plot.dot(x, y, {
        r: i === s.selectedIndex ? 5 : 2.2,
        fill: Y[i] === 1 ? COLORS.class1 : COLORS.class0,
        stroke: i === s.selectedIndex ? '#111827' : null,
        width: 2,
      });
    });
    plot.axes({ grid: false, xTicks: 3, yTicks: 3 });
    plot.frame();
  }

  // -------------------------------------------------------------------------
  // Parameter table
  // -------------------------------------------------------------------------

  function renderTable(s) {
    const grads = s.pendingGrads ?? trainingGradients(s);
    const maxAbs = Math.max(1e-12, ...PARAM_SPECS.map((sp) => Math.abs(getGrad(grads, sp))));
    const rows = PARAM_SPECS.map((spec) => {
      const w = getParam(s.net, spec);
      const g = getGrad(grads, spec);
      const d = -s.learningRate * g;
      const on = s.inspectedParam === spec.id;
      return `
        <tr class="${on ? 'on' : ''}" data-id="${spec.id}">
          <td class="pname">${spec.html}</td>
          <td class="pdesc">${spec.desc}</td>
          <td class="pv">${fmt(w, 5)}</td>
          <td class="pv">${num(g, 5)}</td>
          <td class="pbar"><i style="width:${(Math.abs(g) / maxAbs) * 100}%;background:${signedColor(g, maxAbs, 0.85)}"></i></td>
          <td class="pv">${num(d, 5)}</td>
          <td class="pv strong">${fmt(w + d, 5)}</td>
        </tr>`;
    }).join('');

    refs.table.innerHTML = `
      <table class="ptable">
        <thead>
          <tr>
            <th>parameter</th><th>meaning</th><th>w</th>
            <th>${pd('L', 'w')}</th><th>|gradient|</th>
            <th>Δw = −η·${pd('L', 'w')}</th><th>w after update</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
    for (const tr of refs.table.querySelectorAll('tbody tr')) {
      tr.addEventListener('click', () => store.set({ inspectedParam: tr.dataset.id }));
    }
  }

  // -------------------------------------------------------------------------

  function render() {
    const s = store.state;

    refs.lrOut.textContent = fmt(s.learningRate, 4);
    if (document.activeElement !== refs.lr) refs.lr.value = lrToPos(s.learningRate);
    refs.speedOut.textContent = String(s.speed);
    refs.mode.value = s.mode;
    refs.axisX.value = s.surfaceX;
    refs.axisY.value = s.surfaceY;
    refs.run.textContent = s.running ? '❚❚ Pause' : '▶ Run training';
    refs.run.classList.toggle('danger', s.running);

    const phaseNames = {
      idle: 'ready',
      forward: 'forward pass done — gradients not yet computed',
      backward: 'gradients computed — waiting to apply them',
      updated: 'weights updated',
    };
    refs.phase.innerHTML = `<span class="phase-dot ${s.phase}"></span> ${phaseNames[s.phase]}`;
    refs.update.classList.toggle('primary', s.phase === 'backward');
    refs.stepB.classList.toggle('primary', s.phase === 'forward');

    refs.stats.innerHTML = `
      epoch <b>${Math.round(s.epoch)}</b> ·
      updates <b>${s.updates}</b> ·
      loss <b>${fmt(s.loss, 5)}</b> ·
      accuracy <b>${(s.acc * 100).toFixed(1)}%</b>`;

    // learning-rate guidance
    const lr = s.learningRate;
    refs.lrHint.innerHTML =
      lr < 0.02
        ? '<span class="tone cold">Very small.</span> Each step barely moves. Correct direction, glacial progress — run a few hundred epochs and watch how little the loss falls.'
        : lr > 8
          ? '<span class="tone hot">Very large.</span> Expect the loss to climb rather than fall: each step flies past the minimum and lands somewhere worse.'
          : lr > 2.5
            ? '<span class="tone warm">Large.</span> Watch for the loss bouncing up and down instead of settling — the classic sign of overshooting.'
            : '<span class="tone good">In a sensible range</span> for this problem.';

    const d = store.convergenceDiagnosis();
    if (d.kind === 'unknown' || !d.title) {
      refs.diagnosis.className = 'diagnosis hidden';
      refs.diagnosis.innerHTML = '';
    } else {
      refs.diagnosis.className = `diagnosis ${d.kind}`;
      refs.diagnosis.innerHTML = `<b>${d.title}</b><span>${d.detail}</span>`;
    }

    renderRule(s);
    renderSurface(s);
    renderLossCurve(s);
    renderBoundary(s);
    renderTable(s);
  }

  return { render };
}
