// Tab 2 — Forward Pass.
//
// Steps a single data point through the network one stage at a time, showing
// the actual arithmetic at each stage next to the diagram it is happening in.
// Also carries the two whole-dataset views: the decision boundary and the
// loss-vs-epoch curve.

import { el, fmt, num, COLORS, classColor } from './dom.js';
import { Plot, marchingSquares } from './plots.js';
import { NetworkDiagram } from './netdiagram.js';
import { getActivation } from '../math/activations.js';
import { forward, bceLoss } from '../math/network.js';

const STAGES = [
  { id: 0, name: 'Inputs', short: 'x' },
  { id: 1, name: 'Hidden pre-activations', short: 'z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾' },
  { id: 2, name: 'Hidden activations', short: 'a⁽¹⁾ = f(z⁽¹⁾)' },
  { id: 3, name: 'Output pre-activation', short: 'z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾' },
  { id: 4, name: 'Prediction', short: 'ŷ = σ(z⁽²⁾)' },
  { id: 5, name: 'Loss', short: 'L = BCE(ŷ, y)' },
];

export function createForwardTab(host, store) {
  const refs = {};
  let animating = null;

  host.replaceChildren(build());
  wire();

  function build() {
    return el('div', { class: 'stack' }, [
      el('div', { class: 'card' }, [
        el('div', { class: 'card-head' }, [
          el('h3', { text: 'Trace one point through the network' }),
          el('div', { class: 'row gap' }, [
            refs.pointPrev = el('button', { class: 'btn ghost sm', text: '‹ point' }),
            refs.pointLabel = el('span', { class: 'pill' }),
            refs.pointNext = el('button', { class: 'btn ghost sm', text: 'point ›' }),
          ]),
        ]),
        refs.stageBar = el('div', { class: 'stagebar' }),
        el('div', { class: 'row gap stage-controls' }, [
          refs.back = el('button', { class: 'btn ghost', text: '← Previous stage' }),
          refs.next = el('button', { class: 'btn primary', text: 'Next stage →' }),
          refs.animate = el('button', { class: 'btn', text: '▶ Animate the whole pass' }),
          refs.reset = el('button', { class: 'btn ghost', text: 'Back to start' }),
        ]),
        refs.diagram = el('div', { class: 'diagram-host' }),
        el('div', { class: 'legend' }, [
          el('span', {}, [el('i', { class: 'sw pos' }), ' positive weight']),
          el('span', {}, [el('i', { class: 'sw neg' }), ' negative weight']),
          el('span', { text: 'thickness = |weight|' }),
          el('span', { class: 'dim', text: 'hover or click an edge to inspect it in the Backpropagation tab' }),
        ]),
      ]),

      el('div', { class: 'card' }, [
        el('h3', { text: 'The arithmetic at this stage' }),
        refs.stageMath = el('div', { class: 'work' }),
      ]),

      el('div', { class: 'two-col equal' }, [
        el('div', { class: 'card' }, [
          el('div', { class: 'card-head' }, [
            el('h3', { text: 'Decision boundary' }),
            refs.accBadge = el('span', { class: 'pill' }),
          ]),
          el('p', { class: 'blurb tight', html: 'The network\'s output over the whole input plane. Click any point to trace it.' }),
          refs.boundaryCanvas = el('canvas', { class: 'canvas h320 clickable' }),
          el('div', { class: 'legend' }, [
            el('span', {}, [el('i', { class: 'sw c0' }), ' class 0']),
            el('span', {}, [el('i', { class: 'sw c1' }), ' class 1']),
            el('span', { class: 'dim', text: 'white line = ŷ = 0.5' }),
          ]),
        ]),
        el('div', { class: 'card' }, [
          el('div', { class: 'card-head' }, [
            el('h3', { text: 'Loss vs epoch' }),
            refs.lossBadge = el('span', { class: 'pill' }),
          ]),
          el('p', { class: 'blurb tight', html: 'Mean cross-entropy over all points, recorded after every epoch. Train on the Gradient Descent tab and watch it fall.' }),
          refs.lossCanvas = el('canvas', { class: 'canvas h320' }),
          refs.lossNote = el('div', { class: 'legend' }),
        ]),
      ]),
    ]);
  }

  function wire() {
    refs.net = new NetworkDiagram(refs.diagram, {
      onSelectParam: (id) => store.set({ inspectedParam: id }),
      onHoverParam: () => render(),
    });

    STAGES.forEach((s) => {
      const b = el('button', { class: 'stage', dataset: { stage: s.id } }, [
        el('span', { class: 'stage-n', text: String(s.id) }),
        el('span', { class: 'stage-name', text: s.name }),
        el('span', { class: 'stage-eq mono', text: s.short }),
      ]);
      b.addEventListener('click', () => setStage(s.id));
      refs.stageBar.append(b);
    });

    refs.next.addEventListener('click', () => setStage(Math.min(5, store.state.forwardStage + 1)));
    refs.back.addEventListener('click', () => setStage(Math.max(0, store.state.forwardStage - 1)));
    refs.reset.addEventListener('click', () => setStage(0));
    refs.animate.addEventListener('click', animate);
    refs.pointPrev.addEventListener('click', () => store.selectPoint(store.state.selectedIndex - 1));
    refs.pointNext.addEventListener('click', () => store.selectPoint(store.state.selectedIndex + 1));

    refs.boundaryCanvas.addEventListener('click', (e) => {
      if (!refs.boundaryPlot) return;
      const p = refs.boundaryPlot.eventToData(e);
      const { X } = store.state.data;
      let best = 0;
      let bestD = Infinity;
      X.forEach(([a, b], i) => {
        const d = (a - p.x) ** 2 + (b - p.y) ** 2;
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      });
      store.selectPoint(best);
    });
  }

  function setStage(stage) {
    stopAnimation();
    store.set({ forwardStage: stage });
  }

  function stopAnimation() {
    if (animating) {
      clearInterval(animating);
      animating = null;
      refs.animate.textContent = '▶ Animate the whole pass';
    }
  }

  function animate() {
    if (animating) {
      stopAnimation();
      return;
    }
    store.set({ forwardStage: 0 });
    refs.animate.textContent = '■ Stop';
    animating = setInterval(() => {
      const s = store.state.forwardStage;
      if (s >= 5) {
        stopAnimation();
        return;
      }
      store.set({ forwardStage: s + 1 });
    }, 850);
  }

  // -------------------------------------------------------------------------

  function render() {
    const s = store.state;
    const { X, Y } = s.data;
    const i = s.selectedIndex;
    const cache = s.cache;
    const loss = bceLoss(cache.yhat, Y[i]);

    refs.pointLabel.innerHTML = `point <b>#${i}</b> &nbsp; x = (${fmt(X[i][0], 2)}, ${fmt(X[i][1], 2)}) &nbsp; y = <b>${Y[i]}</b>`;

    for (const b of refs.stageBar.children) {
      const n = Number(b.dataset.stage);
      b.classList.toggle('on', n === s.forwardStage);
      b.classList.toggle('done', n < s.forwardStage);
    }

    refs.net.render({
      net: s.net,
      cache,
      grads: s.pointGrads,
      encode: 'weights',
      stage: s.forwardStage,
      selected: s.inspectedParam,
      showValues: true,
      target: Y[i],
      loss,
      label: `activation: ${getActivation(s.activation).name}`,
    });

    renderStageMath(s, cache, Y[i], loss);
    renderBoundary(s);
    renderLossCurve(s);
  }

  function renderStageMath(s, cache, y, loss) {
    const act = getActivation(s.activation);
    const net = s.net;
    const stage = s.forwardStage;
    let html = '';

    if (stage === 0) {
      html = `
        <div class="stage-title">Stage 0 — the input vector</div>
        <div class="wline"><span class="lhs">x</span><span class="eq">=</span>
          <span class="sub">[ ${fmt(cache.x[0], 4)}, ${fmt(cache.x[1], 4)} ]</span>
          <span class="tagline">two features, standardized to zero mean and unit variance</span></div>
        <div class="wline"><span class="lhs">y</span><span class="eq">=</span>${num(y, 0)}
          <span class="tagline">the true label we are trying to predict</span></div>`;
    } else if (stage === 1) {
      const rows = net.W1.map((row, j) => `
        <div class="wline">
          <span class="lhs">z⁽¹⁾<sub>${j + 1}</sub></span><span class="eq">=</span>
          <span class="sym">W⁽¹⁾<sub>${j + 1},1</sub>x<sub>1</sub> + W⁽¹⁾<sub>${j + 1},2</sub>x<sub>2</sub> + b⁽¹⁾<sub>${j + 1}</sub></span>
          <span class="eq">=</span>
          <span class="sub">${fmt(row[0], 3)}×${fmt(cache.x[0], 3)} + ${fmt(row[1], 3)}×${fmt(cache.x[1], 3)} + ${fmt(net.b1[j], 3)}</span>
          <span class="eq">=</span>${num(cache.z1[j], 4)}
        </div>`).join('');
      html = `
        <div class="stage-title">Stage 1 — weighted sums into the hidden layer</div>
        <div class="shape">z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾ &nbsp;&nbsp; (4 × 2)·(2 × 1) + (4 × 1) → (4 × 1)</div>
        ${rows}
        <div class="note">Four independent dot products, one per hidden neuron. Nothing non-linear has happened yet.</div>`;
    } else if (stage === 2) {
      const rows = cache.z1.map((z, j) => {
        const d = act.derivDetail(z);
        return `
        <div class="wline">
          <span class="lhs">a⁽¹⁾<sub>${j + 1}</sub></span><span class="eq">=</span>
          <span class="sym">f(z⁽¹⁾<sub>${j + 1}</sub>)</span><span class="eq">=</span>
          <span class="sub">${act.name}(${fmt(z, 4)})</span><span class="eq">=</span>${num(cache.a1[j], 4)}
          <span class="tagline">f′ = ${fmt(d.value, 4)}</span>
        </div>`;
      }).join('');
      html = `
        <div class="stage-title">Stage 2 — the non-linearity</div>
        <div class="shape">a⁽¹⁾ = f(z⁽¹⁾) applied elementwise &nbsp;&nbsp; f = ${act.name}</div>
        ${rows}
        <div class="note">The trailing f′ on each line is stored away for later: backpropagation multiplies by exactly
        that number when the error signal comes back through this neuron. A neuron whose f′ is near zero will
        barely be updated at all.</div>`;
    } else if (stage === 3) {
      const terms = net.W2.map((w, j) => `${fmt(w, 3)}×${fmt(cache.a1[j], 3)}`).join(' + ');
      html = `
        <div class="stage-title">Stage 3 — weighted sum into the output neuron</div>
        <div class="shape">z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾ &nbsp;&nbsp; (1 × 4)·(4 × 1) + scalar → scalar</div>
        <div class="wline"><span class="lhs">z⁽²⁾</span><span class="eq">=</span>
          <span class="sub">${terms} + ${fmt(net.b2, 3)}</span>
          <span class="eq">=</span>${num(cache.z2, 4)}</div>
        <div class="note">One dot product collapses the four hidden activations into a single score.</div>`;
    } else if (stage === 4) {
      html = `
        <div class="stage-title">Stage 4 — squash the score into a probability</div>
        <div class="wline"><span class="lhs">ŷ</span><span class="eq">=</span>
          <span class="sym">σ(z⁽²⁾) = 1 / (1 + e<sup>−z⁽²⁾</sup>)</span><span class="eq">=</span>
          <span class="sub">1 / (1 + e<sup>−${fmt(cache.z2, 4)}</sup>)</span>
          <span class="eq">=</span>${num(cache.yhat, 4)}</div>
        <div class="wline dim-line"><span class="lhs">σ′(z⁽²⁾)</span><span class="eq">=</span>
          <span class="sub">${fmt(cache.yhat, 4)} × ${fmt(1 - cache.yhat, 4)}</span>
          <span class="eq">=</span>${num(cache.yhat * (1 - cache.yhat), 4)}
          <span class="tagline">← step 2 of the backward pass</span></div>
        <div class="note">The output activation is always the sigmoid here, regardless of the hidden-layer choice:
        it turns an unbounded score into a number in (0, 1) that can be read as P(y = 1).</div>`;
    } else {
      const correct = (cache.yhat >= 0.5 ? 1 : 0) === y;
      html = `
        <div class="stage-title">Stage 5 — score the prediction</div>
        <div class="wline"><span class="lhs">L</span><span class="eq">=</span>
          <span class="sym">−[ y·log ŷ + (1 − y)·log(1 − ŷ) ]</span><span class="eq">=</span>
          <span class="sub">−[ ${y}·log(${fmt(cache.yhat, 4)}) + ${1 - y}·log(${fmt(1 - cache.yhat, 4)}) ]</span>
          <span class="eq">=</span>${num(loss, 4)}</div>
        <div class="wline"><span class="lhs">prediction</span><span class="eq">=</span>
          <span class="sub">${fmt(cache.yhat, 4)} ${cache.yhat >= 0.5 ? '≥' : '<'} 0.5 → class ${cache.yhat >= 0.5 ? 1 : 0}</span>
          <span class="badge ${correct ? 'ok' : 'bad'}">${correct ? 'correct' : 'wrong'}</span></div>
        <div class="note">This single number is what the entire backward pass differentiates. Head to the
        <b>Backpropagation</b> tab to take it apart term by term.</div>`;
    }

    refs.stageMath.innerHTML = html;
  }

  function renderBoundary(s) {
    const { X, Y } = s.data;
    const b = boundsOf(X);
    const plot = new Plot(refs.boundaryCanvas, {
      xDomain: [b.x0, b.x1],
      yDomain: [b.y0, b.y1],
      pad: { left: 38, right: 12, top: 12, bottom: 28 },
    });
    refs.boundaryPlot = plot;

    plot.field((px, py) => {
      const p = forward(s.net, [px, py], s.activation).yhat;
      const c = classColor(p);
      const m = c.match(/[\d.]+/g);
      // Fade toward white near the boundary so the 0.5 contour reads clearly.
      const conf = Math.abs(p - 0.5) * 2;
      const t = 0.14 + 0.86 * conf;
      return [
        Math.round(255 + (Number(m[0]) - 255) * t * 0.8),
        Math.round(255 + (Number(m[1]) - 255) * t * 0.8),
        Math.round(255 + (Number(m[2]) - 255) * t * 0.8),
      ];
    }, { step: 3 });

    // The ŷ = 0.5 contour, traced column by column.
    plot.clipPlot();
    drawHalfContour(plot, s);
    plot.restore();

    X.forEach(([x, y], i) => {
      const sel = i === s.selectedIndex;
      plot.dot(x, y, {
        r: sel ? 6.5 : 3,
        fill: Y[i] === 1 ? COLORS.class1 : COLORS.class0,
        stroke: sel ? '#111827' : 'rgba(255,255,255,0.75)',
        width: sel ? 2.5 : 1,
      });
    });

    plot.axes({ xLabel: 'x₁', yLabel: 'x₂', grid: false });
    plot.frame();

    refs.accBadge.innerHTML = `accuracy <b>${(s.acc * 100).toFixed(1)}%</b>`;
  }

  /** The ŷ = 0.5 level set, traced with marching squares. */
  function drawHalfContour(plot, s) {
    const cols = 100;
    const rows = 100;
    const [x0, x1] = plot.xDomain;
    const [y0, y1] = plot.yDomain;

    const values = new Float64Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const yy = y0 + ((y1 - y0) * r) / (rows - 1);
      for (let c = 0; c < cols; c++) {
        const xx = x0 + ((x1 - x0) * c) / (cols - 1);
        values[r * cols + c] = forward(s.net, [xx, yy], s.activation).yhat - 0.5;
      }
    }

    const segs = marchingSquares(values, rows, cols, x0, x1, y0, y1);
    const ctx = plot.ctx;
    ctx.save();
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    // A dark halo under the white line keeps it readable over pale regions.
    for (const [color, width] of [['rgba(30,36,48,0.35)', 4], ['rgba(255,255,255,0.98)', 2]]) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (const [[ax, ay], [bx, by]] of segs) {
        ctx.moveTo(plot.px(ax), plot.py(ay));
        ctx.lineTo(plot.px(bx), plot.py(by));
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  function renderLossCurve(s) {
    const h = s.lossHistory;
    const finite = h.filter((p) => Number.isFinite(p.loss));
    const maxLoss = Math.max(0.1, ...finite.map((p) => p.loss));
    const maxEpoch = Math.max(1, h[h.length - 1]?.epoch ?? 1);

    const plot = new Plot(refs.lossCanvas, {
      xDomain: [0, maxEpoch],
      yDomain: [0, maxLoss * 1.08],
      pad: { left: 46, right: 12, top: 12, bottom: 30 },
    });
    plot.axes({ xLabel: 'epoch', yLabel: 'mean loss', grid: true });
    plot.clipPlot();
    plot.line(finite.map((p) => [p.epoch, p.loss]), { color: COLORS.accent, width: 2 });
    plot.line(h.map((p) => [p.epoch, p.acc * maxLoss]), { color: COLORS.class0, width: 1.4, dash: [4, 3] });
    plot.restore();
    if (finite.length) {
      const last = finite[finite.length - 1];
      plot.dot(last.epoch, last.loss, { r: 4, fill: '#fff', stroke: COLORS.accent, width: 2 });
    }
    plot.frame();

    refs.lossBadge.innerHTML = `epoch <b>${Math.round(s.epoch)}</b> · loss <b>${fmt(s.loss, 4)}</b>`;
    refs.lossNote.innerHTML = `
      <span><i class="sw acc"></i> mean cross-entropy</span>
      <span><i class="sw c0"></i> accuracy (dashed, 0–100% rescaled to the axis)</span>`;
  }

  function boundsOf(X, margin = 0.18) {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const [a, b] of X) {
      x0 = Math.min(x0, a);
      x1 = Math.max(x1, a);
      y0 = Math.min(y0, b);
      y1 = Math.max(y1, b);
    }
    const dx = (x1 - x0) * margin;
    const dy = (y1 - y0) * margin;
    return { x0: x0 - dx, x1: x1 + dx, y0: y0 - dy, y1: y1 + dy };
  }

  return { render, stop: stopAnimation };
}
