// Tab 1 — Basics.
//
// The minimum mathematics needed to follow the rest of the app. Every idea gets
// a short explanation and a live example: nothing here is a static wall of text,
// and every derivative shown is checked on screen against a finite difference so
// the reader can see that the calculus is not being taken on faith.

import { el, $, fmt, num, frac, pd, COLORS, lossColor } from './dom.js';
import { Plot } from './plots.js';
import { ACTIVATIONS, ACTIVATION_ORDER, sigmoid } from '../math/activations.js';

const H = 1e-5; // step used for every on-screen finite-difference check

/** Central difference: [f(t+h) − f(t−h)] / 2h. */
function numericDeriv(f, t, h = H) {
  return (f(t + h) - f(t - h)) / (2 * h);
}

export function createBasicsTab(host, store) {
  const state = {
    // matrix playground
    W: [
      [0.5, -1.2],
      [1.0, 0.8],
      [-0.4, 0.3],
      [2.0, -0.5],
    ],
    x: [2, 3],
    // single neuron
    n: { w1: 1.4, w2: -0.9, b: 0.3, x1: 0.8, x2: 1.2, act: 'sigmoid' },
    // activation gallery cursor
    z: 0.8,
    // chain rule playground
    chain: { w: 1.2, x: 0.7, b: -0.3, y: 1 },
    partial: { x: 1.5, y: 2 },
    // loss curve
    yhat: 0.75,
    lossLabel: 1,
    // gradient field
    grad: { x: 1.4, y: 0.9 },
    gradTrail: [],
  };

  const refs = {};
  host.replaceChildren(build());
  wire();

  // -------------------------------------------------------------------------
  // Structure
  // -------------------------------------------------------------------------

  function build() {
    return el('div', { class: 'stack' }, [
      el('div', { class: 'card intro' }, [
        el('h2', { text: 'Start here' }),
        el('p', {
          html:
            'Six ideas carry the whole app: <b>matrix multiplication</b> moves numbers through a layer, ' +
            'a <b>neuron</b> is a weighted sum followed by an activation, an <b>activation derivative</b> ' +
            'is the factor backprop multiplies by, the <b>chain rule</b> is how a derivative travels ' +
            'backwards through composed steps, a <b>loss</b> is the number being minimised, and the ' +
            '<b>gradient</b> is the direction to step against. Everything below is live — drag a number ' +
            'and watch the arithmetic redo itself.',
        }),
      ]),

      section('1', 'Vectors, matrices, and matrix multiplication',
        'A layer of neurons is one matrix multiply. Each <em>row</em> of the weight matrix holds one neuron\'s incoming weights; multiplying by the input vector takes the dot product of every row with the input at once.',
        el('div', { class: 'matmul' }, [
          el('div', { class: 'matmul-inputs' }, [
            el('div', {}, [
              el('div', { class: 'mini-label', html: 'W &nbsp;<span class="dim">(4 × 2)</span> — one row per hidden neuron' }),
              refs.wGrid = el('div', { class: 'mgrid mgrid-4x2' }),
            ]),
            el('div', {}, [
              el('div', { class: 'mini-label', html: 'x &nbsp;<span class="dim">(2 × 1)</span>' }),
              refs.xGrid = el('div', { class: 'mgrid mgrid-2x1' }),
            ]),
          ]),
          refs.matmulWork = el('div', { class: 'work' }),
        ])
      ),

      section('2', 'What a neuron computes',
        'Every neuron does exactly two things: a weighted sum of its inputs plus a bias, then a non-linear squash. Without the second step a stack of layers collapses into a single linear map and could never bend a decision boundary.',
        el('div', { class: 'two-col' }, [
          el('div', {}, [
            refs.neuronControls = el('div', { class: 'sliders' }),
            refs.neuronWork = el('div', { class: 'work' }),
          ]),
          el('div', { class: 'plot-box' }, [
            el('div', { class: 'mini-label', html: 'the activation, with this neuron\'s operating point marked' }),
            refs.neuronCanvas = el('canvas', { class: 'canvas h200' }),
          ]),
        ])
      ),

      section('3', 'Activation functions and their derivatives',
        'Left: the function. Right: its derivative — <em>the exact factor backpropagation multiplies by</em> when the error signal passes back through a neuron. Where the derivative is near zero the neuron is saturated and learns almost nothing; that is the vanishing-gradient problem, visible directly.',
        el('div', {}, [
          el('div', { class: 'sliders inline' }, [
            refs.zSlider = slider('z', -4, 4, 0.01, state.z, 'z'),
            refs.zReadout = el('div', { class: 'readout' }),
          ]),
          refs.actGallery = el('div', { class: 'act-gallery' }),
        ])
      ),

      section('4', 'Partial derivatives and the chain rule',
        'A partial derivative asks: if I nudge <em>one</em> variable and freeze the rest, how fast does the output move? The chain rule then lets a derivative travel through a composition of steps by multiplying the derivative of each step. This is the entire engine of backpropagation.',
        el('div', { class: 'stack' }, [
          el('div', { class: 'subcard' }, [
            el('h4', { html: '4a. Partial derivatives — freeze everything but one variable' }),
            el('div', { class: 'two-col' }, [
              el('div', {}, [
                refs.partialControls = el('div', { class: 'sliders' }),
                refs.partialWork = el('div', { class: 'work' }),
              ]),
              el('div', { class: 'plot-box' }, [
                el('div', { class: 'mini-label', html: 'slices through f(x, y) — each partial is the slope of one slice' }),
                refs.partialCanvas = el('canvas', { class: 'canvas h200' }),
              ]),
            ]),
          ]),
          el('div', { class: 'subcard' }, [
            el('h4', { html: '4b. The chain rule on a 2-step composition' }),
            el('p', {
              class: 'tight',
              html:
                'This little composition is a whole neural network in miniature — a weighted sum, an activation, and a loss. ' +
                'Work out how the loss responds to <em>w</em> and you have already done backpropagation.',
            }),
            refs.chainControls = el('div', { class: 'sliders' }),
            refs.chainWork = el('div', { class: 'work' }),
          ]),
          el('div', { class: 'subcard' }, [
            el('h4', { html: '4c. The multivariable chain rule — sum over paths' }),
            el('p', {
              class: 'tight',
              html:
                'When one quantity influences the loss along <em>several</em> routes, the chain rule adds the routes up: ' +
                frac('∂L', '∂a') + ' = <span class="sum">Σ</span><sub>k</sub> ' + frac('∂L', '∂z<sub>k</sub>') +
                ' · ' + frac('∂z<sub>k</sub>', '∂a') + '. ' +
                'A hidden activation feeds every neuron in the next layer, so each of those neurons contributes one term. ' +
                'In this app the output layer has a single neuron, so the sum has exactly one term — but the machinery is the general one.',
            }),
          ]),
        ])
      ),

      section('5', 'Loss functions — the number being minimised',
        'A loss scores how wrong a prediction is. Training is nothing more than nudging the weights to make this number smaller. This app uses <b>binary cross-entropy</b>, which punishes confident mistakes brutally — its curve goes to infinity as the prediction approaches the wrong label.',
        el('div', { class: 'two-col' }, [
          el('div', {}, [
            refs.lossControls = el('div', { class: 'sliders' }),
            refs.lossWork = el('div', { class: 'work' }),
          ]),
          el('div', { class: 'plot-box' }, [
            el('div', { class: 'mini-label', html: 'loss as the prediction ŷ varies' }),
            refs.lossCanvas = el('canvas', { class: 'canvas h220' }),
          ]),
        ])
      ),

      section('6', 'The gradient — and why we walk the other way',
        'The gradient ∇f collects every partial derivative into one vector. It points in the direction of <b>steepest ascent</b>: of all the directions you could step, it is the one that increases f fastest. To <em>minimise</em>, step the opposite way — which is the whole of gradient descent.',
        el('div', { class: 'two-col wide-right' }, [
          el('div', {}, [
            refs.gradWork = el('div', { class: 'work' }),
            el('div', { class: 'row gap' }, [
              refs.gradStep = el('button', { class: 'btn', text: 'Take 12 descent steps' }),
              refs.gradClear = el('button', { class: 'btn ghost', text: 'Clear trail' }),
            ]),
            el('p', { class: 'hint', html: 'Click anywhere on the contour map to move the point.' }),
          ]),
          el('div', { class: 'plot-box' }, [
            el('div', { class: 'mini-label', html: 'f(x, y) = x² + 3y² — brighter is higher' }),
            refs.gradCanvas = el('canvas', { class: 'canvas h300 clickable' }),
          ]),
        ])
      ),
    ]);
  }

  function section(n, title, blurb, body) {
    return el('div', { class: 'card' }, [
      el('h3', { html: `<span class="secnum">${n}</span> ${title}` }),
      el('p', { class: 'blurb', html: blurb }),
      body,
    ]);
  }

  function slider(id, min, max, step, value, label) {
    const input = el('input', { type: 'range', min, max, step, value, id: `bs-${id}` });
    const out = el('span', { class: 'sval', text: fmt(value, 2) });
    const wrap = el('label', { class: 'slider' }, [
      el('span', { class: 'slabel', html: label }),
      input,
      out,
    ]);
    wrap.input = input;
    wrap.out = out;
    return wrap;
  }

  // -------------------------------------------------------------------------
  // Wiring
  // -------------------------------------------------------------------------

  function wire() {
    // --- 1. matrix grids ---------------------------------------------------
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 2; j++) {
        const inp = el('input', {
          type: 'number',
          step: '0.1',
          class: 'cell',
          value: state.W[i][j],
        });
        inp.addEventListener('input', () => {
          state.W[i][j] = parseFloat(inp.value) || 0;
          renderMatmul();
        });
        refs.wGrid.append(inp);
      }
    }
    for (let j = 0; j < 2; j++) {
      const inp = el('input', { type: 'number', step: '0.1', class: 'cell', value: state.x[j] });
      inp.addEventListener('input', () => {
        state.x[j] = parseFloat(inp.value) || 0;
        renderMatmul();
      });
      refs.xGrid.append(inp);
    }

    // --- 2. neuron sliders --------------------------------------------------
    const nDefs = [
      ['w1', -3, 3, 'w<sub>1</sub>'],
      ['w2', -3, 3, 'w<sub>2</sub>'],
      ['b', -3, 3, 'b'],
      ['x1', -3, 3, 'x<sub>1</sub>'],
      ['x2', -3, 3, 'x<sub>2</sub>'],
    ];
    for (const [key, lo, hi, label] of nDefs) {
      const s = slider(`n-${key}`, lo, hi, 0.01, state.n[key], label);
      s.input.addEventListener('input', () => {
        state.n[key] = parseFloat(s.input.value);
        s.out.textContent = fmt(state.n[key], 2);
        renderNeuron();
      });
      refs.neuronControls.append(s);
    }
    const actSel = el('select', { class: 'select' },
      ACTIVATION_ORDER.map((id) => el('option', { value: id, text: ACTIVATIONS[id].name, selected: id === state.n.act }))
    );
    actSel.addEventListener('change', () => {
      state.n.act = actSel.value;
      renderNeuron();
    });
    refs.neuronControls.append(el('label', { class: 'slider' }, [el('span', { class: 'slabel', text: 'f' }), actSel]));

    // --- 3. activation gallery ---------------------------------------------
    refs.actCards = ACTIVATION_ORDER.map((id) => {
      const act = ACTIVATIONS[id];
      const fC = el('canvas', { class: 'canvas h130' });
      const dC = el('canvas', { class: 'canvas h130' });
      const vals = el('div', { class: 'act-vals' });
      refs.actGallery.append(
        el('div', { class: 'act-card' }, [
          el('div', { class: 'act-title' }, [
            el('b', { text: act.name }),
            el('span', { class: 'act-blurb', text: act.blurb }),
          ]),
          el('div', { class: 'act-plots' }, [
            el('div', {}, [el('div', { class: 'mini-label mono', text: act.tex }), fC]),
            el('div', {}, [el('div', { class: 'mini-label mono deriv', text: act.dtex }), dC]),
          ]),
          vals,
        ])
      );
      return { id, act, fC, dC, vals };
    });
    refs.zSlider.input.addEventListener('input', () => {
      state.z = parseFloat(refs.zSlider.input.value);
      refs.zSlider.out.textContent = fmt(state.z, 2);
      renderActivations();
    });

    // --- 4. partial derivatives + chain rule --------------------------------
    for (const [key, label] of [['x', 'x'], ['y', 'y']]) {
      const s = slider(`p-${key}`, -3, 3, 0.01, state.partial[key], label);
      s.input.addEventListener('input', () => {
        state.partial[key] = parseFloat(s.input.value);
        s.out.textContent = fmt(state.partial[key], 2);
        renderPartial();
      });
      refs.partialControls.append(s);
    }

    for (const [key, lo, hi, label] of [
      ['w', -3, 3, 'w'],
      ['x', -3, 3, 'x'],
      ['b', -3, 3, 'b'],
    ]) {
      const s = slider(`c-${key}`, lo, hi, 0.01, state.chain[key], label);
      s.input.addEventListener('input', () => {
        state.chain[key] = parseFloat(s.input.value);
        s.out.textContent = fmt(state.chain[key], 2);
        renderChain();
      });
      refs.chainControls.append(s);
    }
    const ySel = el('select', { class: 'select' }, [
      el('option', { value: '1', text: 'y = 1' }),
      el('option', { value: '0', text: 'y = 0' }),
    ]);
    ySel.addEventListener('change', () => {
      state.chain.y = parseInt(ySel.value, 10);
      renderChain();
    });
    refs.chainControls.append(el('label', { class: 'slider' }, [el('span', { class: 'slabel', text: 'target' }), ySel]));

    // --- 5. loss ------------------------------------------------------------
    const ySlider = slider('yhat', 0.001, 0.999, 0.001, state.yhat, 'ŷ');
    ySlider.input.addEventListener('input', () => {
      state.yhat = parseFloat(ySlider.input.value);
      ySlider.out.textContent = fmt(state.yhat, 3);
      renderLoss();
    });
    refs.lossControls.append(ySlider);
    const lSel = el('select', { class: 'select' }, [
      el('option', { value: '1', text: 'true label y = 1' }),
      el('option', { value: '0', text: 'true label y = 0' }),
    ]);
    lSel.addEventListener('change', () => {
      state.lossLabel = parseInt(lSel.value, 10);
      renderLoss();
    });
    refs.lossControls.append(el('label', { class: 'slider' }, [el('span', { class: 'slabel', text: 'label' }), lSel]));

    // --- 6. gradient field --------------------------------------------------
    refs.gradCanvas.addEventListener('click', (e) => {
      if (!refs.gradPlot) return;
      const p = refs.gradPlot.eventToData(e);
      state.grad = { x: p.x, y: p.y };
      state.gradTrail = [];
      renderGradient();
    });
    refs.gradStep.addEventListener('click', () => {
      const eta = 0.12;
      for (let i = 0; i < 12; i++) {
        state.gradTrail.push([state.grad.x, state.grad.y]);
        const gx = 2 * state.grad.x;
        const gy = 6 * state.grad.y;
        state.grad = { x: state.grad.x - eta * gx, y: state.grad.y - eta * gy };
      }
      renderGradient();
    });
    refs.gradClear.addEventListener('click', () => {
      state.gradTrail = [];
      renderGradient();
    });
  }

  // -------------------------------------------------------------------------
  // Renderers
  // -------------------------------------------------------------------------

  function renderMatmul() {
    const { W, x } = state;
    const rows = W.map((row, i) => {
      const z = row[0] * x[0] + row[1] * x[1];
      return `
        <div class="wline">
          <span class="lhs">z<sub>${i + 1}</sub></span>
          <span class="eq">=</span>
          <span class="sym">W<sub>${i + 1},1</sub>·x<sub>1</sub> + W<sub>${i + 1},2</sub>·x<sub>2</sub></span>
          <span class="eq">=</span>
          <span class="sub">${fmt(row[0], 2)} × ${fmt(x[0], 2)} + ${fmt(row[1], 2)} × ${fmt(x[1], 2)}</span>
          <span class="eq">=</span>
          ${num(z, 3)}
        </div>`;
    });
    refs.matmulWork.innerHTML = `
      <div class="shape">(4 × 2) · (2 × 1) → (4 × 1)&nbsp; — the inner dimensions must match, and the result has one entry per neuron.</div>
      ${rows.join('')}
      <div class="note">Each line is a <b>dot product</b>: walk across a row of W and down the column of x, multiply pairwise, add it all up. Do that for all four rows at once and you have multiplied a matrix by a vector.</div>`;
  }

  function renderNeuron() {
    const { w1, w2, b, x1, x2, act: actId } = state.n;
    const act = ACTIVATIONS[actId];
    const z = w1 * x1 + w2 * x2 + b;
    const a = act.f(z);
    const d = act.derivDetail(z);

    refs.neuronWork.innerHTML = `
      <div class="wline">
        <span class="lhs">z</span><span class="eq">=</span>
        <span class="sym">w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + b</span>
        <span class="eq">=</span>
        <span class="sub">${fmt(w1, 2)}×${fmt(x1, 2)} + ${fmt(w2, 2)}×${fmt(x2, 2)} + ${fmt(b, 2)}</span>
        <span class="eq">=</span>${num(z, 4)}
      </div>
      <div class="wline">
        <span class="lhs">a</span><span class="eq">=</span>
        <span class="sym">f(z)</span><span class="eq">=</span>
        <span class="sub">${act.name}(${fmt(z, 4)})</span>
        <span class="eq">=</span>${num(a, 4)}
      </div>
      <div class="wline dim-line">
        <span class="lhs">f′(z)</span><span class="eq">=</span>
        <span class="sub">${d.substituted}</span>
        <span class="eq">=</span>${num(d.value, 4)}
        <span class="tagline">← the number backprop will multiply by</span>
      </div>
      <div class="note">The weighted sum is linear; <b>f</b> is what makes the neuron non-linear. Notice that pushing the weights up drives |z| large, which for sigmoid and tanh drives f′(z) toward zero — a saturated neuron stops learning.</div>`;

    const plot = new Plot(refs.neuronCanvas, {
      xDomain: act.plotDomain,
      yDomain: paddedRange(act, act.plotDomain),
      pad: { left: 40, right: 12, top: 10, bottom: 26 },
    });
    plot.axes({ xLabel: 'z', grid: true, zeroLines: true });
    plot.fn((t) => act.f(t), { color: COLORS.accent, width: 2.5 });
    const zc = Math.max(act.plotDomain[0], Math.min(act.plotDomain[1], z));
    plot.clipPlot();
    plot.line([[zc, plot.yDomain[0]], [zc, a]], { color: COLORS.muted, width: 1, dash: [3, 3] });
    plot.line([[plot.xDomain[0], a], [zc, a]], { color: COLORS.muted, width: 1, dash: [3, 3] });
    plot.restore();
    plot.dot(zc, a, { r: 5, fill: '#fff', stroke: COLORS.accent, width: 2.5 });
    plot.label(zc, a, `f(z) = ${fmt(a, 3)}`, { dx: 8, dy: -6, color: COLORS.ink, bg: 'rgba(255,255,255,0.85)' });
    plot.frame();
  }

  function renderActivations() {
    refs.zReadout.innerHTML =
      `the marker on all eight plots sits at <b>z = ${fmt(state.z, 2)}</b> — ` +
      `drag it into the flat tails and watch every derivative collapse toward zero`;
    for (const card of refs.actCards) {
      const { act, fC, dC, vals } = card;
      const dom = act.plotDomain;
      const fv = act.f(state.z);
      const dv = act.df(state.z, fv);

      const p1 = new Plot(fC, {
        xDomain: dom,
        yDomain: paddedRange(act, dom),
        pad: { left: 34, right: 8, top: 8, bottom: 22 },
      });
      p1.axes({ grid: true, zeroLines: true, xTicks: 4, yTicks: 3 });
      p1.fn((t) => act.f(t), { color: COLORS.accent, width: 2.2 });
      markCursor(p1, state.z, fv);
      p1.frame();

      const dDom = derivRange(act, dom);
      const p2 = new Plot(dC, {
        xDomain: dom,
        yDomain: dDom,
        pad: { left: 34, right: 8, top: 8, bottom: 22 },
      });
      p2.axes({ grid: true, zeroLines: true, xTicks: 4, yTicks: 3 });
      // Piecewise-constant derivatives must not be drawn as a ramp across the
      // kink, so sample densely and let the near-vertical segment be clipped.
      p2.fn((t) => act.df(t, act.f(t)), { color: '#b45309', width: 2.2, samples: 600 });
      markCursor(p2, state.z, dv, '#b45309');
      p2.frame();

      const check = numericDeriv((t) => act.f(t), state.z);
      const agrees = Math.abs(check - dv) < 1e-4;
      vals.innerHTML = `
        <span>f(z) = ${num(fv, 4)}</span>
        <span>f′(z) = ${num(dv, 4)}</span>
        <span class="check ${agrees ? 'ok' : 'warn'}">
          finite difference ${fmt(check, 4)} ${agrees ? '✓ matches' : '≈ (kink at z = 0)'}
        </span>`;
    }
  }

  function markCursor(plot, z, v, color = COLORS.accent) {
    const zc = Math.max(plot.xDomain[0], Math.min(plot.xDomain[1], z));
    plot.clipPlot();
    plot.line([[zc, plot.yDomain[0]], [zc, plot.yDomain[1]]], { color: '#9aa3b2', width: 1, dash: [3, 3] });
    plot.restore();
    if (v >= plot.yDomain[0] && v <= plot.yDomain[1]) {
      plot.dot(zc, v, { r: 4, fill: '#fff', stroke: color, width: 2 });
    }
  }

  function renderPartial() {
    const { x, y } = state.partial;
    // f(x, y) = x²y + y³
    const f = (a, b) => a * a * b + b * b * b;
    const dfdx = 2 * x * y;
    const dfdy = x * x + 3 * y * y;
    const nx = numericDeriv((t) => f(t, y), x);
    const ny = numericDeriv((t) => f(x, t), y);

    refs.partialWork.innerHTML = `
      <div class="wline"><span class="lhs">f(x, y)</span><span class="eq">=</span>
        <span class="sym">x²y + y³</span><span class="eq">=</span>
        <span class="sub">(${fmt(x, 2)})²·${fmt(y, 2)} + (${fmt(y, 2)})³</span>
        <span class="eq">=</span>${num(f(x, y), 4)}</div>
      <div class="wline"><span class="lhs">${pd('f', 'x')}</span><span class="eq">=</span>
        <span class="sym">2xy</span>
        <span class="tagline">treat y as a constant</span>
        <span class="eq">=</span><span class="sub">2 × ${fmt(x, 2)} × ${fmt(y, 2)}</span>
        <span class="eq">=</span>${num(dfdx, 4)}</div>
      <div class="wline"><span class="lhs">${pd('f', 'y')}</span><span class="eq">=</span>
        <span class="sym">x² + 3y²</span>
        <span class="tagline">treat x as a constant</span>
        <span class="eq">=</span><span class="sub">${fmt(x * x, 3)} + 3 × ${fmt(y * y, 3)}</span>
        <span class="eq">=</span>${num(dfdy, 4)}</div>
      <div class="verify">
        <b>Checked numerically:</b> nudging x alone gives ${fmt(nx, 4)}
        (analytic ${fmt(dfdx, 4)}) · nudging y alone gives ${fmt(ny, 4)}
        (analytic ${fmt(dfdy, 4)}).
      </div>`;

    const plot = new Plot(refs.partialCanvas, {
      xDomain: [-3, 3],
      yDomain: [-30, 30],
      pad: { left: 44, right: 10, top: 10, bottom: 26 },
    });
    plot.axes({ xLabel: 'the varied variable', grid: true, zeroLines: true });
    plot.fn((t) => f(t, y), { color: '#2563eb', width: 2.2 });
    plot.fn((t) => f(x, t), { color: '#db2777', width: 2.2 });
    plot.clipPlot();
    // Tangent lines: slope = the partial derivative at the current point.
    tangent(plot, x, f(x, y), dfdx, '#2563eb');
    tangent(plot, y, f(x, y), dfdy, '#db2777');
    plot.restore();
    plot.dot(x, f(x, y), { r: 4.5, fill: '#fff', stroke: '#2563eb', width: 2 });
    plot.dot(y, f(x, y), { r: 4.5, fill: '#fff', stroke: '#db2777', width: 2 });
    plot.label(-2.9, 27, 'blue: f(·, y) — slope is ∂f/∂x', { color: '#2563eb' });
    plot.label(-2.9, 23, 'pink: f(x, ·) — slope is ∂f/∂y', { color: '#db2777' });
    plot.frame();
  }

  function tangent(plot, t0, v0, slope, color) {
    const d = 0.9;
    plot.line([[t0 - d, v0 - slope * d], [t0 + d, v0 + slope * d]], {
      color,
      width: 1.5,
      dash: [5, 3],
      alpha: 0.9,
    });
  }

  function renderChain() {
    const { w, x, b, y } = state.chain;
    const z = w * x + b;
    const a = sigmoid(z);
    const L = (a - y) ** 2;

    const dL_da = 2 * (a - y);
    const da_dz = a * (1 - a);
    const dz_dw = x;
    const chainProduct = dL_da * da_dz * dz_dw;

    const numeric = numericDeriv((t) => {
      const zz = t * x + b;
      const aa = sigmoid(zz);
      return (aa - y) ** 2;
    }, w);
    const agrees = Math.abs(numeric - chainProduct) < 1e-5;

    refs.chainWork.innerHTML = `
      <div class="chain-defs">
        <span>z = wx + b = ${num(z, 4)}</span>
        <span>a = σ(z) = ${num(a, 4)}</span>
        <span>L = (a − y)² = ${num(L, 4)}</span>
      </div>
      <div class="chain-goal">Goal: ${pd('L', 'w')} — how does the loss move when only <b>w</b> moves?</div>
      <table class="chain-table">
        <thead><tr><th>step</th><th>symbolic</th><th>with numbers</th><th>value</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>${pd('L', 'a')} = 2(a − y)</td>
              <td>2(${fmt(a, 4)} − ${y})</td><td>${num(dL_da, 4)}</td></tr>
          <tr><td>2</td><td>${pd('a', 'z')} = σ(z)(1 − σ(z))</td>
              <td>${fmt(a, 4)} × ${fmt(1 - a, 4)}</td><td>${num(da_dz, 4)}</td></tr>
          <tr><td>3</td><td>${pd('z', 'w')} = x</td>
              <td>${fmt(x, 4)}</td><td>${num(dz_dw, 4)}</td></tr>
          <tr class="total"><td>×</td>
              <td>${pd('L', 'w')} = ${pd('L', 'a')}·${pd('a', 'z')}·${pd('z', 'w')}</td>
              <td>${fmt(dL_da, 4)} × ${fmt(da_dz, 4)} × ${fmt(dz_dw, 4)}</td>
              <td>${num(chainProduct, 4)}</td></tr>
        </tbody>
      </table>
      <div class="verify ${agrees ? '' : 'bad'}">
        <b>Checked numerically:</b> [L(w + h) − L(w − h)] / 2h = ${fmt(numeric, 6)}
        versus ${fmt(chainProduct, 6)} from the chain rule ${agrees ? '✓' : '✗'}
        <span class="dim">(h = ${H})</span>
      </div>
      <div class="note">Read the product right to left and you are watching a signal travel backwards: the loss's
      sensitivity to the activation, then through the activation's own slope, then out onto the weight.
      That is precisely what the Backpropagation tab does — with more factors, because there are more layers.</div>`;
  }

  function renderLoss() {
    const y = state.lossLabel;
    const p = state.yhat;
    const bce = -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    const mse = (p - y) ** 2;
    const dBce = (p - y) / (p * (1 - p));

    refs.lossWork.innerHTML = `
      <div class="wline"><span class="lhs">L<sub>BCE</sub></span><span class="eq">=</span>
        <span class="sym">−[ y·log ŷ + (1 − y)·log(1 − ŷ) ]</span><span class="eq">=</span>
        <span class="sub">−[ ${y}·log(${fmt(p, 3)}) + ${1 - y}·log(${fmt(1 - p, 3)}) ]</span>
        <span class="eq">=</span>${num(bce, 4)}</div>
      <div class="wline dim-line"><span class="lhs">L<sub>MSE</sub></span><span class="eq">=</span>
        <span class="sym">(ŷ − y)²</span><span class="eq">=</span>
        <span class="sub">(${fmt(p, 3)} − ${y})²</span><span class="eq">=</span>${num(mse, 4)}
        <span class="tagline">for comparison</span></div>
      <div class="wline"><span class="lhs">${pd('L', 'ŷ')}</span><span class="eq">=</span>
        <span class="sym">(ŷ − y) / (ŷ(1 − ŷ))</span><span class="eq">=</span>
        <span class="sub">(${fmt(p, 3)} − ${y}) / (${fmt(p, 3)} × ${fmt(1 - p, 3)})</span>
        <span class="eq">=</span>${num(dBce, 4)}
        <span class="tagline">← step 1 of backprop</span></div>
      <div class="note">The sign of ${pd('L', 'ŷ')} tells you which way is wrong: negative means "the prediction is
      too low, raising it lowers the loss". Cross-entropy blows up near a confident mistake, so those examples
      dominate the gradient — exactly the ones worth learning from.</div>`;

    const plot = new Plot(refs.lossCanvas, {
      xDomain: [0, 1],
      yDomain: [0, 5],
      pad: { left: 40, right: 12, top: 12, bottom: 30 },
    });
    plot.axes({ xLabel: 'prediction  ŷ', yLabel: 'loss', grid: true });
    plot.fn((t) => -Math.log(Math.max(1e-9, t)), { color: COLORS.class1, width: 2.4 });
    plot.fn((t) => -Math.log(Math.max(1e-9, 1 - t)), { color: COLORS.class0, width: 2.4 });
    plot.fn((t) => (t - 1) ** 2, { color: COLORS.class1, width: 1.4, dash: [4, 3] });
    plot.fn((t) => t ** 2, { color: COLORS.class0, width: 1.4, dash: [4, 3] });
    plot.clipPlot();
    plot.line([[p, 0], [p, Math.min(5, bce)]], { color: COLORS.muted, width: 1, dash: [3, 3] });
    plot.restore();
    plot.dot(p, Math.min(5, bce), { r: 5, fill: '#fff', stroke: y === 1 ? COLORS.class1 : COLORS.class0, width: 2.5 });
    plot.label(0.03, 4.7, 'solid: cross-entropy   dashed: squared error', { color: COLORS.muted });
    plot.label(0.03, 4.35, 'amber: y = 1   teal: y = 0', { color: COLORS.muted });
    plot.frame();
  }

  function renderGradient() {
    const f = (a, b) => a * a + 3 * b * b;
    const { x, y } = state.grad;
    const gx = 2 * x;
    const gy = 6 * y;
    const mag = Math.hypot(gx, gy);

    refs.gradWork.innerHTML = `
      <div class="wline"><span class="lhs">∇f</span><span class="eq">=</span>
        <span class="sym">[ ${pd('f', 'x')}, ${pd('f', 'y')} ] = [ 2x, 6y ]</span></div>
      <div class="wline"><span class="lhs">at (${fmt(x, 2)}, ${fmt(y, 2)})</span><span class="eq">=</span>
        <span class="sub">[ 2×${fmt(x, 2)}, 6×${fmt(y, 2)} ]</span><span class="eq">=</span>
        [ ${num(gx, 3)}, ${num(gy, 3)} ]</div>
      <div class="wline"><span class="lhs">‖∇f‖</span><span class="eq">=</span>
        <span class="sub">√(${fmt(gx, 2)}² + ${fmt(gy, 2)}²)</span><span class="eq">=</span>${num(mag, 3)}
        <span class="tagline">how steep it is here</span></div>
      <div class="wline"><span class="lhs">f</span><span class="eq">=</span>${num(f(x, y), 4)}</div>
      <div class="note">The red arrow is <b>+∇f</b> — steepest ascent. The green arrow is <b>−∇f</b>, the direction
      gradient descent actually steps. Note the arrow is not aimed at the minimum: it is only the best
      <em>local</em> direction, which is why descent curves rather than going straight there. Notice also that
      because f grows three times faster in y than in x, the arrow leans hard toward the y-axis.</div>`;

    const plot = new Plot(refs.gradCanvas, {
      xDomain: [-3, 3],
      yDomain: [-2.2, 2.2],
      pad: { left: 40, right: 12, top: 12, bottom: 28 },
    });
    refs.gradPlot = plot;

    const maxF = f(3, 2.2);
    plot.field((a, b) => lossColor(Math.sqrt(f(a, b) / maxF)), { step: 3 });

    // Contour rings, drawn as level sets of the ellipse.
    plot.clipPlot();
    for (const level of [0.5, 2, 5, 10, 18, 28]) {
      const pts = [];
      for (let t = 0; t <= 64; t++) {
        const th = (t / 64) * Math.PI * 2;
        pts.push([Math.sqrt(level) * Math.cos(th), Math.sqrt(level / 3) * Math.sin(th)]);
      }
      plot.line(pts, { color: 'rgba(255,255,255,0.28)', width: 1 });
    }
    if (state.gradTrail.length) {
      plot.line([...state.gradTrail, [x, y]], { color: '#ffffff', width: 2 });
      for (const [tx, ty] of state.gradTrail) {
        plot.dot(tx, ty, { r: 2.5, fill: 'rgba(255,255,255,0.8)' });
      }
    }
    plot.restore();

    const s = 0.28; // arrow scale, so the arrows stay readable
    plot.arrow(x, y, x + gx * s, y + gy * s, { color: '#ff5a5f', width: 2.5 });
    plot.arrow(x, y, x - gx * s, y - gy * s, { color: '#34d399', width: 2.5 });
    plot.dot(x, y, { r: 5, fill: '#fff', stroke: '#111827', width: 2 });
    plot.label(x + gx * s, y + gy * s, '+∇f  ascent', {
      color: '#b91c1c', dx: 6, dy: -4, bg: 'rgba(255,255,255,0.86)',
    });
    plot.label(x - gx * s, y - gy * s, '−∇f  descent', {
      color: '#047857', dx: 6, dy: 16, bg: 'rgba(255,255,255,0.86)',
    });
    plot.axes({ xLabel: 'x', yLabel: 'y', grid: false });
    plot.frame('#94a3b8');
  }

  function paddedRange(act, dom) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i <= 120; i++) {
      const t = dom[0] + ((dom[1] - dom[0]) * i) / 120;
      const v = act.f(t);
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const pad = (hi - lo) * 0.12 || 0.2;
    return [lo - pad, hi + pad];
  }

  function derivRange(act, dom) {
    let hi = -Infinity;
    let lo = Infinity;
    for (let i = 0; i <= 120; i++) {
      const t = dom[0] + ((dom[1] - dom[0]) * i) / 120;
      const v = act.df(t, act.f(t));
      hi = Math.max(hi, v);
      lo = Math.min(lo, v);
    }
    const pad = Math.max(0.08, (hi - lo) * 0.18);
    return [Math.min(0, lo) - pad, hi + pad];
  }

  // -------------------------------------------------------------------------

  function render() {
    renderMatmul();
    renderNeuron();
    renderActivations();
    renderPartial();
    renderChain();
    renderLoss();
    renderGradient();
  }

  return { render };
}
