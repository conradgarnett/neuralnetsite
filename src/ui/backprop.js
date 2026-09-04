// Tab 3 — Backpropagation.
//
// The centre of gravity of the app. The full chain rule for the selected data
// point is broken into twelve steps, and every step shows the same three things
// side by side:
//
//     the symbolic expression | the same thing with numbers | the result
//
// Nothing is skipped or folded away: the output delta, the push back through
// W(2), the activation derivative, the hidden delta and both weight gradients
// each get their own step, and any single parameter can be expanded into the
// complete unabbreviated product the chain rule produces — then checked against
// a finite difference on screen.

import { el, fmt, num, frac, pd, COLORS, signedColor } from './dom.js';
import { NetworkDiagram } from './netdiagram.js';
import { getActivation } from '../math/activations.js';
import {
  PARAM_SPECS,
  PARAM_BY_ID,
  getGrad,
  numericalGradient,
  batchGradients,
} from '../math/network.js';

export function createBackpropTab(host, store) {
  const refs = {};
  let hoverParam = null;
  let rankSource = 'point'; // 'point' | 'batch'
  let encode = 'gradients';

  host.replaceChildren(build());
  wire();

  function build() {
    return el('div', { class: 'stack' }, [
      el('div', { class: 'card' }, [
        el('div', { class: 'card-head' }, [
          el('h3', { text: 'The chain rule, one factor at a time' }),
          el('div', { class: 'row gap' }, [
            refs.pointPrev = el('button', { class: 'btn ghost sm', text: '‹ point' }),
            refs.pointLabel = el('span', { class: 'pill' }),
            refs.pointNext = el('button', { class: 'btn ghost sm', text: 'point ›' }),
          ]),
        ]),
        el('p', {
          class: 'blurb tight',
          html:
            'Backpropagation answers one question for all 17 parameters at once: <b>if this number changed a little, ' +
            'how would the loss change?</b> It gets there by walking backwards from the loss, multiplying by one ' +
            'local derivative per step. Step through it below.',
        }),
        refs.stepNav = el('div', { class: 'stepnav' }),
        el('div', { class: 'row gap stage-controls' }, [
          refs.first = el('button', { class: 'btn ghost', text: '⟲ First' }),
          refs.back = el('button', { class: 'btn ghost', text: '← Previous' }),
          refs.counter = el('span', { class: 'counter' }),
          refs.next = el('button', { class: 'btn primary', text: 'Next →' }),
          refs.last = el('button', { class: 'btn ghost', text: 'Last ⟳' }),
        ]),
        refs.stepBody = el('div', { class: 'stepbody' }),
      ]),

      el('div', { class: 'two-col wide-left' }, [
        el('div', { class: 'card' }, [
          el('div', { class: 'card-head' }, [
            el('h3', { text: 'Where this step is happening' }),
            refs.encodeToggle = el('div', { class: 'seg' }),
          ]),
          refs.diagram = el('div', { class: 'diagram-host' }),
          refs.diagramNote = el('div', { class: 'legend' }),
        ]),
        el('div', { class: 'card' }, [
          el('div', { class: 'card-head' }, [
            el('h3', { text: 'Largest gradients' }),
            refs.rankToggle = el('div', { class: 'seg' }),
          ]),
          el('p', { class: 'blurb tight', html: 'The parameters that will move furthest on the next update. Click one to expand its derivation.' }),
          refs.ranking = el('div', { class: 'ranking' }),
        ]),
      ]),

      el('div', { class: 'card' }, [
        el('div', { class: 'card-head' }, [
          refs.inspectTitle = el('h3', { text: 'Full derivation' }),
          refs.checkBtn = el('button', { class: 'btn sm', text: 'Verify against a finite difference' }),
        ]),
        el('p', {
          class: 'blurb tight',
          html: 'Every factor the chain rule produces for one parameter, unabbreviated. Hover or click any edge in the diagram — or any row in the ranking — to change which one.',
        }),
        refs.inspect = el('div', { class: 'work' }),
        refs.checkOut = el('div', { class: 'verify hidden' }),
      ]),
    ]);
  }

  function wire() {
    refs.net = new NetworkDiagram(refs.diagram, {
      onSelectParam: (id) => store.set({ inspectedParam: id }),
      onHoverParam: (id) => {
        hoverParam = id;
        render();
      },
    });

    refs.next.addEventListener('click', () => go(store.state.backpropStep + 1));
    refs.back.addEventListener('click', () => go(store.state.backpropStep - 1));
    refs.first.addEventListener('click', () => go(0));
    refs.last.addEventListener('click', () => go(999));
    refs.pointPrev.addEventListener('click', () => store.selectPoint(store.state.selectedIndex - 1));
    refs.pointNext.addEventListener('click', () => store.selectPoint(store.state.selectedIndex + 1));

    for (const [key, label] of [['weights', 'weights'], ['gradients', 'gradients']]) {
      const b = el('button', { class: 'segbtn', text: label });
      b.addEventListener('click', () => {
        encode = key;
        render();
      });
      refs.encodeToggle.append(b);
      b.dataset.key = key;
    }
    for (const [key, label] of [['point', 'this point'], ['batch', 'full batch']]) {
      const b = el('button', { class: 'segbtn', text: label });
      b.addEventListener('click', () => {
        rankSource = key;
        render();
      });
      refs.rankToggle.append(b);
      b.dataset.key = key;
    }

    refs.checkBtn.addEventListener('click', runCheck);

    document.addEventListener('keydown', (e) => {
      if (!host.offsetParent) return; // only when this tab is visible
      if (e.target.matches('input, select, textarea')) return;
      if (e.key === 'ArrowRight') go(store.state.backpropStep + 1);
      if (e.key === 'ArrowLeft') go(store.state.backpropStep - 1);
    });
  }

  function go(i) {
    const steps = buildSteps(store.state);
    store.set({ backpropStep: Math.max(0, Math.min(steps.length - 1, i)) });
  }

  // -------------------------------------------------------------------------
  // The twelve steps
  // -------------------------------------------------------------------------

  function buildSteps(s) {
    const act = getActivation(s.activation);
    const net = s.net;
    const c = s.cache;
    const g = s.pointGrads;
    const y = s.data.Y[s.selectedIndex];
    const D2 = 'δ<sup>(2)</sup>';
    const D1 = (j) => `δ<sup>(1)</sup><sub>${j + 1}</sub>`;

    const steps = [];

    steps.push({
      title: 'The loss, and its derivative with respect to the prediction',
      target: pd('L', 'ŷ'),
      stage: 4,
      rows: [
        {
          sym: `L = −[ y·log ŷ + (1 − y)·log(1 − ŷ) ]`,
          sub: `−[ ${y}·log(${fmt(c.yhat, 4)}) + ${1 - y}·log(${fmt(1 - c.yhat, 4)}) ]`,
          val: g.loss,
        },
        {
          sym: `${pd('L', 'ŷ')} = (ŷ − y) / (ŷ(1 − ŷ))`,
          sub: `(${fmt(c.yhat, 4)} − ${y}) / (${fmt(c.yhat, 4)} × ${fmt(1 - c.yhat, 4)})`,
          val: g.dL_dyhat,
        },
      ],
      note:
        'Everything downstream is built on this single number. Its sign says which way the prediction is wrong: ' +
        'negative means the prediction is too <em>low</em>, so increasing ŷ would lower the loss.',
    });

    steps.push({
      title: 'Through the output activation',
      target: pd('ŷ', 'z⁽²⁾'),
      stage: 4,
      rows: [
        {
          sym: `${pd('ŷ', 'z⁽²⁾')} = σ′(z⁽²⁾) = σ(z⁽²⁾)(1 − σ(z⁽²⁾)) = ŷ(1 − ŷ)`,
          sub: `${fmt(c.yhat, 4)} × ${fmt(1 - c.yhat, 4)}`,
          val: g.dyhat_dz2,
        },
      ],
      note:
        'The output neuron is always a sigmoid here, so this factor never changes when you switch the hidden activation. ' +
        'It is largest at ŷ = 0.5 (value 0.25) and collapses toward zero as the network becomes confident.',
    });

    steps.push({
      title: `Multiply them: the output error signal ${D2}`,
      target: pd('L', 'z⁽²⁾'),
      stage: 4,
      rows: [
        {
          sym: `${D2} = ${pd('L', 'z⁽²⁾')} = ${pd('L', 'ŷ')} · ${pd('ŷ', 'z⁽²⁾')}`,
          sub: `${fmt(g.dL_dyhat, 4)} × ${fmt(g.dyhat_dz2, 4)}`,
          val: g.dL_dyhat * g.dyhat_dz2,
        },
        {
          sym: `&nbsp;&nbsp;= (ŷ − y)/(ŷ(1 − ŷ)) · ŷ(1 − ŷ) = ŷ − y`,
          sub: `${fmt(c.yhat, 4)} − ${y}`,
          val: g.dz2,
          highlight: true,
        },
      ],
      note:
        'The ŷ(1 − ŷ) factors cancel exactly. That cancellation is the reason cross-entropy is paired with a sigmoid ' +
        'output: the error signal reduces to the raw residual <b>ŷ − y</b>, which never vanishes when the network is ' +
        'confidently wrong. Squared error would leave the ŷ(1 − ŷ) factor in place and stall on exactly those examples.',
    });

    steps.push({
      title: 'Gradients of the output weights',
      target: pd('L', 'W⁽²⁾'),
      stage: 3,
      rows: net.W2.map((_, j) => ({
        sym: `${pd('L', `W⁽²⁾<sub>1,${j + 1}</sub>`)} = ${D2} · ${pd('z⁽²⁾', `W⁽²⁾<sub>1,${j + 1}</sub>`)} = ${D2} · a⁽¹⁾<sub>${j + 1}</sub>`,
        sub: `${fmt(g.dz2, 4)} × ${fmt(c.a1[j], 4)}`,
        val: g.dW2[j],
      })),
      note:
        'Since z⁽²⁾ = Σ W⁽²⁾<sub>1,j</sub>·a⁽¹⁾<sub>j</sub> + b⁽²⁾, the derivative of z⁽²⁾ with respect to one weight is ' +
        'just the activation that weight multiplies. So a weight is blamed in proportion to how active its neuron was — ' +
        'a neuron that output zero gets no gradient at all.',
    });

    steps.push({
      title: 'Gradient of the output bias',
      target: pd('L', 'b⁽²⁾'),
      stage: 4,
      rows: [
        {
          sym: `${pd('L', 'b⁽²⁾')} = ${D2} · ${pd('z⁽²⁾', 'b⁽²⁾')} = ${D2} · 1`,
          sub: `${fmt(g.dz2, 4)} × 1`,
          val: g.db2,
        },
      ],
      note: 'The bias enters z⁽²⁾ with a coefficient of 1, so its gradient <em>is</em> the error signal.',
    });

    steps.push({
      title: 'Push the signal back onto the hidden activations',
      target: pd('L', 'a⁽¹⁾'),
      stage: 3,
      rows: net.W2.map((w, j) => ({
        sym: `${pd('L', `a⁽¹⁾<sub>${j + 1}</sub>`)} = ${D2} · ${pd('z⁽²⁾', `a⁽¹⁾<sub>${j + 1}</sub>`)} = ${D2} · W⁽²⁾<sub>1,${j + 1}</sub>`,
        sub: `${fmt(g.dz2, 4)} × ${fmt(w, 4)}`,
        val: g.da1[j],
      })),
      note:
        'This is the step that gives backpropagation its name: the <em>same</em> weights that carried activations forward ' +
        'now carry the error backwards. In general this is the multivariable chain rule, ' +
        `${pd('L', 'a⁽¹⁾<sub>j</sub>')} = Σ<sub>k</sub> ${pd('L', 'z⁽²⁾<sub>k</sub>')}·${pd('z⁽²⁾<sub>k</sub>', 'a⁽¹⁾<sub>j</sub>')} ` +
        '— one term per neuron the activation feeds. This network has a single output neuron, so each sum has exactly one term.',
    });

    steps.push({
      title: `Through the hidden activation's derivative — f = ${act.name}`,
      target: pd('a⁽¹⁾', 'z⁽¹⁾'),
      stage: 2,
      rows: c.z1.map((z, j) => {
        const d = act.derivDetail(z);
        return {
          sym: `${pd(`a⁽¹⁾<sub>${j + 1}</sub>`, `z⁽¹⁾<sub>${j + 1}</sub>`)} = ${d.symbolic}`,
          sub: d.substituted,
          val: g.fPrime1[j],
        };
      }),
      note:
        `<b>This is the only factor that changes when you switch the activation function.</b> Change the dropdown ` +
        `in the header and watch this column — and everything below it — recompute. Any neuron whose f′ is near zero ` +
        `is a bottleneck: it will pass almost nothing back to its own weights.`,
    });

    steps.push({
      title: 'Multiply: the hidden error signals',
      target: pd('L', 'z⁽¹⁾'),
      stage: 2,
      rows: g.dz1.map((d, j) => ({
        sym: `${D1(j)} = ${pd('L', `z⁽¹⁾<sub>${j + 1}</sub>`)} = ${pd('L', `a⁽¹⁾<sub>${j + 1}</sub>`)} · ${pd(`a⁽¹⁾<sub>${j + 1}</sub>`, `z⁽¹⁾<sub>${j + 1}</sub>`)}`,
        sub: `${fmt(g.da1[j], 4)} × ${fmt(g.fPrime1[j], 4)}`,
        val: d,
      })),
      note:
        'Each hidden neuron now has its own error signal, exactly analogous to δ⁽²⁾. From here the last step is ' +
        'identical in form to the output layer.',
    });

    steps.push({
      title: 'Gradients of the hidden weights',
      target: pd('L', 'W⁽¹⁾'),
      stage: 1,
      rows: (() => {
        const out = [];
        for (let j = 0; j < 4; j++) {
          for (let k = 0; k < 2; k++) {
            out.push({
              sym: `${pd('L', `W⁽¹⁾<sub>${j + 1},${k + 1}</sub>`)} = ${D1(j)} · x<sub>${k + 1}</sub>`,
              sub: `${fmt(g.dz1[j], 4)} × ${fmt(c.x[k], 4)}`,
              val: g.dW1[j][k],
            });
          }
        }
        return out;
      })(),
      note:
        'Same shape as the output layer: error signal times the input that the weight multiplied. ' +
        'Notice the consequence — an input feature that happens to be near zero for this example produces a near-zero ' +
        'gradient for every weight attached to it.',
    });

    steps.push({
      title: 'Gradients of the hidden biases',
      target: pd('L', 'b⁽¹⁾'),
      stage: 1,
      rows: g.db1.map((d, j) => ({
        sym: `${pd('L', `b⁽¹⁾<sub>${j + 1}</sub>`)} = ${D1(j)} · 1`,
        sub: `${fmt(g.dz1[j], 4)} × 1`,
        val: d,
      })),
      note: 'All 17 gradients are now known, from one forward pass and one backward pass.',
    });

    const spec = PARAM_BY_ID[inspectedId()];
    steps.push({
      title: `The whole product at once — ${spec.text}`,
      target: pd('L', spec.html),
      stage: spec.layer === 1 ? 1 : 3,
      rawHtml: derivationHtml(spec, s, { compact: true }),
      note:
        'Every factor above, collected into the single product the chain rule actually produces. Nothing is folded ' +
        'away or approximated — this is the number that the update rule will use.',
    });

    steps.push({
      title: 'From one example to the whole dataset',
      target: pd('L', 'θ'),
      stage: null,
      rows: [
        {
          sym: `${pd('L', 'θ')} = (1/N) Σ<sub>n</sub> ${pd('L⁽ⁿ⁾', 'θ')}`,
          sub: `average of ${s.data.X.length} per-example gradients`,
          val: null,
        },
        ...PARAM_SPECS.slice(0, 4).map((sp) => ({
          sym: pd('L', sp.html),
          sub: `this point ${fmt(getGrad(s.pointGrads, sp), 5)} &nbsp;→&nbsp; averaged over ${s.data.X.length} points`,
          val: getGrad(s.batchGrads, sp),
        })),
      ],
      note:
        'Everything above was the gradient for <b>one</b> point. Full-batch gradient descent runs that computation for ' +
        'every point and averages, which is what the Gradient Descent tab does by default. Single-point (SGD) mode ' +
        'skips the averaging and updates from this one example — noisier per step, but far cheaper.',
    });

    return steps;
  }

  function inspectedId() {
    const id = hoverParam ?? store.state.inspectedParam;
    return PARAM_BY_ID[id] ? id : 'W1_0_0';
  }

  // -------------------------------------------------------------------------
  // The unabbreviated derivation for a single parameter
  // -------------------------------------------------------------------------

  function derivationHtml(spec, s, { compact = false } = {}) {
    const act = getActivation(s.activation);
    const c = s.cache;
    const g = s.pointGrads;
    const net = s.net;
    const y = s.data.Y[s.selectedIndex];
    const j = spec.i;
    const k = spec.j;

    // Build the factor list: symbolic name, what it equals, and its value.
    const factors = [
      { sym: pd('L', 'ŷ'), law: '(ŷ − y)/(ŷ(1 − ŷ))', val: g.dL_dyhat },
      { sym: pd('ŷ', 'z⁽²⁾'), law: 'ŷ(1 − ŷ)', val: g.dyhat_dz2 },
    ];

    if (spec.kind === 'W2') {
      factors.push({ sym: pd('z⁽²⁾', spec.html), law: `a⁽¹⁾<sub>${j + 1}</sub>`, val: c.a1[j] });
    } else if (spec.kind === 'b2') {
      factors.push({ sym: pd('z⁽²⁾', 'b⁽²⁾'), law: '1', val: 1 });
    } else {
      factors.push({
        sym: pd('z⁽²⁾', `a⁽¹⁾<sub>${j + 1}</sub>`),
        law: `W⁽²⁾<sub>1,${j + 1}</sub>`,
        val: net.W2[j],
      });
      factors.push({
        sym: pd(`a⁽¹⁾<sub>${j + 1}</sub>`, `z⁽¹⁾<sub>${j + 1}</sub>`),
        law: act.derivDetail(c.z1[j]).symbolic.replace(/^[^=]*=\s*/, ''),
        val: g.fPrime1[j],
      });
      if (spec.kind === 'W1') {
        factors.push({ sym: pd(`z⁽¹⁾<sub>${j + 1}</sub>`, spec.html), law: `x<sub>${k + 1}</sub>`, val: c.x[k] });
      } else {
        factors.push({ sym: pd(`z⁽¹⁾<sub>${j + 1}</sub>`, `b⁽¹⁾<sub>${j + 1}</sub>`), law: '1', val: 1 });
      }
    }

    const total = factors.reduce((p, f) => p * f.val, 1);
    const analytic = getGrad(g, spec);

    // A rule like "1 − tanh²(z)" needs brackets to read correctly as one factor
    // in a product; a bare symbol like "x₁" does not.
    const bracket = (law) => (/[+−]/.test(law.replace(/e[+−]\d/gi, '')) && !/^\(.*\)$/.test(law) ? `(${law})` : law);

    const symLine = factors.map((f) => f.sym).join(' <span class="op">·</span> ');
    const lawLine = factors.map((f) => `<span class="lawf">${bracket(f.law)}</span>`).join(' <span class="op">·</span> ');
    const numLine = factors.map((f) => `<span class="factor">${fmt(f.val, 4)}</span>`).join(' <span class="op">×</span> ');

    const table = factors.map((f, i) => `
      <tr>
        <td class="ix">${i + 1}</td>
        <td class="sym">${f.sym}</td>
        <td class="law">${bracket(f.law)}</td>
        <td class="val">${num(f.val, 5)}</td>
        <td class="run">${num(factors.slice(0, i + 1).reduce((p, q) => p * q.val, 1), 5)}</td>
      </tr>`).join('');

    return `
      <div class="deriv">
        <div class="deriv-head">
          <span class="deriv-target">${pd('L', spec.html)}</span>
          <span class="deriv-desc">${spec.desc}</span>
          <span class="deriv-cur">current value ${num(paramValue(net, spec), 5)}</span>
        </div>
        <div class="deriv-line"><span class="dlbl">chain</span><span class="dbody">${symLine}</span></div>
        <div class="deriv-line"><span class="dlbl">rules</span><span class="dbody">${lawLine}</span></div>
        <div class="deriv-line"><span class="dlbl">numbers</span><span class="dbody">${numLine} <span class="op">=</span> ${num(total, 6)}</span></div>
        ${compact ? '' : `
        <table class="factor-table">
          <thead><tr><th>#</th><th>factor</th><th>evaluates to</th><th>value</th><th>running product</th></tr></thead>
          <tbody>${table}</tbody>
        </table>`}
        <div class="deriv-foot">
          backprop computed ${num(analytic, 6)}
          ${Math.abs(analytic - total) < 1e-9
            ? '<span class="badge ok">the expanded product matches</span>'
            : `<span class="badge bad">mismatch of ${fmt(analytic - total, 8)}</span>`}
        </div>
      </div>`;
  }

  function paramValue(net, spec) {
    switch (spec.kind) {
      case 'W1': return net.W1[spec.i][spec.j];
      case 'b1': return net.b1[spec.i];
      case 'W2': return net.W2[spec.i];
      default: return net.b2;
    }
  }

  function runCheck() {
    const s = store.state;
    const spec = PARAM_BY_ID[inspectedId()];
    const i = s.selectedIndex;
    const X = [s.data.X[i]];
    const Y = [s.data.Y[i]];
    const h = 1e-5;
    const numericV = numericalGradient(s.net, X, Y, s.activation, spec, h);
    const analytic = getGrad(s.pointGrads, spec);
    const absErr = Math.abs(numericV - analytic);
    const relErr = absErr / Math.max(1e-12, Math.abs(numericV) + Math.abs(analytic));
    const ok = absErr < 1e-7 || relErr < 1e-6;

    refs.checkOut.className = `verify ${ok ? '' : 'bad'}`;
    refs.checkOut.innerHTML = `
      <b>Finite-difference check for ${spec.text}</b> — perturb the parameter by h = ${h} and re-run the forward pass:
      <div class="check-grid">
        <span>[ L(θ + h) − L(θ − h) ] / 2h</span><span>${fmt(numericV, 8)}</span>
        <span>backpropagation</span><span>${fmt(analytic, 8)}</span>
        <span>absolute difference</span><span>${absErr.toExponential(2)}</span>
        <span>relative difference</span><span>${relErr.toExponential(2)}</span>
      </div>
      <span class="badge ${ok ? 'ok' : 'bad'}">${ok ? 'agrees — backprop is correct here' : 'differs (expected near a ReLU kink, where the derivative does not exist)'}</span>`;
  }

  // -------------------------------------------------------------------------

  function render() {
    const s = store.state;
    const steps = buildSteps(s);
    const idx = Math.max(0, Math.min(steps.length - 1, s.backpropStep));
    const step = steps[idx];
    const i = s.selectedIndex;

    refs.pointLabel.innerHTML =
      `point <b>#${i}</b> &nbsp; x = (${fmt(s.data.X[i][0], 2)}, ${fmt(s.data.X[i][1], 2)}) &nbsp; y = <b>${s.data.Y[i]}</b> &nbsp; ŷ = <b>${fmt(s.cache.yhat, 4)}</b>`;
    refs.counter.innerHTML = `Step <b>${idx + 1}</b> of ${steps.length}`;
    refs.back.disabled = idx === 0;
    refs.next.disabled = idx === steps.length - 1;

    // step navigator chips
    if (refs.stepNav.children.length !== steps.length) {
      refs.stepNav.replaceChildren(
        ...steps.map((st, n) => {
          const b = el('button', { class: 'chip', title: stripTags(st.title) }, [
            el('b', { text: String(n + 1) }),
            el('span', { html: st.target }),
          ]);
          b.addEventListener('click', () => go(n));
          return b;
        })
      );
    }
    Array.from(refs.stepNav.children).forEach((b, n) => {
      b.classList.toggle('on', n === idx);
      b.classList.toggle('done', n < idx);
    });

    // step body
    const rowsHtml = step.rawHtml
      ? step.rawHtml
      : `<table class="chain-table wide">
           <thead><tr><th>symbolic</th><th>with the current numbers</th><th>value</th></tr></thead>
           <tbody>${step.rows.map((r) => `
             <tr class="${r.highlight ? 'total' : ''}">
               <td class="sym">${r.sym}</td>
               <td class="sub">${r.sub}</td>
               <td class="val">${r.val === null ? '' : num(r.val, 5)}</td>
             </tr>`).join('')}</tbody>
         </table>`;

    refs.stepBody.innerHTML = `
      <div class="step-head">
        <span class="step-target">${step.target}</span>
        <span class="step-title">${step.title}</span>
      </div>
      ${rowsHtml}
      <div class="note">${step.note}</div>`;

    // diagram
    const gradsForDiagram = rankSource === 'batch' ? s.batchGrads : s.pointGrads;
    refs.net.render({
      net: s.net,
      cache: s.cache,
      grads: gradsForDiagram,
      encode,
      stage: step.stage,
      selected: inspectedId(),
      showValues: true,
      target: s.data.Y[i],
      loss: s.pointGrads.loss,
      label: encode === 'gradients' ? 'thickness = |∂L/∂w|' : 'thickness = |w|',
    });
    for (const b of refs.encodeToggle.children) b.classList.toggle('on', b.dataset.key === encode);
    for (const b of refs.rankToggle.children) b.classList.toggle('on', b.dataset.key === rankSource);
    refs.diagramNote.innerHTML = encode === 'gradients'
      ? '<span><i class="sw pos"></i> gradient &gt; 0 (this weight should <b>decrease</b>)</span><span><i class="sw neg"></i> gradient &lt; 0 (should <b>increase</b>)</span><span class="dim">highlighted edges are the ones this step is about</span>'
      : '<span><i class="sw pos"></i> positive weight</span><span><i class="sw neg"></i> negative weight</span><span class="dim">switch to "gradients" to see what backprop just computed</span>';

    // ranking
    const src = rankSource === 'batch' ? s.batchGrads : s.pointGrads;
    const ranked = PARAM_SPECS
      .map((sp) => ({ spec: sp, g: getGrad(src, sp) }))
      .sort((a, b) => Math.abs(b.g) - Math.abs(a.g));
    const maxAbs = Math.max(1e-12, Math.abs(ranked[0].g));
    refs.ranking.replaceChildren(
      ...ranked.map((r, n) => {
        const row = el('button', { class: `rank-row${inspectedId() === r.spec.id ? ' on' : ''}${n < 3 ? ' top' : ''}` }, [
          el('span', { class: 'rank-n', text: String(n + 1) }),
          el('span', { class: 'rank-name', html: r.spec.html }),
          el('span', { class: 'rank-bar' }, [
            el('i', {
              style: {
                width: `${(Math.abs(r.g) / maxAbs) * 100}%`,
                background: signedColor(r.g, maxAbs, 0.9),
              },
            }),
          ]),
          el('span', { class: `rank-val ${r.g > 0 ? 'pos' : r.g < 0 ? 'neg' : 'zero'}`, text: fmt(r.g, 5) }),
        ]);
        row.addEventListener('click', () => store.set({ inspectedParam: r.spec.id }));
        return row;
      })
    );

    // inspector
    const spec = PARAM_BY_ID[inspectedId()];
    refs.inspectTitle.innerHTML = `Full derivation — <span class="mono">${spec.text}</span>${hoverParam ? ' <span class="dim">(hovering)</span>' : ''}`;
    refs.inspect.innerHTML = derivationHtml(spec, s);
    refs.checkOut.classList.add('hidden');
  }

  function stripTags(h) {
    return h.replace(/<[^>]*>/g, '');
  }

  return { render };
}
