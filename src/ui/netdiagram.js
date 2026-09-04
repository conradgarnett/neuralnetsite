// The node-and-edge picture of the network, shared by the Forward Pass and
// Backpropagation tabs.
//
// Edge thickness encodes magnitude and edge colour encodes sign, for either the
// weights themselves or the gradients flowing back through them. Every edge and
// bias badge is clickable: clicking one selects that parameter for inspection.

import { signedColor, fmt } from './dom.js';
import { PARAM_BY_ID } from '../math/network.js';

const NS = 'http://www.w3.org/2000/svg';

const VW = 720;
const VH = 360;
const R = 26;

const POS = {
  input: { x: 80, ys: [138, 228] },
  hidden: { x: 322, ys: [64, 142, 220, 298] },
  output: { x: 556, y: 181 },
};

function svg(tag, attrs = {}, children = []) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c) n.append(typeof c === 'string' || typeof c === 'number' ? String(c) : c);
  }
  return n;
}

export class NetworkDiagram {
  /**
   * @param {HTMLElement} host
   * @param {{onSelectParam?:Function, onHoverParam?:Function}} handlers
   */
  constructor(host, handlers = {}) {
    this.host = host;
    this.handlers = handlers;
    this.hovered = null;
    this.root = svg('svg', {
      viewBox: `0 0 ${VW} ${VH}`,
      class: 'netdiagram',
      preserveAspectRatio: 'xMidYMid meet',
    });
    host.replaceChildren(this.root);
  }

  /**
   * @param {object} opts
   *   net          the network
   *   cache        forward-pass values for the traced point (optional)
   *   grads        per-example gradients (optional)
   *   encode       'weights' | 'gradients' — what thickness/colour mean
   *   stage        0..5, which forward-pass stage to highlight (null = none)
   *   selected     parameter id currently being inspected
   *   showValues   whether to print node values
   */
  render(opts) {
    const {
      net,
      cache = null,
      grads = null,
      encode = 'weights',
      stage = null,
      selected = null,
      showValues = true,
      label = '',
    } = opts;

    const g = svg('g');

    // Magnitude scale for the encoded quantity, so thickness is comparable
    // across the whole diagram.
    const values = [];
    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 2; k++) values.push(this.encodedValue(encode, net, grads, 'W1', j, k));
      values.push(this.encodedValue(encode, net, grads, 'W2', j, 0));
    }
    const scale = Math.max(1e-9, ...values.map(Math.abs));

    // --- column headings ----------------------------------------------------
    g.append(
      svg('text', { x: POS.input.x, y: 24, class: 'nd-head', 'text-anchor': 'middle' }, 'inputs  x'),
      svg('text', { x: POS.hidden.x, y: 24, class: 'nd-head', 'text-anchor': 'middle' }, 'hidden layer  (4 neurons)'),
      svg('text', { x: POS.output.x, y: 24, class: 'nd-head', 'text-anchor': 'middle' }, 'output  ŷ')
    );
    if (label) {
      g.append(svg('text', { x: VW - 8, y: 24, class: 'nd-head', 'text-anchor': 'end' }, label));
    }

    // --- edges (drawn first so nodes sit on top) ---------------------------
    const edges = svg('g', { class: 'nd-edges' });

    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 2; k++) {
        edges.append(
          this.edge({
            id: `W1_${j}_${k}`,
            x1: POS.input.x + R,
            y1: POS.input.ys[k],
            x2: POS.hidden.x - R,
            y2: POS.hidden.ys[j],
            value: this.encodedValue(encode, net, grads, 'W1', j, k),
            weight: net.W1[j][k],
            scale,
            active: stage === 1,
            selected: selected === `W1_${j}_${k}`,
          })
        );
      }
    }
    for (let j = 0; j < 4; j++) {
      edges.append(
        this.edge({
          id: `W2_${j}`,
          x1: POS.hidden.x + R,
          y1: POS.hidden.ys[j],
          x2: POS.output.x - R,
          y2: POS.output.y,
          value: this.encodedValue(encode, net, grads, 'W2', j, 0),
          weight: net.W2[j],
          scale,
          active: stage === 3,
          selected: selected === `W2_${j}`,
        })
      );
    }
    g.append(edges);

    // --- nodes --------------------------------------------------------------
    for (let k = 0; k < 2; k++) {
      g.append(
        this.node({
          x: POS.input.x,
          y: POS.input.ys[k],
          main: cache && showValues ? fmt(cache.x[k], 3) : `x${k + 1}`,
          caption: cache && showValues ? `x${k + 1}` : '',
          fillT: 0.5,
          active: stage === 0,
        })
      );
    }

    for (let j = 0; j < 4; j++) {
      const z = cache ? cache.z1[j] : null;
      const a = cache ? cache.a1[j] : null;
      g.append(
        this.node({
          x: POS.hidden.x,
          y: POS.hidden.ys[j],
          main: a !== null && showValues ? fmt(a, 3) : `h${j + 1}`,
          caption: z !== null && showValues ? `z = ${fmt(z, 3)}` : '',
          fillT: a === null ? 0.5 : squash(a),
          active: stage === 1 || stage === 2,
          emphasis: stage === 2,
        })
      );
      g.append(
        this.biasBadge({
          id: `b1_${j}`,
          x: POS.hidden.x - R - 4,
          y: POS.hidden.ys[j] + R - 2,
          value: net.b1[j],
          grad: grads ? grads.db1[j] : null,
          selected: selected === `b1_${j}`,
        })
      );
    }

    g.append(
      this.node({
        x: POS.output.x,
        y: POS.output.y,
        main: cache && showValues ? fmt(cache.yhat, 3) : 'ŷ',
        caption: cache && showValues ? `z = ${fmt(cache.z2, 3)}` : '',
        fillT: cache ? cache.yhat : 0.5,
        active: stage === 3 || stage === 4,
        emphasis: stage === 4,
        classColored: true,
      })
    );
    g.append(
      this.biasBadge({
        id: 'b2',
        x: POS.output.x - R - 4,
        y: POS.output.y + R - 2,
        value: net.b2,
        grad: grads ? grads.db2 : null,
        selected: selected === 'b2',
      })
    );

    // --- prediction / loss readout -----------------------------------------
    if (cache && showValues) {
      const gx = POS.output.x + R + 22;
      g.append(
        svg('line', {
          x1: POS.output.x + R,
          y1: POS.output.y,
          x2: gx - 6,
          y2: POS.output.y,
          class: `nd-out-link${stage === 5 ? ' active' : ''}`,
        }),
        svg('text', { x: gx, y: POS.output.y - 6, class: 'nd-out' }, `ŷ = ${fmt(cache.yhat, 4)}`),
        svg('text', { x: gx, y: POS.output.y + 12, class: 'nd-out-sub' },
          `y = ${opts.target !== undefined ? opts.target : '—'}`),
        opts.loss !== undefined
          ? svg('text', { x: gx, y: POS.output.y + 30, class: `nd-out-sub${stage === 5 ? ' hot' : ''}` }, `L = ${fmt(opts.loss, 4)}`)
          : null
      );
    }

    this.root.replaceChildren(g);
    return this;
  }

  encodedValue(encode, net, grads, kind, i, j) {
    if (encode === 'gradients') {
      if (!grads) return 0;
      return kind === 'W1' ? grads.dW1[i][j] : grads.dW2[i];
    }
    return kind === 'W1' ? net.W1[i][j] : net.W2[i];
  }

  edge({ id, x1, y1, x2, y2, value, weight, scale, active, selected }) {
    const mag = Math.abs(value) / scale;
    const width = 1 + 6 * Math.sqrt(mag);
    const hovered = this.hovered === id;
    const gEl = svg('g', {
      class: `nd-edge${active ? ' active' : ''}${selected ? ' selected' : ''}${hovered ? ' hovered' : ''}`,
      'data-param': id,
    });

    // A wide invisible line makes thin edges comfortably clickable.
    gEl.append(
      svg('line', { x1, y1, x2, y2, stroke: signedColor(value, scale, 0.92), 'stroke-width': width, 'stroke-linecap': 'round' }),
      svg('line', { x1, y1, x2, y2, stroke: 'transparent', 'stroke-width': Math.max(14, width + 8), class: 'nd-hit' })
    );

    if (selected || hovered) {
      gEl.append(
        svg('text', {
          x: (x1 + x2) / 2,
          y: (y1 + y2) / 2 - 6,
          class: 'nd-edge-label',
          'text-anchor': 'middle',
        }, fmt(weight, 3))
      );
    }

    const spec = PARAM_BY_ID[id];
    gEl.addEventListener('click', () => this.handlers.onSelectParam?.(id));
    gEl.addEventListener('mouseenter', () => {
      this.hovered = id;
      this.handlers.onHoverParam?.(id, spec);
    });
    gEl.addEventListener('mouseleave', () => {
      if (this.hovered === id) this.hovered = null;
      this.handlers.onHoverParam?.(null, null);
    });
    return gEl;
  }

  biasBadge({ id, x, y, value, grad, selected }) {
    const hovered = this.hovered === id;
    const gEl = svg('g', {
      class: `nd-bias${selected ? ' selected' : ''}${hovered ? ' hovered' : ''}`,
      'data-param': id,
    });
    gEl.append(
      svg('circle', { cx: x, cy: y, r: 9, fill: signedColor(value, Math.max(0.5, Math.abs(value)), 0.85) }),
      svg('text', { x, y: y + 3.5, 'text-anchor': 'middle', class: 'nd-bias-t' }, 'b')
    );
    if (selected || hovered) {
      gEl.append(
        svg('text', { x, y: y + 24, 'text-anchor': 'middle', class: 'nd-edge-label' }, fmt(value, 3))
      );
    }
    gEl.addEventListener('click', () => this.handlers.onSelectParam?.(id));
    gEl.addEventListener('mouseenter', () => {
      this.hovered = id;
      this.handlers.onHoverParam?.(id, PARAM_BY_ID[id]);
    });
    gEl.addEventListener('mouseleave', () => {
      if (this.hovered === id) this.hovered = null;
      this.handlers.onHoverParam?.(null, null);
    });
    return gEl;
  }

  node({ x, y, main, caption, fillT = 0.5, active = false, emphasis = false, classColored = false }) {
    const gEl = svg('g', { class: `nd-node${active ? ' active' : ''}${emphasis ? ' emph' : ''}` });
    const fill = classColored
      ? `rgba(${mix([13, 148, 136], [217, 119, 6], fillT).join(',')}, 0.20)`
      : `rgba(${mix([59, 130, 246], [244, 114, 60], fillT).join(',')}, 0.18)`;
    gEl.append(
      svg('circle', { cx: x, cy: y, r: R, fill, class: 'nd-node-c' }),
      svg('text', { x, y: y + 4, 'text-anchor': 'middle', class: 'nd-node-v' }, main)
    );
    if (caption) {
      gEl.append(
        svg('text', { x, y: y + R + 14, 'text-anchor': 'middle', class: 'nd-node-cap' }, caption)
      );
    }
    return gEl;
  }
}

function mix(a, b, t) {
  const u = Math.max(0, Math.min(1, t));
  return a.map((v, i) => Math.round(v + (b[i] - v) * u));
}

/** Map an unbounded activation into [0, 1] for colouring only. */
function squash(a) {
  return 1 / (1 + Math.exp(-a));
}
