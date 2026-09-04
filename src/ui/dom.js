// Tiny DOM + formatting helpers. No framework.

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(typeof c === 'string' || typeof c === 'number' ? String(c) : c);
  }
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/** Fixed-width-ish decimal, falling back to exponent notation at the extremes. */
export function fmt(x, digits = 4) {
  if (x === null || x === undefined) return '—';
  if (Number.isNaN(x)) return 'NaN';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-4 || a >= 1e6)) return x.toExponential(2).replace('-', '−');
  return x.toFixed(digits).replace('-', '−');
}

/** Always shows a sign; used inside chain-rule products. */
export function fmtSigned(x, digits = 4) {
  const s = fmt(x, digits);
  return s.startsWith('−') ? s : `+${s}`;
}

/** Wraps a number in a span coloured by its sign. */
export function num(x, digits = 4, extraClass = '') {
  const cls = !Number.isFinite(x) ? 'nan' : x > 0 ? 'pos' : x < 0 ? 'neg' : 'zero';
  return `<span class="n ${cls} ${extraClass}">${fmt(x, digits)}</span>`;
}

/** A parenthesised number, for use as a factor in a product. */
export function factor(x, digits = 4) {
  return `<span class="factor">(${fmt(x, digits)})</span>`;
}

// ---------------------------------------------------------------------------
// Math notation (plain HTML + CSS — no external typesetting library)
// ---------------------------------------------------------------------------

/** Stacked fraction. */
export function frac(numerator, denominator) {
  return `<span class="frac"><span class="fnum">${numerator}</span><span class="fden">${denominator}</span></span>`;
}

/** Partial derivative ∂top/∂bottom. */
export function pd(top, bottom) {
  return frac(`∂${top}`, `∂${bottom}`);
}

export const DOT = ' · ';
export const TIMES = ' × ';

/** Joins factors with the multiplication dot. */
export function product(parts) {
  return parts.join(TIMES);
}

// ---------------------------------------------------------------------------
// Colour scales shared by the diagram, the heatmaps and the tables
// ---------------------------------------------------------------------------

export const COLORS = {
  positive: '#1d4ed8',
  negative: '#dc2626',
  class0: '#0d9488',
  class1: '#d97706',
  accent: '#4f46e5',
  ink: '#1e2430',
  muted: '#6b7280',
  grid: '#e5e7eb',
};

/**
 * Blue (negative) → grey (zero) → red (positive), used for weights and
 * gradients wherever sign and magnitude both matter.
 */
export function signedColor(v, scale = 1, alpha = 1) {
  const t = Math.max(-1, Math.min(1, v / (scale || 1)));
  const [r, g, b] =
    t >= 0
      ? [
          Math.round(150 + (220 - 150) * t),
          Math.round(150 - 112 * t),
          Math.round(150 - 112 * t),
        ]
      : [
          Math.round(150 - 121 * -t),
          Math.round(150 - 72 * -t),
          Math.round(150 + 66 * -t),
        ];
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Teal (class 0) ↔ amber (class 1), used for labels and the decision field. */
export function classColor(p, alpha = 1) {
  const t = Math.max(0, Math.min(1, p));
  const c0 = [13, 148, 136];
  const c1 = [217, 119, 6];
  const rgb = c0.map((v, i) => Math.round(v + (c1[i] - v) * t));
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

/** Perceptually-ordered sequential ramp for the loss surface. */
export function lossColor(t) {
  const stops = [
    [0.0, [40, 30, 80]],
    [0.25, [45, 90, 140]],
    [0.5, [40, 150, 130]],
    [0.75, [180, 200, 90]],
    [1.0, [252, 240, 190]],
  ];
  const u = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (u <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const k = (u - t0) / (t1 - t0);
      return c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
    }
  }
  return stops[stops.length - 1][1];
}
