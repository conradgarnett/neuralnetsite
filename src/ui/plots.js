// Small canvas plotting layer: axes, line plots, scatter plots, and the two
// pixel-field renderers (decision boundary and loss surface).
//
// Deliberately generic — it knows nothing about neural networks.

import { COLORS } from './dom.js';

/** Resize a canvas's backing store to its CSS size, accounting for the DPR. */
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth));
  const h = Math.max(1, Math.round(canvas.clientHeight));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, dpr };
}

export class Plot {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{xDomain:[number,number], yDomain:[number,number], pad?:object}} opts
   */
  constructor(canvas, opts = {}) {
    const { ctx, w, h } = fitCanvas(canvas);
    this.canvas = canvas;
    this.ctx = ctx;
    this.w = w;
    this.h = h;
    this.pad = { left: 44, right: 12, top: 12, bottom: 30, ...(opts.pad || {}) };
    this.setDomain(opts.xDomain || [0, 1], opts.yDomain || [0, 1]);
  }

  setDomain(xDomain, yDomain) {
    this.xDomain = xDomain.slice();
    this.yDomain = yDomain.slice();
    return this;
  }

  get plotW() {
    return this.w - this.pad.left - this.pad.right;
  }

  get plotH() {
    return this.h - this.pad.top - this.pad.bottom;
  }

  /** Data x → pixel x. */
  px(v) {
    const [a, b] = this.xDomain;
    return this.pad.left + ((v - a) / (b - a || 1)) * this.plotW;
  }

  /** Data y → pixel y (flipped: larger y is higher on screen). */
  py(v) {
    const [a, b] = this.yDomain;
    return this.pad.top + this.plotH - ((v - a) / (b - a || 1)) * this.plotH;
  }

  /** Pixel x → data x (for mouse interaction). */
  ix(p) {
    const [a, b] = this.xDomain;
    return a + ((p - this.pad.left) / (this.plotW || 1)) * (b - a);
  }

  /** Pixel y → data y. */
  iy(p) {
    const [a, b] = this.yDomain;
    return a + ((this.pad.top + this.plotH - p) / (this.plotH || 1)) * (b - a);
  }

  /** Mouse event → data coordinates. */
  eventToData(evt) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: this.ix(evt.clientX - r.left),
      y: this.iy(evt.clientY - r.top),
      px: evt.clientX - r.left,
      py: evt.clientY - r.top,
    };
  }

  clipPlot() {
    const c = this.ctx;
    c.save();
    c.beginPath();
    c.rect(this.pad.left, this.pad.top, this.plotW, this.plotH);
    c.clip();
    return this;
  }

  restore() {
    this.ctx.restore();
    return this;
  }

  // -------------------------------------------------------------------------

  axes({
    xLabel = '',
    yLabel = '',
    xTicks = 5,
    yTicks = 4,
    grid = true,
    zeroLines = false,
    xFormat = (v) => shortNum(v),
    yFormat = (v) => shortNum(v),
  } = {}) {
    const c = this.ctx;
    c.save();
    c.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    c.fillStyle = COLORS.muted;
    c.strokeStyle = COLORS.grid;
    c.lineWidth = 1;

    const xs = niceTicks(this.xDomain[0], this.xDomain[1], xTicks);
    const ys = niceTicks(this.yDomain[0], this.yDomain[1], yTicks);

    if (grid) {
      c.beginPath();
      for (const v of xs) {
        const x = Math.round(this.px(v)) + 0.5;
        c.moveTo(x, this.pad.top);
        c.lineTo(x, this.pad.top + this.plotH);
      }
      for (const v of ys) {
        const y = Math.round(this.py(v)) + 0.5;
        c.moveTo(this.pad.left, y);
        c.lineTo(this.pad.left + this.plotW, y);
      }
      c.stroke();
    }

    if (zeroLines) {
      c.strokeStyle = '#c3c8d2';
      c.lineWidth = 1.2;
      c.beginPath();
      if (this.xDomain[0] < 0 && this.xDomain[1] > 0) {
        const x = Math.round(this.px(0)) + 0.5;
        c.moveTo(x, this.pad.top);
        c.lineTo(x, this.pad.top + this.plotH);
      }
      if (this.yDomain[0] < 0 && this.yDomain[1] > 0) {
        const y = Math.round(this.py(0)) + 0.5;
        c.moveTo(this.pad.left, y);
        c.lineTo(this.pad.left + this.plotW, y);
      }
      c.stroke();
    }

    c.fillStyle = COLORS.muted;
    c.textAlign = 'center';
    c.textBaseline = 'top';
    for (const v of xs) c.fillText(xFormat(v), this.px(v), this.pad.top + this.plotH + 6);

    c.textAlign = 'right';
    c.textBaseline = 'middle';
    for (const v of ys) c.fillText(yFormat(v), this.pad.left - 6, this.py(v));

    c.fillStyle = COLORS.ink;
    c.font = '11px system-ui, sans-serif';
    if (xLabel) {
      c.textAlign = 'center';
      c.textBaseline = 'bottom';
      c.fillText(xLabel, this.pad.left + this.plotW / 2, this.h - 1);
    }
    if (yLabel) {
      c.save();
      c.translate(10, this.pad.top + this.plotH / 2);
      c.rotate(-Math.PI / 2);
      c.textAlign = 'center';
      c.textBaseline = 'top';
      c.fillText(yLabel, 0, 0);
      c.restore();
    }
    c.restore();
    return this;
  }

  frame(color = '#d5d9e0') {
    const c = this.ctx;
    c.save();
    c.strokeStyle = color;
    c.lineWidth = 1;
    c.strokeRect(
      Math.round(this.pad.left) + 0.5,
      Math.round(this.pad.top) + 0.5,
      Math.round(this.plotW),
      Math.round(this.plotH)
    );
    c.restore();
    return this;
  }

  /** Polyline through data points [[x, y], ...]. */
  line(points, { color = COLORS.accent, width = 2, dash = null, alpha = 1 } = {}) {
    if (points.length < 2) return this;
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    c.strokeStyle = color;
    c.lineWidth = width;
    c.lineJoin = 'round';
    c.lineCap = 'round';
    if (dash) c.setLineDash(dash);
    c.beginPath();
    points.forEach(([x, y], i) => {
      const X = this.px(x);
      const Y = this.py(y);
      if (i === 0) c.moveTo(X, Y);
      else c.lineTo(X, Y);
    });
    c.stroke();
    c.restore();
    return this;
  }

  /** Plot y = f(x) across the current x-domain. */
  fn(f, { color = COLORS.accent, width = 2, samples = 240, dash = null } = {}) {
    const [a, b] = this.xDomain;
    const pts = [];
    for (let i = 0; i <= samples; i++) {
      const x = a + ((b - a) * i) / samples;
      const y = f(x);
      if (Number.isFinite(y)) pts.push([x, y]);
    }
    this.clipPlot();
    this.line(pts, { color, width, dash });
    this.restore();
    return this;
  }

  dot(x, y, { r = 3, fill = COLORS.accent, stroke = null, width = 1.5, alpha = 1 } = {}) {
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    c.beginPath();
    c.arc(this.px(x), this.py(y), r, 0, Math.PI * 2);
    if (fill) {
      c.fillStyle = fill;
      c.fill();
    }
    if (stroke) {
      c.strokeStyle = stroke;
      c.lineWidth = width;
      c.stroke();
    }
    c.restore();
    return this;
  }

  /** Arrow in data coordinates, with a fixed-size pixel head. */
  arrow(x0, y0, x1, y1, { color = COLORS.ink, width = 2, head = 8 } = {}) {
    const c = this.ctx;
    const X0 = this.px(x0);
    const Y0 = this.py(y0);
    const X1 = this.px(x1);
    const Y1 = this.py(y1);
    const ang = Math.atan2(Y1 - Y0, X1 - X0);
    if (!Number.isFinite(ang)) return this;
    c.save();
    c.strokeStyle = color;
    c.fillStyle = color;
    c.lineWidth = width;
    c.lineCap = 'round';
    c.beginPath();
    c.moveTo(X0, Y0);
    c.lineTo(X1, Y1);
    c.stroke();
    c.beginPath();
    c.moveTo(X1, Y1);
    c.lineTo(X1 - head * Math.cos(ang - 0.4), Y1 - head * Math.sin(ang - 0.4));
    c.lineTo(X1 - head * Math.cos(ang + 0.4), Y1 - head * Math.sin(ang + 0.4));
    c.closePath();
    c.fill();
    c.restore();
    return this;
  }

  label(x, y, text, { color = COLORS.ink, align = 'left', baseline = 'bottom', dx = 0, dy = 0, font = '11px system-ui, sans-serif', bg = null } = {}) {
    const c = this.ctx;
    c.save();
    c.font = font;
    c.textAlign = align;
    c.textBaseline = baseline;
    const X = this.px(x) + dx;
    const Y = this.py(y) + dy;
    if (bg) {
      const m = c.measureText(text);
      c.fillStyle = bg;
      c.fillRect(X - (align === 'center' ? m.width / 2 : 0) - 3, Y - 12, m.width + 6, 14);
    }
    c.fillStyle = color;
    c.fillText(text, X, Y);
    c.restore();
    return this;
  }

  /**
   * Fill the plot area pixel-by-pixel from a function of data coordinates.
   * `colorAt(x, y)` must return [r, g, b] (0-255). `step` trades speed for
   * resolution; the field is drawn at step-pixel granularity.
   */
  field(colorAt, { step = 2, alpha = 255 } = {}) {
    const c = this.ctx;
    const x0 = Math.floor(this.pad.left);
    const y0 = Math.floor(this.pad.top);
    const w = Math.floor(this.plotW);
    const h = Math.floor(this.plotH);
    const cols = Math.ceil(w / step);
    const rows = Math.ceil(h / step);

    const img = c.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      const yData = this.iy(y0 + r * step + step / 2);
      for (let q = 0; q < cols; q++) {
        const xData = this.ix(x0 + q * step + step / 2);
        const [R, G, B] = colorAt(xData, yData);
        const o = (r * cols + q) * 4;
        img.data[o] = R;
        img.data[o + 1] = G;
        img.data[o + 2] = B;
        img.data[o + 3] = alpha;
      }
    }

    // Blit the low-resolution field up to full size via an offscreen canvas.
    const off = document.createElement('canvas');
    off.width = cols;
    off.height = rows;
    off.getContext('2d').putImageData(img, 0, 0);

    c.save();
    c.imageSmoothingEnabled = true;
    c.drawImage(off, x0, y0, w, h);
    c.restore();
    return this;
  }
}

// ---------------------------------------------------------------------------
// Tick helpers
// ---------------------------------------------------------------------------

export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const span = max - min;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const stepMul = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  const step = stepMul * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

export function shortNum(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 0.001 || a >= 1e5) return v.toExponential(0).replace('-', '−');
  const s = a < 1 ? v.toFixed(2) : a < 10 ? v.toFixed(1) : v.toFixed(0);
  return s.replace('-', '−');
}

// ---------------------------------------------------------------------------
// Contouring
// ---------------------------------------------------------------------------

/**
 * Marching squares: extract the level set { f = 0 } from a scalar grid.
 *
 * `values` is row-major, `rows` x `cols`, where row r corresponds to
 * y = y0 + (y1 - y0)·r/(rows-1) and column c to x = x0 + (x1 - x0)·c/(cols-1).
 * Returns line segments [[ax, ay], [bx, by]] in data coordinates.
 *
 * Each cell's four corners are classified above/below the level; the resulting
 * 4-bit code selects which pair of cell edges the contour crosses, and the exact
 * crossing point on each edge comes from linear interpolation. The two
 * ambiguous "saddle" cases (5 and 10) emit both segments.
 */
export function marchingSquares(values, rows, cols, x0, x1, y0, y1) {
  const segs = [];
  const gx = (c) => x0 + ((x1 - x0) * c) / (cols - 1);
  const gy = (r) => y0 + ((y1 - y0) * r) / (rows - 1);
  const at = (r, c) => values[r * cols + c];

  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const v0 = at(r, c);         // top-left
      const v1 = at(r, c + 1);     // top-right
      const v2 = at(r + 1, c + 1); // bottom-right
      const v3 = at(r + 1, c);     // bottom-left
      if (!Number.isFinite(v0 + v1 + v2 + v3)) continue;

      const code = (v0 > 0 ? 1 : 0) | (v1 > 0 ? 2 : 0) | (v2 > 0 ? 4 : 0) | (v3 > 0 ? 8 : 0);
      if (code === 0 || code === 15) continue;

      // Crossing points on each of the four cell edges.
      const eTop = () => [gx(c + v0 / (v0 - v1)), gy(r)];
      const eRight = () => [gx(c + 1), gy(r + v1 / (v1 - v2))];
      const eBottom = () => [gx(c + v3 / (v3 - v2)), gy(r + 1)];
      const eLeft = () => [gx(c), gy(r + v0 / (v0 - v3))];

      switch (code) {
        case 1: case 14: segs.push([eLeft(), eTop()]); break;
        case 2: case 13: segs.push([eTop(), eRight()]); break;
        case 3: case 12: segs.push([eLeft(), eRight()]); break;
        case 4: case 11: segs.push([eRight(), eBottom()]); break;
        case 6: case 9: segs.push([eTop(), eBottom()]); break;
        case 7: case 8: segs.push([eLeft(), eBottom()]); break;
        case 5: segs.push([eLeft(), eTop()], [eRight(), eBottom()]); break;
        case 10: segs.push([eTop(), eRight()], [eLeft(), eBottom()]); break;
      }
    }
  }
  return segs;
}
