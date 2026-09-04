// Synthetic 2-D datasets. Both are deliberately *not* linearly separable, so a
// hidden layer is genuinely required and the decision boundary is interesting
// to watch bend into shape.

import { makeRng } from './rng.js';

/**
 * Two interleaving half-circles ("two moons").
 */
export function twoMoons(n = 200, noise = 0.14, seed = 7) {
  const rng = makeRng(seed);
  const X = [];
  const Y = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    const t = Math.PI * (i / (half - 1));
    X.push([Math.cos(t) + rng.normal(0, noise), Math.sin(t) + rng.normal(0, noise)]);
    Y.push(0);
  }
  for (let i = 0; i < n - half; i++) {
    const t = Math.PI * (i / (n - half - 1));
    X.push([
      1 - Math.cos(t) + rng.normal(0, noise),
      0.5 - Math.sin(t) + rng.normal(0, noise),
    ]);
    Y.push(1);
  }
  return standardize({ X, Y, name: 'Two moons' });
}

/**
 * Two-armed Archimedean spiral. Much harder than the moons for a 4-neuron
 * hidden layer -- a good way to see a model that is too small to fit the data.
 */
export function spiral(n = 200, noise = 0.14, seed = 7) {
  const rng = makeRng(seed);
  const X = [];
  const Y = [];
  const per = Math.floor(n / 2);

  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < per; i++) {
      const r = 0.15 + 1.5 * (i / per);
      const t = 2.2 * Math.PI * (i / per) + c * Math.PI;
      X.push([
        r * Math.cos(t) + rng.normal(0, noise * r),
        r * Math.sin(t) + rng.normal(0, noise * r),
      ]);
      Y.push(c);
    }
  }
  return standardize({ X, Y, name: 'Spiral' });
}

/**
 * Two Gaussian blobs -- linearly separable, and therefore the easy case to
 * fall back on when demonstrating clean convergence.
 */
export function blobs(n = 200, noise = 0.55, seed = 7) {
  const rng = makeRng(seed);
  const X = [];
  const Y = [];
  const centres = [
    [-1.1, -0.8],
    [1.1, 0.8],
  ];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    X.push([
      centres[c][0] + rng.normal(0, noise),
      centres[c][1] + rng.normal(0, noise),
    ]);
    Y.push(c);
  }
  return standardize({ X, Y, name: 'Two blobs' });
}

/**
 * Centre each feature on zero and divide both by a single shared scale, chosen
 * so the average per-feature variance is 1.
 *
 * Not cosmetic: it puts the inputs in a range where one learning rate suits
 * every weight and the initial pre-activations land in the useful
 * (non-saturated) part of the activation function.
 *
 * The scale is deliberately *shared* rather than per-feature. Scaling each
 * feature to unit variance separately would stretch one axis relative to the
 * other and shear the shape of the data -- the moons stop looking like moons,
 * and the decision boundary is much harder to read.
 */
export function standardize(ds) {
  const n = ds.X.length;
  const dims = ds.X[0].length;
  const mean = new Array(dims).fill(0);
  const varr = new Array(dims).fill(0);

  for (const row of ds.X) for (let d = 0; d < dims; d++) mean[d] += row[d] / n;
  for (const row of ds.X)
    for (let d = 0; d < dims; d++) varr[d] += (row[d] - mean[d]) ** 2 / n;

  const scale = Math.sqrt(varr.reduce((a, b) => a + b, 0) / dims) || 1;

  return {
    ...ds,
    X: ds.X.map((row) => row.map((v, d) => (v - mean[d]) / scale)),
    normalization: { mean, scale },
  };
}

export const DATASETS = {
  moons: { id: 'moons', name: 'Two moons', build: twoMoons },
  spiral: { id: 'spiral', name: 'Spiral', build: spiral },
  blobs: { id: 'blobs', name: 'Two blobs (easy)', build: blobs },
};

export function buildDataset(id, n, noise, seed) {
  return (DATASETS[id] ?? DATASETS.moons).build(n, noise, seed);
}

/** Axis-aligned bounding box with a margin, for plotting. */
export function bounds(X, margin = 0.35) {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const [a, b] of X) {
    if (a < x0) x0 = a;
    if (a > x1) x1 = a;
    if (b < y0) y0 = b;
    if (b > y1) y1 = b;
  }
  const dx = (x1 - x0) * margin;
  const dy = (y1 - y0) * margin;
  return { x0: x0 - dx, x1: x1 + dx, y0: y0 - dy, y1: y1 + dy };
}
