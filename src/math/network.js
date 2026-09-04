// The network itself: forward pass, backward pass, and gradient descent,
// implemented from scratch with no ML libraries.
//
// Architecture (fixed):
//
//     x (2)  --W1,b1-->  z1 (4)  --f-->  a1 (4)  --W2,b2-->  z2 (1)  --σ-->  ŷ
//
//   f  = the user-selected hidden activation (sigmoid / tanh / ReLU / leaky ReLU)
//   σ  = logistic sigmoid, fixed on the output because this is binary classification
//   L  = binary cross-entropy
//
// Shapes:
//   W1: 4 x 2   b1: 4     W2: 4 (a 1x4 row, stored flat)   b2: scalar
//
// Everything is a pure function of (net, x, y, activation); nothing here
// touches the DOM, so `tests/run-tests.mjs` can check it directly against
// finite-difference gradients.

import { matVec, addVec, dot, cloneMat, clamp } from './linalg.js';
import { sigmoid, getActivation } from './activations.js';
import { makeRng } from './rng.js';

export const N_INPUT = 2;
export const N_HIDDEN = 4;
export const N_OUTPUT = 1;

/** Keeps log() finite when ŷ saturates to exactly 0 or 1. */
const EPS = 1e-12;
/** Used only for *display* of terms that individually blow up (see below). */
const DISPLAY_EPS = 1e-9;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/**
 * Random initialisation.
 *
 * Scale matters: too large and the sigmoid/tanh units start saturated (tiny
 * derivatives, no learning); too small and every neuron computes the same
 * thing. We use Xavier/Glorot for the saturating activations and He for the
 * ReLU family, which is the standard pairing.
 */
export function createNetwork(seed = 42, activationId = 'tanh') {
  const rng = makeRng(seed);
  const act = getActivation(activationId);
  const scheme = act.initScheme;

  const std1 =
    scheme === 'he'
      ? Math.sqrt(2 / N_INPUT)
      : Math.sqrt(2 / (N_INPUT + N_HIDDEN));
  const std2 =
    scheme === 'he'
      ? Math.sqrt(2 / N_HIDDEN)
      : Math.sqrt(2 / (N_HIDDEN + N_OUTPUT));

  return {
    W1: Array.from({ length: N_HIDDEN }, () =>
      Array.from({ length: N_INPUT }, () => rng.normal(0, std1))
    ),
    b1: Array.from({ length: N_HIDDEN }, () => 0),
    W2: Array.from({ length: N_HIDDEN }, () => rng.normal(0, std2)),
    b2: 0,
    meta: { seed, initScheme: scheme, std1, std2 },
  };
}

export function cloneNetwork(net) {
  return {
    W1: cloneMat(net.W1),
    b1: net.b1.slice(),
    W2: net.W2.slice(),
    b2: net.b2,
    meta: { ...net.meta },
  };
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/**
 * One forward pass for a single input vector x = [x1, x2].
 *
 *   z1 = W1 · x + b1        (4x2 times 2x1, plus a 4-vector)
 *   a1 = f(z1)              (elementwise)
 *   z2 = W2 · a1 + b2       (dot product of two 4-vectors, plus a scalar)
 *   ŷ  = σ(z2)
 *
 * Returns the full cache of intermediate values, because backprop needs every
 * one of them and the UI wants to display them.
 */
export function forward(net, x, activationId) {
  const act = getActivation(activationId);
  const z1 = addVec(matVec(net.W1, x), net.b1);
  const a1 = z1.map((z) => act.f(z));
  const z2 = dot(net.W2, a1) + net.b2;
  const yhat = sigmoid(z2);
  return { x: x.slice(), z1, a1, z2, yhat };
}

/**
 * Binary cross-entropy for one example:
 *
 *   L = −[ y·log(ŷ) + (1 − y)·log(1 − ŷ) ]
 *
 * Minimised when ŷ = y; grows without bound as ŷ approaches the wrong label.
 */
export function bceLoss(yhat, y) {
  const p = clamp(yhat, EPS, 1 - EPS);
  return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

/**
 * Backpropagation for a single example, with every intermediate term kept.
 *
 * The chain, term by term:
 *
 *   ∂L/∂ŷ      = (ŷ − y) / (ŷ(1 − ŷ))          derivative of BCE
 *   ∂ŷ/∂z₂     = ŷ(1 − ŷ)                       derivative of the output sigmoid
 *   δ₂ = ∂L/∂z₂ = ∂L/∂ŷ · ∂ŷ/∂z₂ = ŷ − y        (the two factors cancel exactly)
 *
 *   ∂L/∂W₂ⱼ    = δ₂ · a₁ⱼ                       since ∂z₂/∂W₂ⱼ = a₁ⱼ
 *   ∂L/∂b₂     = δ₂                             since ∂z₂/∂b₂ = 1
 *
 *   ∂L/∂a₁ⱼ    = δ₂ · W₂ⱼ                       push back through the weights
 *   ∂a₁ⱼ/∂z₁ⱼ  = f'(z₁ⱼ)                        the activation's own derivative
 *   δ₁ⱼ = ∂L/∂z₁ⱼ = ∂L/∂a₁ⱼ · f'(z₁ⱼ)
 *
 *   ∂L/∂W₁ⱼₖ   = δ₁ⱼ · xₖ
 *   ∂L/∂b₁ⱼ    = δ₁ⱼ
 *
 * Note on δ₂: we return the *simplified* ŷ − y as the true value, because the
 * unsimplified product is 0/0-unstable when ŷ saturates. The two individual
 * factors are still returned (computed on a clamped ŷ) so the walkthrough can
 * show the cancellation happening.
 */
export function backward(net, cache, y, activationId) {
  const act = getActivation(activationId);
  const { x, z1, a1, yhat } = cache;

  // --- output layer -------------------------------------------------------
  const yc = clamp(yhat, DISPLAY_EPS, 1 - DISPLAY_EPS);
  const dL_dyhat = (yc - y) / (yc * (1 - yc));
  const dyhat_dz2 = yhat * (1 - yhat);
  const dz2 = yhat - y; // == dL_dyhat * dyhat_dz2, algebraically

  const dW2 = a1.map((a) => dz2 * a);
  const db2 = dz2;

  // --- hidden layer -------------------------------------------------------
  const da1 = net.W2.map((w) => dz2 * w); // ∂L/∂a₁ⱼ = δ₂ · W₂ⱼ
  const fPrime1 = z1.map((z, j) => act.df(z, a1[j])); // f'(z₁ⱼ)
  const dz1 = da1.map((d, j) => d * fPrime1[j]); // δ₁ⱼ

  const dW1 = dz1.map((d) => x.map((xi) => d * xi)); // 4 x 2
  const db1 = dz1.slice();

  return {
    loss: bceLoss(yhat, y),
    y,
    dL_dyhat,
    dyhat_dz2,
    dz2,
    dW2,
    db2,
    da1,
    fPrime1,
    dz1,
    dW1,
    db1,
  };
}

/** Convenience: forward + backward for one example. */
export function forwardBackward(net, x, y, activationId) {
  const cache = forward(net, x, activationId);
  const grads = backward(net, cache, y, activationId);
  return { cache, grads };
}

// ---------------------------------------------------------------------------
// Batch quantities
// ---------------------------------------------------------------------------

export function emptyGradients() {
  return {
    dW1: [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    db1: [0, 0, 0, 0],
    dW2: [0, 0, 0, 0],
    db2: 0,
  };
}

/**
 * Mean loss over a dataset:  L = (1/N) Σ_n BCE(ŷ⁽ⁿ⁾, y⁽ⁿ⁾)
 */
export function batchLoss(net, X, Y, activationId) {
  let sum = 0;
  for (let n = 0; n < X.length; n++) {
    sum += bceLoss(forward(net, X[n], activationId).yhat, Y[n]);
  }
  return sum / X.length;
}

/** Fraction of examples classified correctly at a 0.5 threshold. */
export function accuracy(net, X, Y, activationId) {
  let correct = 0;
  for (let n = 0; n < X.length; n++) {
    const p = forward(net, X[n], activationId).yhat;
    if ((p >= 0.5 ? 1 : 0) === Y[n]) correct++;
  }
  return correct / X.length;
}

/**
 * Full-batch gradient: the average of the per-example gradients.
 *
 *   ∂L/∂θ = (1/N) Σ_n ∂L⁽ⁿ⁾/∂θ
 *
 * Averaging (rather than summing) keeps the useful learning-rate range
 * independent of the dataset size.
 */
export function batchGradients(net, X, Y, activationId, indices = null) {
  const idx = indices ?? X.map((_, i) => i);
  const g = emptyGradients();
  let loss = 0;

  for (const n of idx) {
    const { cache, grads } = forwardBackward(net, X[n], Y[n], activationId);
    loss += bceLoss(cache.yhat, Y[n]);
    for (let j = 0; j < N_HIDDEN; j++) {
      for (let k = 0; k < N_INPUT; k++) g.dW1[j][k] += grads.dW1[j][k];
      g.db1[j] += grads.db1[j];
      g.dW2[j] += grads.dW2[j];
    }
    g.db2 += grads.db2;
  }

  const m = idx.length || 1;
  for (let j = 0; j < N_HIDDEN; j++) {
    for (let k = 0; k < N_INPUT; k++) g.dW1[j][k] /= m;
    g.db1[j] /= m;
    g.dW2[j] /= m;
  }
  g.db2 /= m;
  g.loss = loss / m;
  g.count = m;
  return g;
}

// ---------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------

/**
 * The update rule, applied to every parameter:
 *
 *   θ := θ − η · ∂L/∂θ
 *
 * η (the learning rate) is the only knob. Too large and the step overshoots
 * the minimum and the loss climbs; too small and it crawls.
 * Returns a *new* network so callers can diff before/after.
 */
export function gradientDescentStep(net, grads, lr) {
  const out = cloneNetwork(net);
  for (let j = 0; j < N_HIDDEN; j++) {
    for (let k = 0; k < N_INPUT; k++) {
      out.W1[j][k] = net.W1[j][k] - lr * grads.dW1[j][k];
    }
    out.b1[j] = net.b1[j] - lr * grads.db1[j];
    out.W2[j] = net.W2[j] - lr * grads.dW2[j];
  }
  out.b2 = net.b2 - lr * grads.db2;
  return out;
}

// ---------------------------------------------------------------------------
// Parameter addressing
//
// A flat, stable description of all 17 parameters, so the UI can build tables,
// rankings, dropdowns and loss-surface axes without special-casing each block.
// ---------------------------------------------------------------------------

/** Subscript-style HTML label, e.g. W<sup>(1)</sup><sub>2,1</sub>. */
function label(kind, i, j) {
  if (kind === 'W1') return `W<sup>(1)</sup><sub>${i + 1},${j + 1}</sub>`;
  if (kind === 'b1') return `b<sup>(1)</sup><sub>${i + 1}</sub>`;
  if (kind === 'W2') return `W<sup>(2)</sup><sub>1,${i + 1}</sub>`;
  return `b<sup>(2)</sup>`;
}

/** Plain-text label for places that cannot take HTML (e.g. <option>). */
function plainLabel(kind, i, j) {
  if (kind === 'W1') return `W1[${i + 1},${j + 1}]`;
  if (kind === 'b1') return `b1[${i + 1}]`;
  if (kind === 'W2') return `W2[1,${i + 1}]`;
  return 'b2';
}

export const PARAM_SPECS = (() => {
  const specs = [];
  for (let j = 0; j < N_HIDDEN; j++) {
    for (let k = 0; k < N_INPUT; k++) {
      specs.push({
        id: `W1_${j}_${k}`,
        kind: 'W1',
        i: j,
        j: k,
        layer: 1,
        html: label('W1', j, k),
        text: plainLabel('W1', j, k),
        desc: `hidden neuron ${j + 1} ← input x${k + 1}`,
      });
    }
  }
  for (let j = 0; j < N_HIDDEN; j++) {
    specs.push({
      id: `b1_${j}`,
      kind: 'b1',
      i: j,
      j: 0,
      layer: 1,
      html: label('b1', j),
      text: plainLabel('b1', j),
      desc: `bias of hidden neuron ${j + 1}`,
    });
  }
  for (let j = 0; j < N_HIDDEN; j++) {
    specs.push({
      id: `W2_${j}`,
      kind: 'W2',
      i: j,
      j: 0,
      layer: 2,
      html: label('W2', j),
      text: plainLabel('W2', j),
      desc: `output ← hidden neuron ${j + 1}`,
    });
  }
  specs.push({
    id: 'b2',
    kind: 'b2',
    i: 0,
    j: 0,
    layer: 2,
    html: label('b2'),
    text: 'b2',
    desc: 'bias of the output neuron',
  });
  return specs;
})();

export const PARAM_BY_ID = Object.fromEntries(PARAM_SPECS.map((s) => [s.id, s]));

export function getParam(net, spec) {
  switch (spec.kind) {
    case 'W1':
      return net.W1[spec.i][spec.j];
    case 'b1':
      return net.b1[spec.i];
    case 'W2':
      return net.W2[spec.i];
    default:
      return net.b2;
  }
}

export function setParam(net, spec, value) {
  switch (spec.kind) {
    case 'W1':
      net.W1[spec.i][spec.j] = value;
      break;
    case 'b1':
      net.b1[spec.i] = value;
      break;
    case 'W2':
      net.W2[spec.i] = value;
      break;
    default:
      net.b2 = value;
  }
  return net;
}

export function getGrad(grads, spec) {
  if (!grads) return 0;
  switch (spec.kind) {
    case 'W1':
      return grads.dW1[spec.i][spec.j];
    case 'b1':
      return grads.db1[spec.i];
    case 'W2':
      return grads.dW2[spec.i];
    default:
      return grads.db2;
  }
}

// ---------------------------------------------------------------------------
// Gradient checking
// ---------------------------------------------------------------------------

/**
 * Central finite difference for one parameter:
 *
 *   ∂L/∂θ ≈ [ L(θ + h) − L(θ − h) ] / 2h
 *
 * Slower and less accurate than backprop, but derived from nothing but the
 * definition of a derivative -- which is exactly why it is the right way to
 * check that the analytic gradients above are correct.
 */
export function numericalGradient(net, X, Y, activationId, spec, h = 1e-5) {
  const probe = cloneNetwork(net);
  const original = getParam(probe, spec);

  setParam(probe, spec, original + h);
  const lossPlus = batchLoss(probe, X, Y, activationId);

  setParam(probe, spec, original - h);
  const lossMinus = batchLoss(probe, X, Y, activationId);

  setParam(probe, spec, original);
  return (lossPlus - lossMinus) / (2 * h);
}

/** Numerical gradients for every parameter, in PARAM_SPECS order. */
export function numericalGradients(net, X, Y, activationId, h = 1e-5) {
  return PARAM_SPECS.map((spec) => ({
    spec,
    value: numericalGradient(net, X, Y, activationId, spec, h),
  }));
}

// ---------------------------------------------------------------------------
// A fast path, used only by the loss-surface renderer
// ---------------------------------------------------------------------------

/**
 * Mean loss over a dataset, computed without allocating anything.
 *
 * This is a performance-only specialisation of `batchLoss` for the fixed
 * 2 -> 4 -> 1 architecture. The loss surface evaluates the loss at thousands of
 * grid points per frame, and the intermediate arrays that `forward` allocates
 * (one per layer, per example) dominate that cost.
 *
 * `forward` / `batchLoss` above remain the reference implementation -- they are
 * the readable ones, and the test suite asserts that this function agrees with
 * them to the last bit. Do not let the two drift.
 *
 * @param {Float64Array} xs  flattened inputs, [x0, x1] per example
 * @param {Float64Array|number[]} ys  one label per example
 */
export function batchLossFlat(net, xs, ys, activationId) {
  const f = getActivation(activationId).f;
  const W1 = net.W1;
  const w00 = W1[0][0], w01 = W1[0][1];
  const w10 = W1[1][0], w11 = W1[1][1];
  const w20 = W1[2][0], w21 = W1[2][1];
  const w30 = W1[3][0], w31 = W1[3][1];
  const b10 = net.b1[0], b11 = net.b1[1], b12 = net.b1[2], b13 = net.b1[3];
  const v0 = net.W2[0], v1 = net.W2[1], v2 = net.W2[2], v3 = net.W2[3];
  const b2 = net.b2;

  const n = ys.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x0 = xs[i * 2];
    const x1 = xs[i * 2 + 1];

    const a0 = f(w00 * x0 + w01 * x1 + b10);
    const a1 = f(w10 * x0 + w11 * x1 + b11);
    const a2 = f(w20 * x0 + w21 * x1 + b12);
    const a3 = f(w30 * x0 + w31 * x1 + b13);

    const z2 = v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3 + b2;
    const yhat = sigmoid(z2);

    const y = ys[i];
    const p = yhat < EPS ? EPS : yhat > 1 - EPS ? 1 - EPS : yhat;
    sum += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  }
  return sum / n;
}

/** Pack a dataset into the flat arrays `batchLossFlat` expects. */
export function flattenDataset(X, Y, stride = 1) {
  const count = Math.ceil(X.length / stride);
  const xs = new Float64Array(count * 2);
  const ys = new Float64Array(count);
  let k = 0;
  for (let i = 0; i < X.length; i += stride) {
    xs[k * 2] = X[i][0];
    xs[k * 2 + 1] = X[i][1];
    ys[k] = Y[i];
    k++;
  }
  return { xs, ys };
}
