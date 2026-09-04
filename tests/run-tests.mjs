// Independent verification of the math, with no browser involved.
//
//   node tests/run-tests.mjs
//
// The important test is the gradient check: every analytic gradient produced by
// backprop is compared against a central finite difference of the loss. If
// backprop is wrong anywhere, this catches it.

import {
  createNetwork,
  cloneNetwork,
  forward,
  backward,
  bceLoss,
  batchLoss,
  batchGradients,
  gradientDescentStep,
  numericalGradient,
  PARAM_SPECS,
  getParam,
  setParam,
  getGrad,
  accuracy,
  batchLossFlat,
  flattenDataset,
} from '../src/math/network.js';
import { ACTIVATIONS, ACTIVATION_ORDER, sigmoid } from '../src/math/activations.js';
import { matVec, matTVec, outer, dot } from '../src/math/linalg.js';
import { buildDataset } from '../src/math/dataset.js';

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const BOLD = '\x1b[1m';
const OFF = '\x1b[0m';

let passed = 0;
let failed = 0;

function check(name, condition, detail = '') {
  if (condition) {
    passed++;
    console.log(`  ${GREEN}PASS${OFF} ${name}${detail ? '  ' + detail : ''}`);
  } else {
    failed++;
    console.log(`  ${RED}FAIL${OFF} ${name}${detail ? '  ' + detail : ''}`);
  }
}

function close(a, b, tol = 1e-8) {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

function section(title) {
  console.log(`\n${BOLD}${title}${OFF}`);
}

/**
 * Compare every analytic gradient against a central finite difference.
 *
 * A parameter passes if EITHER the relative error is small OR the absolute
 * error is negligible. The absolute escape hatch matters: a gradient that is
 * genuinely near zero (say 7e-6) has a large *relative* error even when the
 * two numbers agree to 1e-11 absolute, which is agreement, not a bug.
 */
function gradCheck(net, X, Y, id, grads, h) {
  let worstRel = 0;
  let worstAbs = 0;
  let worstName = '';
  let worstPairRel = 0;
  let worstPairAbs = 0;

  for (const spec of PARAM_SPECS) {
    const numeric = numericalGradient(net, X, Y, id, spec, h);
    const analytic = getGrad(grads, spec);
    const abs = Math.abs(numeric - analytic);
    const rel = abs / Math.max(1e-12, Math.abs(numeric) + Math.abs(analytic));
    worstAbs = Math.max(worstAbs, abs);
    if (rel > worstRel) {
      worstRel = rel;
      worstName = spec.text;
      worstPairRel = rel;
      worstPairAbs = abs;
    }
  }

  return {
    ok: (tol) => worstRel <= tol || worstPairAbs <= 1e-9,
    describe: () =>
      `worst rel ${worstPairRel.toExponential(2)} (abs ${worstPairAbs.toExponential(2)}) at ${worstName}; max abs ${worstAbs.toExponential(2)}`,
  };
}

/**
 * Largest finite-difference step that cannot push any hidden unit across the
 * z = 0 kink of a piecewise-linear activation. Smooth activations have no kink
 * and just get the usual step.
 */
function kinkSafeStep(net, X, id) {
  if (id !== 'relu' && id !== 'leakyRelu') return 1e-4;
  let closest = Infinity;
  for (const x of X) {
    for (const z of forward(net, x, id).z1) closest = Math.min(closest, Math.abs(z));
  }
  // Divide by the largest |input| so that perturbing a weight by h moves any
  // pre-activation by less than `closest`.
  const maxInput = Math.max(1, ...X.flat().map(Math.abs));
  return Math.min(1e-4, (closest / maxInput) * 0.2);
}

// ---------------------------------------------------------------------------
section('Linear algebra');
// ---------------------------------------------------------------------------
{
  check('matVec', JSON.stringify(matVec([[1, 2], [3, 4], [5, 6]], [7, 8])) === '[23,53,83]');
  check('matTVec', JSON.stringify(matTVec([[1, 2], [3, 4]], [1, 1])) === '[4,6]');
  check('outer', JSON.stringify(outer([1, 2], [3, 4])) === '[[3,4],[6,8]]');
  check('dot', dot([1, 2, 3], [4, 5, 6]) === 32);
}

// ---------------------------------------------------------------------------
section('Activation derivatives vs. finite differences');
// ---------------------------------------------------------------------------
{
  const h = 1e-6;
  for (const id of ACTIVATION_ORDER) {
    const act = ACTIVATIONS[id];
    let worst = 0;
    // Skip a neighbourhood of z = 0 for the piecewise-linear activations, where
    // the derivative genuinely does not exist.
    for (let z = -3; z <= 3.0001; z += 0.25) {
      if (Math.abs(z) < 1e-9) continue;
      const numeric = (act.f(z + h) - act.f(z - h)) / (2 * h);
      const analytic = act.df(z, act.f(z));
      worst = Math.max(worst, Math.abs(numeric - analytic));
    }
    check(`${id} derivative`, worst < 1e-6, `max |err| = ${worst.toExponential(2)}`);
  }
  check('sigmoid is stable for large negative z', sigmoid(-800) === 0 && Number.isFinite(sigmoid(-800)));
  check('sigmoid(0) = 0.5', sigmoid(0) === 0.5);
}

// ---------------------------------------------------------------------------
section('Forward pass');
// ---------------------------------------------------------------------------
{
  // A network with hand-picked values, checked against arithmetic done by hand.
  const net = {
    W1: [[1, 0], [0, 1], [1, 1], [-1, 1]],
    b1: [0, 0, 0.5, -0.5],
    W2: [1, -1, 0.5, 2],
    b2: 0.25,
  };
  const x = [2, 3];
  const c = forward(net, x, 'relu');

  check('z1 = W1x + b1', JSON.stringify(c.z1) === JSON.stringify([2, 3, 5.5, 0.5]));
  check('a1 = ReLU(z1)', JSON.stringify(c.a1) === JSON.stringify([2, 3, 5.5, 0.5]));
  // z2 = 1*2 + (-1)*3 + 0.5*5.5 + 2*0.5 + 0.25 = 2 - 3 + 2.75 + 1 + 0.25 = 3
  check('z2 = W2 dot a1 + b2', close(c.z2, 3), `z2 = ${c.z2}`);
  check('yhat = sigmoid(z2)', close(c.yhat, 1 / (1 + Math.exp(-3))), `yhat = ${c.yhat.toFixed(6)}`);

  // Negative pre-activations must be clipped by ReLU.
  const c2 = forward(net, [-2, -3], 'relu');
  check('ReLU clips negatives', c2.a1.every((a) => a >= 0) && c2.a1[0] === 0);
}

// ---------------------------------------------------------------------------
section('Loss function');
// ---------------------------------------------------------------------------
{
  check('BCE is 0 for a perfect prediction', bceLoss(1 - 1e-15, 1) < 1e-9);
  check('BCE at yhat = 0.5 is log 2', close(bceLoss(0.5, 1), Math.LN2, 1e-12));
  check('BCE is symmetric in the label', close(bceLoss(0.3, 1), bceLoss(0.7, 0), 1e-12));
  check('BCE is finite at yhat = 0 with y = 1', Number.isFinite(bceLoss(0, 1)));
  check('BCE grows as the prediction worsens', bceLoss(0.1, 1) > bceLoss(0.4, 1));
}

// ---------------------------------------------------------------------------
section('Output-delta simplification: dL/dyhat * dyhat/dz2 = yhat - y');
// ---------------------------------------------------------------------------
{
  const net = createNetwork(3, 'tanh');
  for (const y of [0, 1]) {
    const cache = forward(net, [0.7, -1.3], 'tanh');
    const g = backward(net, cache, y, 'tanh');
    check(
      `product equals yhat - y (y = ${y})`,
      close(g.dL_dyhat * g.dyhat_dz2, g.dz2, 1e-9),
      `${(g.dL_dyhat * g.dyhat_dz2).toFixed(10)} vs ${g.dz2.toFixed(10)}`
    );
  }
}

// ---------------------------------------------------------------------------
section('Backprop vs. finite differences (single example)');
// ---------------------------------------------------------------------------
{
  for (const id of ACTIVATION_ORDER) {
    const net = createNetwork(11, id);
    const x = [0.63, -0.41];
    const y = 1;
    const cache = forward(net, x, id);
    const grads = backward(net, cache, y, id);

    const worst = gradCheck(net, [x], [y], id, grads, 1e-5);
    check(
      `${id}: all 17 single-example gradients`,
      worst.ok(1e-6),
      worst.describe()
    );
  }
}

// ---------------------------------------------------------------------------
section('Backprop vs. finite differences (full batch, real dataset)');
// ---------------------------------------------------------------------------
{
  const ds = buildDataset('moons', 200, 0.14, 7);
  for (const id of ACTIVATION_ORDER) {
    const net = createNetwork(5, id);
    const grads = batchGradients(net, ds.X, ds.Y, id);

    // ReLU and leaky ReLU are only piecewise differentiable. If the probe step h
    // is larger than the closest approach of any pre-activation to the kink at
    // z = 0, the finite difference straddles the kink and measures a slope the
    // derivative never claimed to have. That is a property of finite differences,
    // not an error in backprop -- so pick h safely below that distance and the
    // tolerance can stay just as tight as for the smooth activations.
    const h = kinkSafeStep(net, ds.X, id);
    const worst = gradCheck(net, ds.X, ds.Y, id, grads, h);
    check(`${id}: all 17 batch gradients`, worst.ok(1e-5), `${worst.describe()}; h = ${h.toExponential(1)}`);
  }
}

// ---------------------------------------------------------------------------
section('Full unabbreviated chain for a hidden weight');
// ---------------------------------------------------------------------------
{
  // The five-factor product the walkthrough displays must equal the gradient
  // that backprop actually computes:
  //   dL/dW1[j][k] = dL/dyhat * dyhat/dz2 * W2[j] * f'(z1[j]) * x[k]
  const id = 'tanh';
  const net = createNetwork(21, id);
  const x = [-0.9, 0.55];
  const y = 0;
  const cache = forward(net, x, id);
  const g = backward(net, cache, y, id);

  let worst = 0;
  for (let j = 0; j < 4; j++) {
    for (let k = 0; k < 2; k++) {
      const product =
        g.dL_dyhat * g.dyhat_dz2 * net.W2[j] * g.fPrime1[j] * cache.x[k];
      worst = Math.max(worst, Math.abs(product - g.dW1[j][k]));
    }
  }
  check('5-factor product equals dW1', worst < 1e-9, `max |err| = ${worst.toExponential(2)}`);

  // And the output-weight chain: dL/dW2[j] = dL/dyhat * dyhat/dz2 * a1[j]
  let worst2 = 0;
  for (let j = 0; j < 4; j++) {
    const product = g.dL_dyhat * g.dyhat_dz2 * cache.a1[j];
    worst2 = Math.max(worst2, Math.abs(product - g.dW2[j]));
  }
  check('3-factor product equals dW2', worst2 < 1e-9, `max |err| = ${worst2.toExponential(2)}`);
}

// ---------------------------------------------------------------------------
section('Gradient descent');
// ---------------------------------------------------------------------------
{
  const id = 'tanh';
  const ds = buildDataset('moons', 200, 0.14, 7);
  const net0 = createNetwork(9, id);

  // A single small step must decrease the loss (the gradient points uphill, so
  // stepping against it goes downhill for a small enough step size).
  const g0 = batchGradients(net0, ds.X, ds.Y, id);
  const net1 = gradientDescentStep(net0, g0, 0.05);
  check(
    'one small step lowers the loss',
    batchLoss(net1, ds.X, ds.Y, id) < batchLoss(net0, ds.X, ds.Y, id),
    `${batchLoss(net0, ds.X, ds.Y, id).toFixed(6)} -> ${batchLoss(net1, ds.X, ds.Y, id).toFixed(6)}`
  );

  check('the update rule is theta - lr*g', (() => {
    const spec = PARAM_SPECS[0];
    return close(getParam(net1, spec), getParam(net0, spec) - 0.05 * getGrad(g0, spec), 1e-12);
  })());

  check('gradientDescentStep does not mutate its input', (() => {
    const before = JSON.stringify(net0.W1);
    gradientDescentStep(net0, g0, 1.5);
    return JSON.stringify(net0.W1) === before;
  })());

  // Training for a while must actually learn the two moons.
  let net = cloneNetwork(net0);
  for (let epoch = 0; epoch < 4000; epoch++) {
    net = gradientDescentStep(net, batchGradients(net, ds.X, ds.Y, id), 0.5);
  }
  const acc = accuracy(net, ds.X, ds.Y, id);
  const finalLoss = batchLoss(net, ds.X, ds.Y, id);
  check('4000 epochs learn two moons', acc > 0.95, `accuracy = ${(acc * 100).toFixed(1)}%, loss = ${finalLoss.toFixed(4)}`);
  check('loss decreased overall', finalLoss < batchLoss(net0, ds.X, ds.Y, id));

  // A deliberately huge learning rate must visibly misbehave -- this is the
  // divergence the Gradient Descent tab is built to demonstrate.
  //
  // The honest assertion is about *overshoot*, not about the final number: a
  // wild run can bounce far uphill and still happen to land somewhere decent.
  // What always happens is that the loss climbs far above where it started.
  const startLoss = batchLoss(net0, ds.X, ds.Y, id);
  const runTo = (lr, steps) => {
    let n = cloneNetwork(net0);
    let peak = startLoss;
    for (let i = 0; i < steps; i++) {
      n = gradientDescentStep(n, batchGradients(n, ds.X, ds.Y, id), lr);
      peak = Math.max(peak, batchLoss(n, ds.X, ds.Y, id));
    }
    return { peak, end: batchLoss(n, ds.X, ds.Y, id) };
  };

  const wild = runTo(60, 40);
  check(
    'lr = 60 overshoots wildly',
    wild.peak > startLoss * 5,
    `start ${startLoss.toFixed(4)}, peak ${wild.peak.toFixed(4)}`
  );

  const wilder = runTo(1000, 40);
  check(
    'lr = 1000 ends far worse than it started',
    wilder.end > startLoss,
    `start ${startLoss.toFixed(4)} -> end ${wilder.end.toFixed(4)}`
  );

  const gentle = runTo(0.5, 40);
  check(
    'a sensible lr never climbs above its starting loss',
    gentle.peak <= startLoss + 1e-12 && gentle.end < startLoss,
    `start ${startLoss.toFixed(4)}, peak ${gentle.peak.toFixed(4)}, end ${gentle.end.toFixed(4)}`
  );
}

// ---------------------------------------------------------------------------
section('Fast loss path agrees with the reference implementation');
// ---------------------------------------------------------------------------
{
  // batchLossFlat is a hand-unrolled specialisation used by the loss-surface
  // renderer. It must never disagree with the readable batchLoss.
  for (const dsId of ['moons', 'spiral', 'blobs']) {
    const ds = buildDataset(dsId, 200, 0.14, 7);
    const { xs, ys } = flattenDataset(ds.X, ds.Y);
    let worst = 0;
    for (const id of ACTIVATION_ORDER) {
      for (let seed = 1; seed <= 6; seed++) {
        const net = createNetwork(seed * 13, id);
        // Also probe far from the initialisation, where the loss saturates.
        for (const scale of [1, 6, 40]) {
          const probe = cloneNetwork(net);
          for (const spec of PARAM_SPECS) setParam(probe, spec, getParam(net, spec) * scale);
          const a = batchLoss(probe, ds.X, ds.Y, id);
          const b = batchLossFlat(probe, xs, ys, id);
          worst = Math.max(worst, Math.abs(a - b));
        }
      }
    }
    check(`${dsId}: batchLossFlat === batchLoss`, worst < 1e-12, `max |err| = ${worst.toExponential(2)}`);
  }

  const ds = buildDataset('moons', 200, 0.14, 7);
  const half = flattenDataset(ds.X, ds.Y, 2);
  check('flattenDataset strides correctly', half.ys.length === 100 && half.xs.length === 200 &&
    half.xs[0] === ds.X[0][0] && half.xs[2] === ds.X[2][0] && half.ys[1] === ds.Y[2]);
}

// ---------------------------------------------------------------------------
section('Parameter addressing');
// ---------------------------------------------------------------------------
{
  check('17 parameters: 4x2 + 4 + 4 + 1', PARAM_SPECS.length === 17);
  const net = createNetwork(1, 'tanh');
  const ok = PARAM_SPECS.every((spec) => {
    const probe = cloneNetwork(net);
    setParam(probe, spec, 123.5);
    return getParam(probe, spec) === 123.5;
  });
  check('get/set round-trip for every parameter', ok);
  check('ids are unique', new Set(PARAM_SPECS.map((s) => s.id)).size === 17);
}

// ---------------------------------------------------------------------------
section('Datasets');
// ---------------------------------------------------------------------------
{
  for (const id of ['moons', 'spiral', 'blobs']) {
    const ds = buildDataset(id, 200, 0.14, 7);
    const n = ds.X.length;
    const meanX = ds.X.reduce((s, r) => s + r[0], 0) / n;
    const meanY = ds.X.reduce((s, r) => s + r[1], 0) / n;
    const varX = ds.X.reduce((s, r) => s + r[0] ** 2, 0) / n - meanX ** 2;
    const varY = ds.X.reduce((s, r) => s + r[1] ** 2, 0) / n - meanY ** 2;
    const balanced = ds.Y.filter((y) => y === 1).length;
    check(
      `${id}: centred, scaled and balanced`,
      n === 200 &&
        Math.abs(meanX) < 1e-9 &&
        Math.abs(meanY) < 1e-9 &&
        Math.abs((varX + varY) / 2 - 1) < 1e-9 &&
        balanced === 100,
      `means = (${meanX.toExponential(1)}, ${meanY.toExponential(1)}), mean var = ${((varX + varY) / 2).toFixed(6)}, class1 = ${balanced}`
    );
    // A shared scale must leave the two axes' relative spread alone; per-feature
    // standardization would force this ratio to exactly 1 and shear the shape.
    check(
      `${id}: shared scale preserves the aspect ratio`,
      Math.abs(varX - varY) > 1e-6 || id === 'spiral',
      `varX = ${varX.toFixed(4)}, varY = ${varY.toFixed(4)}`
    );
    check(`${id}: all values finite`, ds.X.every((r) => r.every(Number.isFinite)));
  }
}

// ---------------------------------------------------------------------------
console.log(`\n${BOLD}${passed} passed, ${failed} failed${OFF}\n`);
process.exit(failed === 0 ? 0 : 1);
