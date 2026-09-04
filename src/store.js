// Single source of truth for the whole app.
//
// Every tab reads from and writes to this one store, which is why progress
// carries over when you switch tabs: the network, the selected data point, the
// learning rate and the training history all live here, not in any tab.
//
// The store owns *state and orchestration only*. All of the actual mathematics
// lives in src/math/ and is called from here.

import {
  createNetwork,
  cloneNetwork,
  forward,
  backward,
  batchGradients,
  batchLoss,
  accuracy,
  gradientDescentStep,
  bceLoss,
  PARAM_SPECS,
  getParam,
} from './math/network.js';
import { buildDataset } from './math/dataset.js';
import { makeRng } from './math/rng.js';

/** How many training snapshots to keep for the descent path (thinned beyond). */
const MAX_PATH = 2000;
/** How many loss-vs-epoch samples to keep. */
const MAX_HISTORY = 4000;

function flattenParams(net) {
  const out = new Float64Array(PARAM_SPECS.length);
  PARAM_SPECS.forEach((spec, i) => {
    out[i] = getParam(net, spec);
  });
  return out;
}

function thin(arr, max) {
  if (arr.length <= max) return arr;
  const out = [];
  for (let i = 0; i < arr.length; i += 2) out.push(arr[i]);
  return out;
}

class Store {
  constructor() {
    this.listeners = new Set();

    this.state = {
      // --- model -----------------------------------------------------------
      activation: 'tanh',
      net: null,
      weightSeed: 42,

      // --- data ------------------------------------------------------------
      datasetId: 'moons',
      dataSeed: 7,
      noise: 0.14,
      nPoints: 200,
      data: null,
      selectedIndex: 0,

      // --- training --------------------------------------------------------
      learningRate: 0.5,
      mode: 'batch', // 'batch' (all points) | 'sgd' (the selected point only)
      epoch: 0,
      updates: 0,
      phase: 'idle', // idle -> forward -> backward -> updated
      pendingGrads: null, // gradients snapshotted by "Step Backward"
      running: false,
      speed: 4, // epochs per animation frame when running
      diverged: false,

      // --- derived (recomputed by refresh()) --------------------------------
      cache: null, // forward pass for the selected point
      pointGrads: null, // backprop for the selected point
      batchGrads: null, // full-batch gradients
      loss: 0,
      acc: 0,

      // --- history ---------------------------------------------------------
      lossHistory: [], // [{ epoch, loss, acc }]
      path: [], // Float64Array snapshots of all 17 parameters

      // --- walkthrough / inspection UI -------------------------------------
      forwardStage: 0, // 0..5 in the forward-pass animation
      backpropStep: 0, // index into the backprop walkthrough
      inspectedParam: 'W1_0_0', // which weight's derivation is on screen
      surfaceX: 'W1_0_0', // loss-surface horizontal axis
      surfaceY: 'W1_0_1', // loss-surface vertical axis
    };

    this.regenerateData({ notify: false });
    this.resetWeights({ notify: false });
  }

  // -------------------------------------------------------------------------
  // Subscription
  // -------------------------------------------------------------------------

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  notify(changed = 'all') {
    for (const fn of this.listeners) fn(this.state, changed);
  }

  /** Recompute everything derived from (net, data, selected point). */
  refresh() {
    const s = this.state;
    const { X, Y } = s.data;
    const i = Math.min(s.selectedIndex, X.length - 1);

    s.cache = forward(s.net, X[i], s.activation);
    s.pointGrads = backward(s.net, s.cache, Y[i], s.activation);
    s.batchGrads = batchGradients(s.net, X, Y, s.activation);
    s.loss = s.batchGrads.loss;
    s.acc = accuracy(s.net, X, Y, s.activation);
    s.diverged = !Number.isFinite(s.loss);
  }

  /** The gradients the training controls will actually apply. */
  trainingGradients() {
    const s = this.state;
    if (s.mode === 'sgd') {
      const i = s.selectedIndex;
      return batchGradients(s.net, s.data.X, s.data.Y, s.activation, [i]);
    }
    return batchGradients(s.net, s.data.X, s.data.Y, s.activation);
  }

  // -------------------------------------------------------------------------
  // Setup actions
  // -------------------------------------------------------------------------

  regenerateData({ seed, notify = true } = {}) {
    const s = this.state;
    if (seed !== undefined) s.dataSeed = seed;
    s.data = buildDataset(s.datasetId, s.nPoints, s.noise, s.dataSeed);
    s.selectedIndex = Math.min(s.selectedIndex, s.data.X.length - 1);
    if (s.net) this.refresh();
    if (notify) this.notify('data');
  }

  setDataset(id) {
    this.state.datasetId = id;
    this.state.dataSeed = (this.state.dataSeed + 1) | 0;
    this.regenerateData({ notify: false });
    this.clearHistory();
    this.notify('data');
  }

  setNoise(v) {
    this.state.noise = v;
    this.regenerateData({ notify: false });
    this.clearHistory();
    this.notify('data');
  }

  /** Re-initialise every weight randomly and wipe the training history. */
  resetWeights({ seed, notify = true } = {}) {
    const s = this.state;
    s.weightSeed = seed ?? ((Math.random() * 1e9) | 0);
    s.net = createNetwork(s.weightSeed, s.activation);
    s.epoch = 0;
    s.updates = 0;
    s.phase = 'idle';
    s.pendingGrads = null;
    s.running = false;
    s.lossHistory = [];
    s.path = [];
    this.refresh();
    this.record();
    if (notify) this.notify('net');
  }

  clearHistory() {
    const s = this.state;
    s.epoch = 0;
    s.updates = 0;
    s.lossHistory = [];
    s.path = [];
    s.phase = 'idle';
    s.pendingGrads = null;
    this.refresh();
    this.record();
  }

  /**
   * Switching the activation keeps the current weights on purpose: the point is
   * to see the *same* network behave differently because one derivative in the
   * chain changed.
   */
  setActivation(id) {
    this.state.activation = id;
    this.state.pendingGrads = null;
    this.state.phase = 'idle';
    this.refresh();
    this.notify('activation');
  }

  selectPoint(i) {
    const s = this.state;
    s.selectedIndex = Math.max(0, Math.min(i, s.data.X.length - 1));
    s.forwardStage = 0;
    s.backpropStep = 0;
    this.refresh();
    this.notify('selection');
  }

  setLearningRate(lr) {
    this.state.learningRate = lr;
    this.notify('lr');
  }

  setMode(mode) {
    this.state.mode = mode;
    this.state.phase = 'idle';
    this.state.pendingGrads = null;
    this.notify('mode');
  }

  set(patch) {
    Object.assign(this.state, patch);
    this.notify('ui');
  }

  // -------------------------------------------------------------------------
  // Training actions
  // -------------------------------------------------------------------------

  /** Record a point on the loss curve and on the descent path. */
  record() {
    const s = this.state;
    s.lossHistory.push({ epoch: s.epoch, loss: s.loss, acc: s.acc });
    s.path.push(flattenParams(s.net));
    if (s.lossHistory.length > MAX_HISTORY) s.lossHistory = thin(s.lossHistory, MAX_HISTORY);
    if (s.path.length > MAX_PATH) s.path = thin(s.path, MAX_PATH);
  }

  /** Phase 1: run the forward pass and note the loss (no weights change). */
  stepForward() {
    const s = this.state;
    this.refresh();
    s.phase = 'forward';
    this.notify('step');
  }

  /** Phase 2: backpropagate, snapshotting the gradients that will be applied. */
  stepBackward() {
    const s = this.state;
    this.refresh();
    s.pendingGrads = this.trainingGradients();
    s.phase = 'backward';
    this.notify('step');
  }

  /** Phase 3: apply theta := theta - lr * grad using the snapshotted gradients. */
  updateWeights() {
    const s = this.state;
    const grads = s.pendingGrads ?? this.trainingGradients();
    s.net = gradientDescentStep(s.net, grads, s.learningRate);
    s.updates += 1;
    s.epoch += s.mode === 'sgd' ? 1 / s.data.X.length : 1;
    s.phase = 'updated';
    s.pendingGrads = null;
    this.refresh();
    this.record();
    this.notify('step');
  }

  /**
   * One full epoch.
   *
   * In full-batch mode an epoch *is* one update (the gradient already used
   * every example). In SGD mode it is one update per example, in shuffled
   * order -- N cheap, noisy steps instead of one exact one.
   */
  runEpoch({ notify = true } = {}) {
    const s = this.state;
    if (s.mode === 'batch') {
      const g = batchGradients(s.net, s.data.X, s.data.Y, s.activation);
      s.net = gradientDescentStep(s.net, g, s.learningRate);
      s.updates += 1;
    } else {
      const rng = makeRng((s.weightSeed + s.epoch * 7919) | 0);
      for (const i of rng.shuffled(s.data.X.length)) {
        const g = batchGradients(s.net, s.data.X, s.data.Y, s.activation, [i]);
        s.net = gradientDescentStep(s.net, g, s.learningRate);
        s.updates += 1;
      }
    }
    s.epoch += 1;
    s.phase = 'updated';
    s.pendingGrads = null;
    this.refresh();
    this.record();
    if (notify) this.notify('step');
  }

  toggleRunning(force) {
    const s = this.state;
    s.running = force ?? !s.running;
    this.notify('running');
    if (s.running) this.tick();
  }

  tick() {
    const s = this.state;
    if (!s.running) return;
    for (let i = 0; i < s.speed; i++) {
      this.runEpoch({ notify: false });
      // Stop rather than spin on NaN once the run has blown up.
      if (!Number.isFinite(s.loss)) {
        s.running = false;
        s.diverged = true;
        break;
      }
    }
    this.notify('step');
    if (s.running) requestAnimationFrame(() => this.tick());
  }

  // -------------------------------------------------------------------------
  // Diagnostics used by the Gradient Descent tab's callouts
  // -------------------------------------------------------------------------

  /**
   * Classify what the loss curve is currently doing, so the UI can say
   * "this is diverging because the learning rate is too large" at the moment it
   * actually happens rather than as a static caption.
   */
  convergenceDiagnosis() {
    const s = this.state;
    const h = s.lossHistory;
    if (!Number.isFinite(s.loss)) {
      return {
        kind: 'diverged',
        title: 'Diverged (loss is not a number)',
        detail:
          'The steps grew without bound until the arithmetic overflowed. The learning rate is far too large — reset the weights and try a smaller one.',
      };
    }
    if (h.length < 6) return { kind: 'unknown', title: '', detail: '' };

    const window = h.slice(-12);
    const first = window[0].loss;
    const last = window[window.length - 1].loss;
    const spread = Math.max(...window.map((p) => p.loss)) - Math.min(...window.map((p) => p.loss));

    // Count direction changes: an oscillating loss bounces across the valley.
    let flips = 0;
    for (let i = 2; i < window.length; i++) {
      const d1 = window[i - 1].loss - window[i - 2].loss;
      const d2 = window[i].loss - window[i - 1].loss;
      if (d1 * d2 < 0) flips++;
    }

    if (last > first * 1.05 && last > 0.05) {
      return {
        kind: 'diverging',
        title: 'Diverging — the learning rate is too large',
        detail:
          'The loss is climbing. Each step overshoots the minimum and lands somewhere worse than it started, so the gradient gets bigger and the next step overshoots further.',
      };
    }
    if (flips >= 4 && spread > 0.02) {
      return {
        kind: 'oscillating',
        title: 'Oscillating — the learning rate is a bit too large',
        detail:
          'The loss bounces up and down instead of settling. The steps are jumping back and forth across the valley floor rather than sliding down it. Reduce the learning rate.',
      };
    }
    if (s.epoch > 25 && Math.abs(first - last) < 1e-4 && last > 0.25) {
      return {
        kind: 'slow',
        title: 'Converging very slowly — the learning rate is too small',
        detail:
          'The loss is still well above zero but barely moving. The gradient direction is right, the step size just is not covering any ground. Increase the learning rate.',
      };
    }
    if (last < first) {
      return {
        kind: 'converging',
        title: 'Converging steadily',
        detail: 'The loss is falling smoothly — this learning rate suits this loss surface.',
      };
    }
    return {
      kind: 'flat',
      title: 'Settled',
      detail: 'The loss has stopped changing much; the run is at or near a minimum.',
    };
  }
}

export const store = new Store();
