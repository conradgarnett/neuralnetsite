// Activation functions and their derivatives.
//
// Every entry exposes:
//   f(z)            the activation
//   df(z, a)        its derivative, given the pre-activation z (and, when it
//                   is cheaper that way, the already-computed output a = f(z))
//   tex / dtex      display strings for the function and its derivative
//   derivDetail(z)  the derivative worked out with a real number substituted,
//                   used by the backprop walkthrough
//
// The derivative is not decoration: it is *exactly* the factor that appears in
// the chain rule when the error signal passes back through the neuron.

/** Slope of leaky ReLU on the negative side. */
export const LEAKY_ALPHA = 0.1;

const f4 = (x) => (Number.isFinite(x) ? x.toFixed(4) : String(x));

/** Numerically stable logistic sigmoid. */
export function sigmoid(z) {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z));
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** Derivative of the sigmoid, expressed through its own output. */
export function dsigmoid(z, a = sigmoid(z)) {
  return a * (1 - a);
}

export const ACTIVATIONS = {
  sigmoid: {
    id: 'sigmoid',
    name: 'Sigmoid',
    tex: 'σ(z) = 1 / (1 + e⁻ᶻ)',
    dtex: "σ'(z) = σ(z)·(1 − σ(z))",
    range: [0, 1],
    plotDomain: [-6, 6],
    initScheme: 'xavier',
    blurb: 'Squashes any real number into (0, 1). Saturates at both ends, so its derivative vanishes for large |z| — the classic source of vanishing gradients.',
    f: sigmoid,
    df: (z, a) => {
      const s = a === undefined ? sigmoid(z) : a;
      return s * (1 - s);
    },
    derivDetail(z) {
      const s = sigmoid(z);
      return {
        symbolic: "σ'(z) = σ(z)(1 − σ(z))",
        substituted: `σ(${f4(z)})·(1 − σ(${f4(z)})) = ${f4(s)} × ${f4(1 - s)}`,
        value: s * (1 - s),
      };
    },
  },

  tanh: {
    id: 'tanh',
    name: 'Tanh',
    tex: 'tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)',
    dtex: "tanh'(z) = 1 − tanh²(z)",
    range: [-1, 1],
    plotDomain: [-4, 4],
    initScheme: 'xavier',
    blurb: 'Like the sigmoid but zero-centred, mapping into (−1, 1). Its derivative peaks at 1 (vs. 0.25 for the sigmoid), so gradients shrink more slowly.',
    f: Math.tanh,
    df: (z, a) => {
      const t = a === undefined ? Math.tanh(z) : a;
      return 1 - t * t;
    },
    derivDetail(z) {
      const t = Math.tanh(z);
      return {
        symbolic: "tanh'(z) = 1 − tanh²(z)",
        substituted: `1 − (${f4(t)})² = 1 − ${f4(t * t)}`,
        value: 1 - t * t,
      };
    },
  },

  relu: {
    id: 'relu',
    name: 'ReLU',
    tex: 'ReLU(z) = max(0, z)',
    dtex: "ReLU'(z) = 1 if z > 0, else 0",
    range: [0, Infinity],
    plotDomain: [-4, 4],
    initScheme: 'he',
    blurb: 'Passes positives through untouched and clips negatives to zero. The derivative is a hard switch: a neuron with z ≤ 0 contributes no gradient at all (a "dead" unit).',
    f: (z) => (z > 0 ? z : 0),
    df: (z) => (z > 0 ? 1 : 0),
    derivDetail(z) {
      const on = z > 0;
      return {
        symbolic: "ReLU'(z) = 1 if z > 0, else 0",
        substituted: `z = ${f4(z)} ${on ? '>' : '≤'} 0  →  ${on ? '1' : '0'}`,
        value: on ? 1 : 0,
      };
    },
  },

  leakyRelu: {
    id: 'leakyRelu',
    name: `Leaky ReLU (α = ${LEAKY_ALPHA})`,
    tex: `LReLU(z) = z if z > 0, else αz   (α = ${LEAKY_ALPHA})`,
    dtex: `LReLU'(z) = 1 if z > 0, else α`,
    range: [-Infinity, Infinity],
    plotDomain: [-4, 4],
    initScheme: 'he',
    blurb: `Same as ReLU but with a small slope α = ${LEAKY_ALPHA} on the negative side, so a neuron sitting at z ≤ 0 still passes a little gradient back and can recover.`,
    f: (z) => (z > 0 ? z : LEAKY_ALPHA * z),
    df: (z) => (z > 0 ? 1 : LEAKY_ALPHA),
    derivDetail(z) {
      const on = z > 0;
      return {
        symbolic: "LReLU'(z) = 1 if z > 0, else α",
        substituted: `z = ${f4(z)} ${on ? '>' : '≤'} 0  →  ${on ? '1' : `α = ${LEAKY_ALPHA}`}`,
        value: on ? 1 : LEAKY_ALPHA,
      };
    },
  },
};

/** Ordered list, for dropdowns and the Basics-tab gallery. */
export const ACTIVATION_ORDER = ['sigmoid', 'tanh', 'relu', 'leakyRelu'];

export function getActivation(id) {
  return ACTIVATIONS[id] ?? ACTIVATIONS.tanh;
}
