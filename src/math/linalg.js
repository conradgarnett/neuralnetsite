// Minimal linear algebra, written out explicitly.
//
// Matrices are plain arrays of arrays (row-major): M[i][j] is row i, column j.
// Vectors are plain arrays. Everything here is a pure function -- no state,
// no DOM, no dependencies -- so it can be unit-tested in Node.

/** Zero vector of length n. */
export function zeros(n) {
  return new Array(n).fill(0);
}

/** r x c matrix of zeros. */
export function zerosMat(r, c) {
  return Array.from({ length: r }, () => new Array(c).fill(0));
}

/**
 * Matrix-vector product: (r x c) * (c) -> (r)
 *
 *   out[i] = sum_j M[i][j] * v[j]
 *
 * This is the operation at the heart of a dense layer: each row of M holds one
 * neuron's incoming weights, and the dot product of that row with the input
 * vector is that neuron's weighted sum.
 */
export function matVec(M, v) {
  const r = M.length;
  const c = v.length;
  const out = new Array(r).fill(0);
  for (let i = 0; i < r; i++) {
    let sum = 0;
    for (let j = 0; j < c; j++) sum += M[i][j] * v[j];
    out[i] = sum;
  }
  return out;
}

/** Dot product of two equal-length vectors. */
export function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** Elementwise sum of two vectors. */
export function addVec(a, b) {
  return a.map((v, i) => v + b[i]);
}

/** Elementwise difference of two vectors. */
export function subVec(a, b) {
  return a.map((v, i) => v - b[i]);
}

/** Scale a vector by a scalar. */
export function scaleVec(a, s) {
  return a.map((v) => v * s);
}

/** Elementwise product (Hadamard product) of two vectors. */
export function hadamard(a, b) {
  return a.map((v, i) => v * b[i]);
}

/**
 * Outer product: (r) x (c) -> (r x c), out[i][j] = a[i] * b[j].
 * This is exactly the shape of a weight-matrix gradient: (delta) outer (input).
 */
export function outer(a, b) {
  return a.map((ai) => b.map((bj) => ai * bj));
}

/**
 * Transpose-matrix times vector: (r x c)^T * (r) -> (c)
 *
 *   out[j] = sum_i M[i][j] * v[i]
 *
 * This is how an error signal is pushed backwards through a layer: the same
 * weights that carried activations forward carry gradients backwards, but
 * summed over the *outgoing* index instead of the incoming one.
 */
export function matTVec(M, v) {
  const r = M.length;
  const c = M[0].length;
  const out = new Array(c).fill(0);
  for (let j = 0; j < c; j++) {
    let sum = 0;
    for (let i = 0; i < r; i++) sum += M[i][j] * v[i];
    out[j] = sum;
  }
  return out;
}

/** Deep copy of a matrix. */
export function cloneMat(M) {
  return M.map((row) => row.slice());
}

/** Clamp helper used to keep logs finite. */
export function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}
