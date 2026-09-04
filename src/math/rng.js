// Seeded pseudo-random number generator.
// Deterministic so that "reset weights" / "regenerate data" are reproducible
// and so the numeric examples in the UI can be reasoned about.

/** mulberry32: small, fast, decent-quality 32-bit PRNG. */
export function makeRng(seed = 42) {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  /** Uniform in [lo, hi). */
  rand.uniform = (lo = 0, hi = 1) => lo + (hi - lo) * rand();

  /** Standard normal via Box-Muller. */
  rand.normal = (mean = 0, std = 1) => {
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  /** Fisher-Yates shuffle of an index array 0..n-1. */
  rand.shuffled = (n) => {
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  };

  return rand;
}
