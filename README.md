# Neural Network Math Visualizer

An interactive walkthrough of the mathematics of a small neural network — the
forward pass, backpropagation, and gradient descent — with every intermediate
number on screen. It is a teaching tool, not a machine-learning library: the aim
is that no step of the calculus is ever hidden.

Every number you see is computed live in the browser from scratch. There are no
machine-learning dependencies, no build step, and no backend.

## Running it

The app uses ES modules, so it needs to be served over HTTP rather than opened
as a `file://` URL. Any static server works:

```bash
python3 -m http.server 8000      # or: npm start
```

Then open <http://localhost:8000>.

To check the mathematics independently of the browser:

```bash
node tests/run-tests.mjs          # or: npm test
```

Node 16+ is enough; the suite has no dependencies.

## The network

Fixed and deliberately tiny, so that every parameter fits on screen:

```
x (2)  --W(1), b(1)-->  z(1) (4)  --f-->  a(1) (4)  --W(2), b(2)-->  z(2)  --σ-->  ŷ
```

- **2 inputs → 4 hidden neurons → 1 output.** 17 parameters in total
  (a 4×2 weight matrix, 4 hidden biases, 4 output weights, 1 output bias).
- **Hidden activation `f`** is selectable at any time — sigmoid, tanh, ReLU, or
  leaky ReLU (α = 0.1). Switching it live-updates the forward pass, every
  derivative in the backward pass, and the decision boundary; the header always
  shows the derivative formula currently in play.
- **Output activation** is always the logistic sigmoid, and the loss is always
  **binary cross-entropy**, because the task is binary classification.
- **Data** is a synthetic 2-D set of 200 points in two classes: two moons, a
  spiral, or two Gaussian blobs. Each is centred on the origin and divided by a
  single shared scale, which keeps the initial pre-activations in the useful part
  of the activation function without distorting the shape of the data.
- **Weight initialisation** is Xavier/Glorot for the saturating activations and
  He for the ReLU family, chosen automatically from the current activation.

## What each panel shows

### Basics

The minimum mathematics needed to follow the rest of the app, each idea with a
live example rather than a paragraph of prose.

1. **Vectors, matrices and matrix multiplication** — an editable 4×2 weight
   matrix times a 2×1 input, with all four dot products written out term by term.
2. **What a neuron computes** — sliders for the weights, bias and inputs; the
   weighted sum and the activation are recomputed as you drag, with the operating
   point marked on the activation curve.
3. **Activation functions and their derivatives** — all four activations, each
   plotted beside its own derivative, with a shared cursor you can drag into the
   saturated tails to watch the derivative collapse toward zero. Each card checks
   its analytic derivative against a finite difference on screen.
4. **Partial derivatives and the chain rule** — first partial derivatives of
   `f(x, y) = x²y + y³` with tangent-line slices; then the chain rule applied to
   `z = wx + b → a = σ(z) → L = (a − y)²`, a neural network in miniature, with
   the three-factor product verified numerically; then the multivariable
   sum-over-paths form used by the real backward pass.
5. **Loss functions** — cross-entropy and squared error plotted against the
   prediction for both labels, with `∂L/∂ŷ` worked out at the current point.
6. **The gradient** — a contour map of `f(x, y) = x² + 3y²` with the ascent and
   descent arrows drawn at a point you can move, plus a button to roll a few
   descent steps downhill and leave a trail.

### Forward Pass

- **Stage-by-stage trace** of one data point through the network: inputs →
  hidden pre-activations → hidden activations → output pre-activation →
  prediction → loss. Step manually or animate the whole pass. The arithmetic for
  the current stage is written out underneath, with the activation derivative
  noted at each hidden unit — it is about to be needed.
- **Network diagram** with the live value at every node (pre-activation below,
  post-activation inside), edge thickness proportional to |weight| and edge
  colour by sign. Hovering or clicking any edge selects it for inspection on the
  Backpropagation tab.
- **Decision boundary** over the whole input plane, with the ŷ = 0.5 contour
  traced by marching squares. Click anywhere to trace the nearest point.
- **Loss vs epoch**, recorded throughout training, with accuracy overlaid.

### Backpropagation

The heart of the app. The full chain rule for the selected data point is broken
into **twelve steps**, and each one shows the same three things side by side:
the symbolic expression, the same expression with the current numbers
substituted in, and the resulting value.

```
 1  ∂L/∂ŷ            the loss and its derivative w.r.t. the prediction
 2  ∂ŷ/∂z(2)         through the output sigmoid
 3  ∂L/∂z(2)         the product — and the exact cancellation to ŷ − y
 4  ∂L/∂W(2)         gradients of the four output weights
 5  ∂L/∂b(2)         gradient of the output bias
 6  ∂L/∂a(1)         pushed back through the output weights
 7  ∂a(1)/∂z(1)      through the hidden activation's derivative
 8  ∂L/∂z(1)         the four hidden error signals
 9  ∂L/∂W(1)         gradients of all eight hidden weights
10  ∂L/∂b(1)         gradients of the four hidden biases
11  ∂L/∂W(1)ⱼₖ       the whole five-factor product at once, unabbreviated
12  ∂L/∂θ            from one example to the batch average
```

Alongside the walkthrough:

- **Full derivation for any single parameter** — every factor the chain rule
  produces, listed with its rule, its value, and the running product, so you can
  watch the number accumulate. Hover or click any edge in the diagram, or any row
  in the ranking, to change which parameter is expanded.
- **Verify against a finite difference** — perturbs that one parameter by
  h = 10⁻⁵, re-runs the forward pass, and compares `[L(θ+h) − L(θ−h)] / 2h`
  against what backpropagation computed.
- **Largest gradients** — all 17 parameters ranked by |gradient|, for this point
  or averaged over the full batch, so it is obvious which weights are about to
  move furthest.
- **Diagram in gradient mode** — the same picture with thickness and colour
  encoding ∂L/∂w instead of w, and the edges relevant to the current step
  highlighted.

### Gradient Descent

- **The update rule**, `w := w − η · ∂L/∂w`, with the current weight, the
  learning rate and the gradient all shown as live numbers being substituted in.
- **Learning-rate slider** on a log scale from 0.001 to 30, with four presets
  (far too small, a good default, too large, wildly too large). Everything —
  the update-rule numbers, the whole parameter table, and the previewed next
  steps on the loss surface — recomputes as you drag, with no reset needed.
- **Step controls** that separate the three phases: *step forward* (run the
  forward pass), *step backward* (compute and freeze the gradients), *update
  weights* (apply exactly those frozen gradients). Plus one epoch at a time, or
  a pausable continuous run.
- **Live diagnosis** of what the loss curve is doing — diverging, oscillating,
  converging, or crawling — so that deliberately setting the learning rate too
  high or too low is explained at the moment it happens rather than in a caption.
- **The loss surface**, a 2-D slice through the 17-dimensional surface: two
  parameters you choose vary over a grid while the other fifteen stay where they
  are, and the mean loss is evaluated at every grid point. Drawn on it are the
  trail of every step taken so far, the actual step the current learning rate is
  about to take (green), the pure descent direction (dashed), and a preview of
  the next six steps (amber). Both axes carry the same units-per-pixel, so the
  gradient arrow really is perpendicular to the contour it sits on.
- **Every parameter, this update** — all 17 rows showing the current value, its
  gradient, the step `−η · ∂L/∂w`, and the value it is about to become.
- **Gradient source** switch: full batch (the gradient averaged over all 200
  points) or single-point SGD (the gradient of the one example being traced on
  the other tabs).

## Code layout

The mathematics is completely separate from the rendering, so it can be read,
tested and reused on its own. Nothing in `src/math/` touches the DOM.

```
src/
  math/
    linalg.js        matrix/vector primitives, written out explicitly
    activations.js   the four activations, their derivatives, and display strings
    network.js       forward pass, backward pass, gradient descent, gradient checking
    dataset.js       two moons / spiral / blobs, plus normalisation
    rng.js           seeded PRNG, so every "reset" is reproducible
  ui/
    dom.js           DOM helpers, number formatting, colour scales
    plots.js         canvas plotting: axes, lines, scatter, fields, marching squares
    netdiagram.js    the SVG node-and-edge diagram
    basics.js        \
    forward.js        |  one module per tab
    backprop.js       |
    gradient-descent.js /
  store.js           shared state, training orchestration, convergence diagnosis
  main.js            app shell and tab routing
tests/
  run-tests.mjs      the verification suite
```

All four tabs read and write the same store, so the network, the selected data
point, the learning rate and the training history carry over as you move between
them.

### Adding an activation function

Add an entry to `ACTIVATIONS` in `src/math/activations.js` with `f`, `df`, the
display strings, and a `derivDetail(z)` that returns the derivative worked out
with a number substituted in, then add its id to `ACTIVATION_ORDER`. It will
appear in the dropdown, in the Basics gallery, and in the backprop walkthrough
with no other changes.

## Verifying the math

`tests/run-tests.mjs` runs 56 checks. The important ones compare **every one of
the 17 analytic gradients against a central finite difference of the loss**, for
all four activations, both for a single example and for the full 200-point
dataset. If backpropagation were wrong anywhere, these would catch it.

The suite also checks the forward pass against arithmetic done by hand, the
exact cancellation `∂L/∂ŷ · ∂ŷ/∂z(2) = ŷ − y`, that the unabbreviated
five-factor product shown in the UI equals the gradient backprop actually
computes, that a small step lowers the loss while an enormous one sends it
climbing, and that the fast loss path used by the surface renderer is bit-identical
to the readable reference implementation.

Two details worth knowing about the gradient checks:

- Gradients that are genuinely near zero are compared by **absolute** rather than
  relative error. A gradient of 7×10⁻⁶ matched to 2×10⁻¹¹ absolute is agreement,
  even though the relative error looks large.
- ReLU and leaky ReLU are only *piecewise* differentiable. If the probe step `h`
  is larger than the closest approach of any pre-activation to the kink at z = 0,
  the finite difference straddles the kink and measures a slope the derivative
  never claimed to have. The suite therefore picks `h` below that distance, which
  lets the tolerance stay as tight as it is for the smooth activations.

## Browser support

Any current browser. Uses ES modules, `ResizeObserver`-free layout, canvas 2D and
inline SVG; no polyfills and no network requests after the initial load.
