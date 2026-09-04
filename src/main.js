// App shell: the header controls that apply everywhere, the tab bar, and the
// render loop that keeps the active tab in sync with the store.

import { store } from './store.js';
import { el, fmt } from './ui/dom.js';
import { ACTIVATIONS, ACTIVATION_ORDER, getActivation } from './math/activations.js';
import { DATASETS } from './math/dataset.js';
import { createBasicsTab } from './ui/basics.js';
import { createForwardTab } from './ui/forward.js';
import { createBackpropTab } from './ui/backprop.js';
import { createGradientDescentTab } from './ui/gradient-descent.js';

const TABS = [
  { id: 'basics', label: 'Basics', hint: 'the maths you need first', factory: createBasicsTab },
  { id: 'forward', label: 'Forward Pass', hint: 'values flowing through', factory: createForwardTab },
  { id: 'backprop', label: 'Backpropagation', hint: 'the chain rule, term by term', factory: createBackpropTab },
  { id: 'descent', label: 'Gradient Descent', hint: 'the update rule and the loss surface', factory: createGradientDescentTab },
];

const app = document.getElementById('app');
const refs = {};
let active = 'basics';
const tabs = {};

app.replaceChildren(buildHeader(), buildTabBar(), buildPanels());
wire();
selectTab('basics');
store.subscribe(onStoreChange);
renderHeader();

// ---------------------------------------------------------------------------

function buildHeader() {
  return el('header', { class: 'appbar' }, [
    el('div', { class: 'brand' }, [
      el('h1', { text: 'Neural Network Math Visualizer' }),
      el('p', { text: '2 inputs → 4 hidden neurons → 1 sigmoid output, all arithmetic done from scratch' }),
    ]),
    el('div', { class: 'appbar-controls' }, [
      el('label', { class: 'field' }, [
        el('span', { class: 'flabel', text: 'hidden activation f' }),
        refs.activation = el('select', { class: 'select' },
          ACTIVATION_ORDER.map((id) => el('option', { value: id, text: ACTIVATIONS[id].name }))
        ),
      ]),
      refs.derivChip = el('div', { class: 'deriv-chip' }),
      el('label', { class: 'field' }, [
        el('span', { class: 'flabel', text: 'dataset' }),
        refs.dataset = el('select', { class: 'select' },
          Object.values(DATASETS).map((d) => el('option', { value: d.id, text: d.name }))
        ),
      ]),
      el('div', { class: 'field' }, [
        el('span', { class: 'flabel', text: 'training' }),
        el('div', { class: 'row gap' }, [
          refs.run = el('button', { class: 'btn primary sm', text: '▶ Run' }),
          refs.epoch = el('button', { class: 'btn sm', text: '+1 epoch' }),
          refs.reset = el('button', { class: 'btn ghost sm', text: '⟲ Reset' }),
        ]),
      ]),
      refs.status = el('div', { class: 'status' }),
    ]),
  ]);
}

function buildTabBar() {
  refs.tabbar = el('nav', { class: 'tabbar' },
    TABS.map((t) => {
      const b = el('button', { class: 'tab', dataset: { tab: t.id } }, [
        el('b', { text: t.label }),
        el('span', { text: t.hint }),
      ]);
      b.addEventListener('click', () => selectTab(t.id));
      return b;
    })
  );
  return refs.tabbar;
}

function buildPanels() {
  refs.panels = el('main', { class: 'panels' },
    TABS.map((t) => el('section', { class: 'panel', id: `panel-${t.id}` }))
  );
  return refs.panels;
}

function wire() {
  refs.activation.value = store.state.activation;
  refs.dataset.value = store.state.datasetId;

  refs.activation.addEventListener('change', () => store.setActivation(refs.activation.value));
  refs.dataset.addEventListener('change', () => store.setDataset(refs.dataset.value));
  refs.run.addEventListener('click', () => store.toggleRunning());
  refs.epoch.addEventListener('click', () => store.runEpoch());
  refs.reset.addEventListener('click', () => store.resetWeights());

  let t = null;
  window.addEventListener('resize', () => {
    clearTimeout(t);
    t = setTimeout(() => tabs[active]?.render(), 120);
  });
}

function selectTab(id) {
  active = id;
  for (const b of refs.tabbar.children) b.classList.toggle('on', b.dataset.tab === id);
  for (const p of refs.panels.children) p.classList.toggle('on', p.id === `panel-${id}`);

  if (!tabs[id]) {
    const def = TABS.find((t) => t.id === id);
    tabs[id] = def.factory(document.getElementById(`panel-${id}`), store);
  }
  tabs[id].render();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function onStoreChange() {
  renderHeader();
  // Only the visible tab renders; the others catch up when they are shown,
  // reading the same shared state, so nothing is lost by switching.
  tabs[active]?.render();
}

function renderHeader() {
  const s = store.state;
  const act = getActivation(s.activation);
  refs.activation.value = s.activation;
  refs.dataset.value = s.datasetId;
  refs.run.textContent = s.running ? '❚❚ Pause' : '▶ Run';
  refs.run.classList.toggle('danger', s.running);
  refs.derivChip.innerHTML = `
    <span class="dc-f mono">${act.tex}</span>
    <span class="dc-d mono">${act.dtex}</span>
    <span class="dc-note">the derivative backprop multiplies by</span>`;
  refs.status.innerHTML = `
    <span>epoch <b>${Math.round(s.epoch)}</b></span>
    <span>loss <b>${fmt(s.loss, 4)}</b></span>
    <span>acc <b>${(s.acc * 100).toFixed(1)}%</b></span>
    <span>η <b>${fmt(s.learningRate, 3)}</b></span>`;
}
