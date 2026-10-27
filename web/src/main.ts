import { encodeLabels, guessLabelColumn, parseTable, type Table } from "./csv";
import { CATEGORICAL, OTHER, RAMP_CSS, categoricalColors, sequentialColors, uniformColors } from "./colors";
import { demoTable } from "./demo";
import type { FromWorker, Quality, RunRequest, ToWorker } from "./protocol";
import { Scatter } from "./scatter";

interface Result {
  id: number;
  title: string;
  embedding: Float32Array | null;
  quality: Quality | null;
  ms: number;
  done: boolean;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const scatter = new Scatter($("plot"));
const ui = {
  file: $<HTMLInputElement>("file"),
  drop: $<HTMLLabelElement>("drop"),
  demo: $<HTMLButtonElement>("demo"),
  summary: $<HTMLDivElement>("data-summary"),
  options: $<HTMLDivElement>("data-options"),
  labelCol: $<HTMLSelectElement>("label-col"),
  standardize: $<HTMLInputElement>("standardize"),
  featureList: $<HTMLDivElement>("feature-list"),
  embed: $<HTMLElement>("embed-section"),
  gamma: $<HTMLInputElement>("gamma"),
  gammaOut: $<HTMLOutputElement>("gamma-out"),
  runScaleRank: $<HTMLButtonElement>("run-scalerank"),
  runUmap: $<HTMLButtonElement>("run-umap"),
  runTsne: $<HTMLButtonElement>("run-tsne"),
  progressSection: $<HTMLElement>("progress-section"),
  progressBar: $<HTMLDivElement>("progress-bar"),
  status: $<HTMLSpanElement>("status"),
  cancel: $<HTMLButtonElement>("cancel"),
  qualitySection: $<HTMLElement>("quality-section"),
  qualityBody: $<HTMLTableSectionElement>("quality-body"),
  tabs: $<HTMLDivElement>("tabs"),
  colorMode: $<HTMLSelectElement>("color-mode"),
  download: $<HTMLButtonElement>("download"),
  empty: $<HTMLDivElement>("empty"),
  legend: $<HTMLDivElement>("legend"),
};

const state = {
  table: null as Table | null,
  labelCol: -1,
  labels: null as { codes: Uint32Array; names: string[] } | null,
  features: new Set<number>(),
  results: [] as Result[],
  active: -1,
  running: null as Result | null,
  nextId: 1,
  dataVersion: 0,
};

// ---- worker ----------------------------------------------------------------

let worker: Worker;
function startWorker() {
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent<FromWorker>) => onWorker(e.data);
  worker.onerror = (e) => showError(e.message || "The compute worker failed.");
}
const send = (msg: ToWorker, transfer: Transferable[] = []) => worker.postMessage(msg, transfer);

function onWorker(msg: FromWorker) {
  switch (msg.type) {
    case "ready":
      break;
    case "loaded": {
      const t = state.table!;
      const parts = [`${fmtInt(t.n)} rows`, `${state.features.size} numeric columns used`];
      if (msg.dropped > 0) parts.push(`${msg.dropped} constant dropped`);
      if (msg.dim < state.features.size - msg.dropped)
        parts.push(`reduced to ${msg.dim} principal components (${(100 * msg.explained).toFixed(0)}% of the variance)`);
      ui.summary.textContent = parts.join(" · ");
      break;
    }
    case "stage":
      if (state.running?.id === msg.id) ui.status.textContent = msg.text;
      break;
    case "progress": {
      const r = state.results.find((x) => x.id === msg.id);
      if (!r) return;
      const first = r.embedding === null;
      r.embedding = msg.embedding;
      ui.progressBar.style.width = `${(100 * msg.step) / msg.total}%`;
      ui.status.textContent = `Optimising… ${msg.step} / ${msg.total}`;
      if (state.active === r.id) showResult(r, first);
      break;
    }
    case "done": {
      const r = state.results.find((x) => x.id === msg.id);
      if (!r) return;
      r.embedding = msg.embedding;
      r.quality = msg.quality;
      r.ms = msg.ms + msg.knnMs;
      r.done = true;
      state.running = null;
      setBusy(false);
      renderQuality();
      renderTabs();
      if (state.active === r.id) showResult(r, false);
      break;
    }
    case "error":
      showError(msg.message);
      break;
  }
}

function showError(text: string) {
  state.running = null;
  setBusy(false);
  ui.progressSection.hidden = false;
  ui.progressBar.style.width = "0";
  ui.status.textContent = `Error: ${text}`;
  ui.cancel.hidden = true;
}

// ---- data ------------------------------------------------------------------

function setTable(t: Table, source: string) {
  state.table = t;
  state.results = [];
  state.active = -1;
  state.labelCol = guessLabelColumn(t);
  ui.labelCol.replaceChildren(
    new Option("(none)", "-1"),
    ...t.columns.map((c, j) => new Option(`${c.name}${c.numeric ? "" : " (text)"}`, String(j))),
  );
  ui.labelCol.value = String(state.labelCol);
  state.features = new Set(t.columns.flatMap((c, j) => (c.numeric && j !== state.labelCol && !isRowIndex(c.values) ? [j] : [])));
  renderFeatureList();
  ui.summary.hidden = false;
  ui.summary.textContent = `${source}: reading…`;
  ui.options.hidden = false;
  scatter.clear();
  ui.empty.hidden = false;
  ui.empty.textContent = "Data loaded. Compute a map to see it here.";
  ui.legend.hidden = true;
  renderTabs();
  renderQuality();
  updateLabels();
  loadIntoWorker();
}

/** A column counting rows (1, 2, 3, ... or 0, 1, 2, ...) is an identifier, not a feature. */
function isRowIndex(v: Float32Array) {
  for (let i = 1; i < v.length; i++) if (v[i] - v[i - 1] !== 1) return false;
  return v.length > 2;
}

function renderFeatureList() {
  const t = state.table!;
  ui.featureList.replaceChildren(
    ...t.columns.flatMap((c, j) => {
      if (!c.numeric) return [];
      const label = document.createElement("label");
      label.className = "check";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = state.features.has(j);
      box.disabled = j === state.labelCol;
      box.addEventListener("change", () => {
        if (box.checked) state.features.add(j);
        else state.features.delete(j);
        dataChanged();
      });
      label.append(box, ` ${c.name}`);
      return [label];
    }),
  );
}

function updateLabels() {
  const t = state.table!;
  state.labels = state.labelCol >= 0 ? encodeLabels(t.columns[state.labelCol], t.n) : null;
}

function loadIntoWorker() {
  const t = state.table!;
  const cols = [...state.features].sort((a, b) => a - b);
  const enough = cols.length > 0 && t.n >= 10;
  ui.embed.classList.toggle("disabled", !enough);
  if (!enough) {
    ui.summary.textContent = "Need at least 10 rows and one numeric column.";
    return;
  }
  const d = cols.length;
  const raw = new Float32Array(t.n * d);
  cols.forEach((j, c) => {
    const v = t.columns[j].values;
    for (let i = 0; i < t.n; i++) raw[i * d + c] = v[i];
  });
  state.dataVersion++;
  send(
    { type: "load", raw, n: t.n, d, standardize: ui.standardize.checked, labels: state.labels?.codes.slice() ?? null },
    [raw.buffer],
  );
}

/** Columns or scaling changed: earlier maps no longer match the data. */
function dataChanged() {
  state.results = [];
  state.active = -1;
  scatter.clear();
  ui.empty.hidden = false;
  ui.legend.hidden = true;
  renderTabs();
  renderQuality();
  loadIntoWorker();
}

async function readFile(f: File) {
  try {
    ui.summary.hidden = false;
    ui.summary.textContent = `Reading ${f.name}…`;
    const t = parseTable(await f.text());
    setTable(t, f.name);
  } catch (err) {
    ui.summary.textContent = `Could not read ${f.name}: ${(err as Error).message}`;
  }
}

// ---- runs ------------------------------------------------------------------

function run(req: RunRequest, title: string) {
  if (state.running || !state.table) return;
  const r: Result = { id: state.nextId++, title, embedding: null, quality: null, ms: 0, done: false };
  state.results.push(r);
  state.running = r;
  state.active = r.id;
  setBusy(true);
  renderTabs();
  send({ type: "run", id: r.id, seed: 1, ...req });
}

function setBusy(busy: boolean) {
  for (const b of [ui.runScaleRank, ui.runUmap, ui.runTsne]) b.disabled = busy;
  ui.progressSection.hidden = !busy;
  ui.cancel.hidden = !busy;
  if (busy) {
    ui.progressBar.style.width = "0";
    ui.status.textContent = "Starting…";
  }
}

function cancel() {
  if (!state.running) return;
  worker.terminate();
  state.results = state.results.filter((r) => r !== state.running);
  state.running = null;
  state.active = state.results.at(-1)?.id ?? -1;
  setBusy(false);
  startWorker();
  loadIntoWorker();
  renderTabs();
  const r = state.results.find((x) => x.id === state.active);
  if (r) showResult(r, true);
  else scatter.clear();
}

// ---- view ------------------------------------------------------------------

function pointSize(n: number) {
  return Math.min(7, Math.max(1.5, 3 * Math.pow(2000 / n, 0.25)));
}

function showResult(r: Result, refit: boolean) {
  if (!r.embedding) return;
  ui.empty.hidden = true;
  scatter.pointSize = pointSize(r.embedding.length / 2);
  scatter.setPositions(r.embedding, refit);
  recolor(r);
  ui.download.disabled = !r.done;
}

function recolor(r: Result) {
  const n = r.embedding!.length / 2;
  const alpha = n > 20000 ? 150 : 210;
  const mode = ui.colorMode.value;
  if (mode === "overlap" && r.quality) {
    scatter.setColors(sequentialColors(r.quality.overlap10, alpha));
    ui.legend.hidden = false;
    ui.legend.innerHTML = `<div>Share of each point's 10 nearest neighbours kept</div>
      <div class="ramp" style="background:${RAMP_CSS}"></div><div class="row-between"><span>0</span><span>1</span></div>`;
  } else if (state.labels) {
    scatter.setColors(categoricalColors(state.labels.codes, alpha));
    renderCategoryLegend();
  } else {
    scatter.setColors(uniformColors(n, CATEGORICAL[0], alpha));
    ui.legend.hidden = true;
  }
}

function renderCategoryLegend() {
  const names = state.labels!.names;
  const items = names.slice(0, CATEGORICAL.length).map((s, k) => [s, CATEGORICAL[k]] as const);
  if (names.length > CATEGORICAL.length) items.push([`${names.length - CATEGORICAL.length} more`, OTHER]);
  ui.legend.hidden = false;
  ui.legend.replaceChildren(
    ...items.map(([s, c]) => {
      const row = document.createElement("div");
      row.className = "legend-item";
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = c;
      row.append(sw, s);
      return row;
    }),
  );
}

function renderTabs() {
  ui.tabs.replaceChildren(
    ...state.results.map((r) => {
      const b = document.createElement("button");
      b.type = "button";
      b.role = "tab";
      b.className = "tab";
      b.setAttribute("aria-selected", String(r.id === state.active));
      b.textContent = r.done ? r.title : `${r.title} …`;
      b.addEventListener("click", () => {
        state.active = r.id;
        renderTabs();
        showResult(r, true);
      });
      return b;
    }),
  );
}

function renderQuality() {
  const done = state.results.filter((r) => r.quality);
  ui.qualitySection.hidden = done.length === 0;
  ui.qualityBody.replaceChildren(
    ...done.map((r) => {
      const q = r.quality!;
      const tr = document.createElement("tr");
      for (const v of [r.title, q.rnx10.toFixed(2), q.auc.toFixed(2), q.distSpearman.toFixed(2), fmtTime(r.ms)]) {
        const td = document.createElement("td");
        td.textContent = v;
        tr.append(td);
      }
      return tr;
    }),
  );
}

function download() {
  const r = state.results.find((x) => x.id === state.active);
  if (!r?.embedding || !state.table) return;
  const lab = state.labelCol >= 0 ? state.table.columns[state.labelCol] : null;
  const head = lab ? `x,y,${csvField(lab.name)}\n` : "x,y\n";
  const lines: string[] = [head];
  for (let i = 0; i < state.table.n; i++) {
    const l = lab ? `,${csvField(lab.text ? lab.text[i] : String(lab.values[i]))}` : "";
    lines.push(`${r.embedding[2 * i]},${r.embedding[2 * i + 1]}${l}\n`);
  }
  const url = URL.createObjectURL(new Blob(lines, { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${r.title.replace(/[^\w.-]+/g, "_")}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

const csvField = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
const fmtInt = (v: number) => v.toLocaleString("en-US");
const fmtTime = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

// ---- wiring ----------------------------------------------------------------

ui.file.addEventListener("change", () => ui.file.files?.[0] && readFile(ui.file.files[0]));
ui.drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  ui.drop.classList.add("over");
});
ui.drop.addEventListener("dragleave", () => ui.drop.classList.remove("over"));
ui.drop.addEventListener("drop", (e) => {
  e.preventDefault();
  ui.drop.classList.remove("over");
  const f = e.dataTransfer?.files[0];
  if (f) readFile(f);
});
ui.demo.addEventListener("click", () => setTable(demoTable(), "Demo data"));
ui.labelCol.addEventListener("change", () => {
  const prev = state.labelCol;
  state.labelCol = Number(ui.labelCol.value);
  updateLabels();
  const t = state.table!;
  // a numeric column used for colouring is not also a feature
  const featuresChanged =
    (prev >= 0 && t.columns[prev].numeric) || (state.labelCol >= 0 && t.columns[state.labelCol].numeric);
  if (prev >= 0 && t.columns[prev].numeric) state.features.add(prev);
  if (state.labelCol >= 0) state.features.delete(state.labelCol);
  renderFeatureList();
  if (featuresChanged) dataChanged();
  else {
    const r = state.results.find((x) => x.id === state.active);
    if (r?.embedding) recolor(r);
  }
});
ui.standardize.addEventListener("change", dataChanged);
ui.gamma.addEventListener("input", () => (ui.gammaOut.textContent = Number(ui.gamma.value).toFixed(1)));
ui.runScaleRank.addEventListener("click", () => {
  const gamma = Number(ui.gamma.value);
  run({ method: "scalerank", params: { gamma, nEpochs: 800, negScale: 0 } }, `ScaleRank ${gamma.toFixed(1)}`);
});
ui.runUmap.addEventListener("click", () => run({ method: "umap", params: { nNeighbors: 15, minDist: 0.1 } }, "UMAP"));
ui.runTsne.addEventListener("click", () => run({ method: "tsne", params: { perplexity: 30 } }, "t-SNE"));
ui.cancel.addEventListener("click", cancel);
ui.colorMode.addEventListener("change", () => {
  const r = state.results.find((x) => x.id === state.active);
  if (r?.embedding) recolor(r);
});
ui.download.addEventListener("click", download);

startWorker();
