// mentats showcase: renders everything from data/*.json.
// Each JSON file carries a `placeholder` flag; when true the section shows a
// "placeholder data" badge so un-measured numbers are never mistaken for real ones.

const DATA_FILES = ["models", "kernels", "history", "reference", "stats"];

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, html = "") => {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  node.innerHTML = html;
  return node;
};

const fmtMs = (ns) => (ns == null ? "n/a" : (ns / 1e6).toFixed(ns < 1e7 ? 2 : 1));
const fmtTime = (ns) => {
  if (ns == null) return "n/a";
  if (ns < 1e3) return `${ns.toFixed(0)} ns`;
  if (ns < 1e6) return `${(ns / 1e3).toFixed(1)} µs`;
  if (ns < 1e9) return `${(ns / 1e6).toFixed(2)} ms`;
  return `${(ns / 1e9).toFixed(2)} s`;
};

async function loadAll() {
  const entries = await Promise.all(
    DATA_FILES.map(async (name) => {
      try {
        const res = await fetch(`data/${name}.json`, { cache: "no-cache" });
        if (!res.ok) throw new Error(res.statusText);
        return [name, await res.json()];
      } catch (err) {
        console.warn(`Failed to load data/${name}.json`, err);
        return [name, null];
      }
    })
  );
  return Object.fromEntries(entries);
}

function markPlaceholders(data) {
  document.querySelectorAll(".badge[data-src]").forEach((badge) => {
    const d = data[badge.dataset.src];
    if (!d) badge.textContent = "data unavailable";
    else if (d.placeholder) badge.textContent = "placeholder data";
  });
}

function chartDefaults() {
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--border");
  Chart.defaults.font.family = css("--mono");
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
}

const palette = () => [css("--accent"), "#8fb8de", css("--accent-2"), "#9ccf8f", "#c79ae0", "#e07a7a"];

// ---------- Hero ----------
function renderHero({ models, stats, kernels }) {
  const best = models?.models?.reduce((a, m) => (m.test_accuracy > (a?.test_accuracy ?? 0) ? m : a), null);
  const matmul = kernels?.benches?.filter(
    (b) => (b.group.toLowerCase().includes("matmul") || b.group.toLowerCase().includes("multiplication")) && b.flops
  ) ?? [];
  const peak = matmul.length ? Math.max(...matmul.map((b) => b.flops / b.median_ns)) : null;
  const items = [
    [best ? `${best.test_accuracy.toFixed(2)}%` : "n/a", `${best?.dataset ?? ""} test accuracy`],
    [stats?.runtime_dependencies ?? "1", "runtime dependency"],
    [stats?.tests ?? "n/a", "tests"],
    [peak ? `${peak.toFixed(1)}${kernels.placeholder ? "*" : ""}` : "n/a", "peak matmul GFLOP/s"],
  ];
  const root = $("#hero-stats");
  items.forEach(([v, k]) => root.append(el("div", { class: "hero-stat" }, `<div class="v">${v}</div><div class="k">${k}</div>`)));
}

// ---------- Model results ----------
function renderModels({ models }) {
  if (!models) return;
  const cards = $("#model-cards");
  models.models.forEach((m) => {
    cards.append(
      el("div", { class: "model-card" },
        `<h3>${m.name}</h3>
         <div class="acc">${m.test_accuracy != null ? m.test_accuracy.toFixed(2) + "%" : "n/a"}</div>
         <div class="meta">${m.dataset} test accuracy · ${m.epochs.length} epochs · ${m.optimiser}</div>
         <code>${m.architecture}</code>`)
    );
  });

  if (typeof Chart === "undefined") return;
  const colors = palette();
  const maxEpochs = Math.max(...models.models.map((m) => m.epochs.length));
  const labels = Array.from({ length: maxEpochs }, (_, i) => `epoch ${i}`);
  const ds = (key) =>
    models.models.map((m, i) => ({
      label: m.name,
      data: m.epochs.map((e) => e[key]),
      borderColor: colors[i % colors.length],
      backgroundColor: colors[i % colors.length],
      tension: 0.3,
    }));
  new Chart($("#chart-loss"), { type: "line", data: { labels, datasets: ds("loss") } });
  new Chart($("#chart-acc"), { type: "line", data: { labels, datasets: ds("train_accuracy") } });
}

// ---------- Kernels ----------
function renderKernels({ kernels }) {
  if (!kernels) return;
  const hw = kernels.hardware || {};
  $("#hardware").textContent =
    `Criterion medians · ${hw.cpu ?? "?"} · ${hw.os ?? "?"} · rustc ${hw.rustc ?? "?"} · ${hw.threads ?? 1} thread` +
    (kernels.generated_at ? ` · measured ${kernels.generated_at}` : "");

  const accent = css("--accent");
  const matmul = kernels.benches.filter(
    (b) => (b.group.toLowerCase().includes("matmul") || b.group.toLowerCase().includes("multiplication")) && b.flops
  );
  new Chart($("#chart-matmul"), {
    type: "bar",
    data: {
      labels: matmul.map((b) => b.label),
      datasets: [{ label: "GFLOP/s", data: matmul.map((b) => Number((b.flops / b.median_ns).toFixed(2))), backgroundColor: accent }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "GFLOP/s" } } },
    },
  });

  const layer = kernels.benches.filter(
    (b) => !b.group.toLowerCase().includes("matmul") && !b.group.toLowerCase().includes("multiplication")
  );
  new Chart($("#chart-kernels"), {
    type: "bar",
    data: {
      labels: layer.map((b) => b.label),
      datasets: [{ label: "ms", data: layer.map((b) => b.median_ns / 1e6), backgroundColor: "#8fb8de" }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } } },
  });

  const table = $("#kernel-table");
  table.innerHTML = `<thead><tr><th>benchmark</th><th>group</th><th class="r">median</th><th class="r">GFLOP/s</th></tr></thead>`;
  const body = el("tbody");
  kernels.benches.forEach((b) => {
    body.append(el("tr", {},
      `<td>${b.label}</td><td class="muted">${b.group}</td><td class="r">${fmtTime(b.median_ns)}</td>` +
      `<td class="r">${b.flops ? (b.flops / b.median_ns).toFixed(2) : "–"}</td>`));
  });
  table.append(body);
}

// ---------- History ----------
function renderHistory({ history }) {
  if (!history) return;
  const colors = palette();
  const labels = history.entries.map((e) => e.label);
  const datasets = history.tracked.map((id, i) => {
    const series = history.entries.map((e) => e.results[id] ?? null);
    const base = series.find((v) => v != null);
    return {
      label: id,
      data: series.map((v) => (v == null || !base ? null : base / v)),
      borderColor: colors[i % colors.length],
      backgroundColor: colors[i % colors.length],
      spanGaps: false,
      tension: 0.2,
    };
  });
  new Chart($("#chart-history"), {
    type: "line",
    data: { labels, datasets },
    options: {
      scales: { y: { title: { display: true, text: "speedup (×)" }, beginAtZero: true } },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y?.toFixed(2)}×` } } },
    },
  });

  const headline = history.tracked[0];
  const first = history.entries.find((e) => e.results[headline] != null)?.results[headline];
  const tl = $("#timeline");
  history.entries.forEach((e) => {
    const v = e.results[headline];
    const sp = v && first ? `${headline}: ${(first / v).toFixed(2)}×` : "";
    const pr = e.pr ? `<a class="pr" href="https://github.com/Sleishm4n/mentats/pull/${e.pr}">#${e.pr}</a>` : `<span class="pr">${e.label}</span>`;
    tl.append(el("li", {}, `${pr}<span>${e.title}</span><span class="sp">${sp}</span>`));
  });
}

// ---------- Reference ----------
function renderReference({ reference }) {
  if (!reference) return;
  const r = reference.reference;
  $("#reference-note").textContent =
    `Same shapes, same architectures. ${r.framework} ${r.version} limited to ${r.threads} thread. ` +
    `100% would match PyTorch's MKL/oneDNN-backed kernels; mentats uses no BLAS at all.`;
  const k = reference.kernels;
  new Chart($("#chart-reference"), {
    type: "bar",
    data: {
      labels: k.map((x) => x.label),
      datasets: [{
        label: "% of PyTorch speed",
        data: k.map((x) => (x.reference_ns / x.mentats_ns) * 100),
        backgroundColor: css("--accent"),
      }],
    },
    options: {
      indexAxis: "y",
      scales: { x: { beginAtZero: true, suggestedMax: 100 } },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { afterLabel: (c) => `mentats ${fmtTime(k[c.dataIndex].mentats_ns)} vs PyTorch ${fmtTime(k[c.dataIndex].reference_ns)}` } },
      },
    },
  });
}

// ---------- Codebase stats ----------
function renderStats({ stats }) {
  if (!stats) return;
  const items = [
    [stats.src_lines?.toLocaleString(), "lines of Rust in src/"],
    [stats.tests, "tests"],
    [stats.coverage_pct != null ? `${stats.coverage_pct.toFixed(1)}%` : "n/a", "line coverage"],
    [stats.examples, "runnable examples"],
    [stats.runtime_dependencies, "runtime dependency"],
    [stats.version, "crate version"],
  ];
  const grid = $("#stat-grid");
  items.forEach(([v, k]) => grid.append(el("div", { class: "stat" }, `<div class="v">${v ?? "n/a"}</div><div class="k">${k}</div>`)));

  const lists = $("#stat-lists");
  [["Layers", stats.layers], ["Losses", stats.losses], ["Optimisers", stats.optimisers]].forEach(([title, xs]) => {
    if (!xs) return;
    lists.append(el("div", { class: "card" }, `<h3>${title}</h3><ul>${xs.map((x) => `<li>${x}</li>`).join("")}</ul>`));
  });
}

(async function main() {
  const data = await loadAll();
  markPlaceholders(data);
  renderHero(data);
  renderStats(data);
  const hasCharts = typeof Chart !== "undefined";
  if (hasCharts) chartDefaults();
  else console.warn("Chart.js failed to load; charts disabled.");
  renderModels(data); // cards render regardless; charts only if Chart.js loaded
  if (!hasCharts) return;
  renderKernels(data);
  renderHistory(data);
  renderReference(data);
})();
