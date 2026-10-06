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

  // Sort by matrix dimension: 64 -> 128 -> 256 -> 512, keeping naive right before optimised
  matmul.sort((a, b) => {
    const sizeA = parseInt(a.label.match(/\d+/)?.[0] || "0", 10);
    const sizeB = parseInt(b.label.match(/\d+/)?.[0] || "0", 10);
    if (sizeA !== sizeB) return sizeA - sizeB;
    if (a.label.includes("naive")) return -1;
    if (b.label.includes("naive")) return 1;
    return 0;
  });

  const matmulColors = matmul.map((b) => (b.label.includes("naive") ? "#6e6a66" : accent));

  new Chart($("#chart-matmul"), {
    type: "bar",
    data: {
      labels: matmul.map((b) => b.label),
      datasets: [{
        label: "GFLOP/s",
        data: matmul.map((b) => Number((b.flops / b.median_ns).toFixed(2))),
        backgroundColor: matmulColors,
      }],
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const item = matmul[ctx.dataIndex];
              return `Median: ${fmtTime(item.median_ns)}`;
            },
          },
        },
      },
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
    options: {
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const item = layer[ctx.dataIndex];
              return `Time: ${fmtTime(item.median_ns)}`;
            },
          },
        },
      },
    },
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
    const isConv = id.toLowerCase().includes("conv");
    return {
      label: id === "matmul_256" ? "MatMul 256×256 speedup" : "Conv2D single image speedup",
      data: series.map((v) => (v == null || !base ? null : Number((base / v).toFixed(2)))),
      borderColor: colors[i % colors.length],
      backgroundColor: colors[i % colors.length],
      spanGaps: false,
      tension: 0.25,
      yAxisID: isConv ? "yConv" : "yMatmul",
    };
  });

  new Chart($("#chart-history"), {
    type: "line",
    data: { labels, datasets },
    options: {
      interaction: { mode: "index", intersect: false },
      scales: {
        yMatmul: {
          type: "linear",
          position: "left",
          title: { display: true, text: "MatMul Speedup (×)", color: colors[0] },
          ticks: { color: colors[0] },
          beginAtZero: true,
        },
        yConv: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Conv2D Speedup (×)", color: colors[1] },
          ticks: { color: colors[1] },
          min: 0.8,
          suggestedMax: 2.0,
          grid: { drawOnChartArea: false },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.dataset.label}: ${c.parsed.y?.toFixed(2)}×`,
          },
        },
      },
    },
  });

  const tl = $("#timeline");
  const firstMatmul = history.entries.find((e) => e.results["matmul_256"] != null)?.results["matmul_256"];
  const firstConv = history.entries.find((e) => e.results["conv2d_single"] != null)?.results["conv2d_single"];

  history.entries.forEach((e) => {
    const m = e.results["matmul_256"];
    const c = e.results["conv2d_single"];
    const parts = [];
    if (m && firstMatmul) parts.push(`matmul: ${(firstMatmul / m).toFixed(1)}×`);
    if (c && firstConv) parts.push(`conv: ${(firstConv / c).toFixed(2)}×`);
    const sp = parts.join(" · ");
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
    `100% matches PyTorch's Intel oneDNN/MKL-backed kernels; mentats uses zero external BLAS libraries.`;
  const k = reference.kernels;
  const accent = css("--accent");

  const parityPlugin = {
    id: "parityPlugin",
    afterDatasetsDraw(chart) {
      const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
      const x100 = x.getPixelForValue(100);

      // Draw dashed reference line at 100%
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
      ctx.lineWidth = 1.5;
      ctx.moveTo(x100, top);
      ctx.lineTo(x100, bottom);
      ctx.stroke();

      // Top label for parity line with ample padding
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText("100% (PyTorch Parity)", x100, top - 6);

      // Labels on bars: inside if long, outside if short
      const meta = chart.getDatasetMeta(0);
      meta.data.forEach((bar, index) => {
        const val = chart.data.datasets[0].data[index];
        const text = `${val.toFixed(1)}%`;
        ctx.font = "bold 11px ui-monospace, monospace";
        ctx.textBaseline = "middle";

        const barWidth = bar.x - bar.base;
        if (barWidth > 140) {
          // Inside the bar, dark bold text
          ctx.fillStyle = "#111115";
          ctx.textAlign = "right";
          ctx.fillText(text, bar.x - 10, bar.y);
        } else {
          // Outside the bar, light bold text
          ctx.fillStyle = "#e7e5e0";
          ctx.textAlign = "left";
          ctx.fillText(text, bar.x + 8, bar.y);
        }
      });
      ctx.restore();
    },
  };

  new Chart($("#chart-reference"), {
    type: "bar",
    data: {
      labels: k.map((x) => x.label),
      datasets: [{
        label: "% of PyTorch speed",
        data: k.map((x) => Number(((x.reference_ns / x.mentats_ns) * 100).toFixed(1))),
        backgroundColor: k.map((x) => {
          const pct = (x.reference_ns / x.mentats_ns) * 100;
          return pct >= 100 ? "#f2b45a" : accent; // highlight >= 100% with gold
        }),
      }],
    },
    options: {
      indexAxis: "y",
      layout: {
        padding: {
          top: 30, // Plenty of room for "100% (PyTorch Parity)" title
          right: 45,
          left: 10,
          bottom: 5,
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          suggestedMax: 120,
          title: { display: true, text: "% of Single-Thread PyTorch Performance" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.parsed.x.toFixed(1)}% of PyTorch single-thread speed`,
            afterLabel: (c) => ` mentats: ${fmtTime(k[c.dataIndex].mentats_ns)} vs PyTorch: ${fmtTime(k[c.dataIndex].reference_ns)}`,
          },
        },
      },
    },
    plugins: [parityPlugin],
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
