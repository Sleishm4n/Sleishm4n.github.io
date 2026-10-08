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

  const classifiers = models.models.filter((m) => m.task === "classification" || m.test_accuracy != null);
  const generative = models.models.filter((m) => m.task === "generative" || (m.task !== "classification" && m.test_accuracy == null));

  models.models.forEach((m) => {
    let mainMetricHtml = "";
    let metaHtml = "";

    if (m.task === "generative" || m.test_accuracy == null) {
      const last = m.epochs[m.epochs.length - 1];
      const recon = last?.recon_loss != null ? last.recon_loss.toFixed(4) : (last?.loss != null ? last.loss.toFixed(4) : "–");
      const kl = last?.kl_loss != null ? ` · KL: ${last.kl_loss.toFixed(4)}` : "";
      mainMetricHtml = `<div class="acc">${recon} <span style="font-size:1.1rem;color:var(--muted);font-weight:normal;">BCE</span></div>`;
      metaHtml = `<div class="meta">${m.dataset} reconstruction loss${kl} · ${m.epochs.length} epochs · ${m.optimiser}</div>`;
    } else {
      mainMetricHtml = `<div class="acc">${m.test_accuracy.toFixed(2)}%</div>`;
      metaHtml = `<div class="meta">${m.dataset} test accuracy · ${m.epochs.length} epochs · ${m.optimiser}</div>`;
    }

    cards.append(
      el("div", { class: "model-card" },
        `<h3>${m.name}</h3>
         ${mainMetricHtml}
         ${metaHtml}
         <code>${m.architecture}</code>`)
    );
  });

  if (typeof Chart === "undefined") return;
  const colors = palette();

  // 1. Classification charts (only classifiers)
  if (classifiers.length > 0) {
    const classEpochs = Math.max(...classifiers.map((m) => m.epochs.length));
    const classLabels = Array.from({ length: classEpochs }, (_, i) => `epoch ${i}`);
    const dsClass = (key) =>
      classifiers.map((m, i) => ({
        label: m.name,
        data: m.epochs.map((e) => e[key]),
        borderColor: colors[i % colors.length],
        backgroundColor: colors[i % colors.length],
        tension: 0.3,
      }));

    new Chart($("#chart-loss"), {
      type: "line",
      data: { labels: classLabels, datasets: dsClass("loss") },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: { y: { beginAtZero: false, title: { display: true, text: "Cross-Entropy Loss" } } },
      },
    });

    new Chart($("#chart-acc"), {
      type: "line",
      data: { labels: classLabels, datasets: dsClass("train_accuracy") },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: { y: { title: { display: true, text: "Train Accuracy (%)" } } },
      },
    });
  }

  // 2. Generative charts (CVAE training dynamics)
  if (generative.length > 0) {
    const cvaeSec = $("#generative-section");
    if (cvaeSec) cvaeSec.style.display = "block";

    const cvae = generative[0];
    const cvaeLabels = cvae.epochs.map((e) => `epoch ${e.epoch}`);

    // Reconstruction loss curve
    new Chart($("#chart-cvae-recon"), {
      type: "line",
      data: {
        labels: cvaeLabels,
        datasets: [{
          label: "Reconstruction Loss (BCE)",
          data: cvae.epochs.map((e) => e.recon_loss ?? e.loss),
          borderColor: css("--accent"),
          backgroundColor: css("--accent"),
          tension: 0.3,
        }],
      },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: { y: { title: { display: true, text: "BCE Loss" } } },
      },
    });

    // KL divergence & beta annealing curve (Dual Axis)
    new Chart($("#chart-cvae-kl"), {
      type: "line",
      data: {
        labels: cvaeLabels,
        datasets: [
          {
            label: "KL Divergence (nats)",
            data: cvae.epochs.map((e) => e.kl_loss ?? 0),
            borderColor: "#8fb8de",
            backgroundColor: "#8fb8de",
            tension: 0.3,
            yAxisID: "yKL",
          },
          {
            label: "Beta Annealing",
            data: cvae.epochs.map((e) => e.beta ?? null),
            borderColor: "#f2b45a",
            backgroundColor: "#f2b45a",
            borderDash: [5, 5],
            tension: 0.1,
            yAxisID: "yBeta",
          },
        ],
      },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: {
          yKL: {
            type: "linear",
            position: "left",
            title: { display: true, text: "KL Divergence (nats)", color: "#8fb8de" },
            ticks: { color: "#8fb8de" },
          },
          yBeta: {
            type: "linear",
            position: "right",
            title: { display: true, text: "Beta Annealing", color: "#f2b45a" },
            ticks: { color: "#f2b45a" },
            min: 0,
            grid: { drawOnChartArea: false },
          },
        },
      },
    });
  }
}

// ---------- Kernels ----------
function renderKernels({ kernels }) {
  if (!kernels) return;
  const hw = kernels.hardware || {};
  $("#hardware").textContent =
    `Criterion medians · ${hw.cpu ?? "?"} · ${hw.os ?? "?"} · rustc ${hw.rustc ?? "?"} · ${hw.threads ?? 1} thread` +
    (kernels.generated_at ? ` · measured ${kernels.generated_at}` : "");

  const accent = css("--accent");

  // Filter 2D matmuls (deduplicate old group names like matmul_varing_sizes if newer varying_sizes exists)
  const hasVarying = kernels.benches.some((b) => b.group === "matmul_varying_sizes");
  const matmul2d = kernels.benches.filter((b) => {
    if (!b.flops) return false;
    if (b.group === "matmul_batched_sizes" || b.group === "matmul_batched_comp") return false;
    if (b.label.startsWith("b16_") || b.label.startsWith("b64_") || b.label === "batched" || b.label === "serialised") return false;
    if (hasVarying && b.group === "matmul_varing_sizes") return false;
    return b.group.toLowerCase().includes("matmul") || b.group.toLowerCase().includes("multiplication");
  });

  // Sort 2D by matrix dimension: 64 -> 128 -> 256 -> 512, keeping naive right before optimised
  matmul2d.sort((a, b) => {
    const sizeA = parseInt(a.label.match(/\d+/)?.[0] || "0", 10);
    const sizeB = parseInt(b.label.match(/\d+/)?.[0] || "0", 10);
    if (sizeA !== sizeB) return sizeA - sizeB;
    if (a.label.includes("naive")) return -1;
    if (b.label.includes("naive")) return 1;
    return 0;
  });

  const matmul2dColors = matmul2d.map((b) => (b.label.includes("naive") ? "#6e6a66" : accent));

  new Chart($("#chart-matmul"), {
    type: "bar",
    data: {
      labels: matmul2d.map((b) => b.label),
      datasets: [{
        label: "GFLOP/s",
        data: matmul2d.map((b) => Number((b.flops / b.median_ns).toFixed(2))),
        backgroundColor: matmul2dColors,
      }],
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const item = matmul2d[ctx.dataIndex];
              return `Median: ${fmtTime(item.median_ns)}`;
            },
          },
        },
      },
      scales: { y: { beginAtZero: true, title: { display: true, text: "GFLOP/s" } } },
    },
  });

  // Batched matmuls: b16, b64 configs
  const matmulBatched = kernels.benches.filter(
    (b) => b.group === "matmul_batched_sizes" && b.flops
  );

  // Sort by batch then size then batched/broadcast
  matmulBatched.sort((a, b) => {
    const parse = (lbl) => {
      const m = lbl.match(/b(\d+)_(\d+)x(\d+)_?(.*)/);
      return m ? { b: parseInt(m[1]), s: parseInt(m[2]), type: m[4] } : { b: 0, s: 0, type: "" };
    };
    const pa = parse(a.label);
    const pb = parse(b.label);
    if (pa.b !== pb.b) return pa.b - pb.b;
    if (pa.s !== pb.s) return pa.s - pb.s;
    return pa.type.localeCompare(pb.type);
  });

  const batchedColors = matmulBatched.map((b) =>
    b.label.includes("broadcast") ? "#8fb8de" : accent
  );

  const batchedCanvas = $("#chart-matmul-batched");
  if (batchedCanvas) {
    new Chart(batchedCanvas, {
      type: "bar",
      data: {
        labels: matmulBatched.map((b) => b.label),
        datasets: [{
          label: "GFLOP/s",
          data: matmulBatched.map((b) => Number((b.flops / b.median_ns).toFixed(2))),
          backgroundColor: batchedColors,
        }],
      },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const item = matmulBatched[ctx.dataIndex];
                return `Median: ${fmtTime(item.median_ns)}`;
              },
            },
          },
        },
        scales: { y: { beginAtZero: true, title: { display: true, text: "GFLOP/s" } } },
      },
    });
  }

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
  const k = reference.kernels;
  const accent = css("--accent");

  let currentMode = "multi"; // Default to full multi-core comparison

  const updateNote = () => {
    const threadCount = currentMode === "single" ? (r.single_threads ?? 1) : (r.multi_threads ?? 16);
    $("#reference-note").textContent =
      `Same shapes, same architectures. PyTorch ${r.version ?? ""} using ${threadCount} thread${threadCount > 1 ? "s" : ""}. ` +
      `Mentats uses smart thresholding (Rayon on large MatMuls, single-thread auto-vectorization on small/single-image passes) with zero external BLAS.`;
    $("#reference-chart-title").textContent =
      `Share of PyTorch ${currentMode === "single" ? "1-thread" : `${r.multi_threads ?? 16}-thread`} speed (%)`;
  };

  updateNote();

  const getPcts = (mode) =>
    k.map((x) => {
      const refTime = mode === "single" ? (x.reference_single_ns ?? x.reference_ns) : (x.reference_multi_ns ?? x.reference_ns);
      return Number(((refTime / x.mentats_ns) * 100).toFixed(1));
    });

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

      // Top label for parity line
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText("100% (PyTorch Parity)", x100, top - 6);

      // Bar percentage labels
      const meta = chart.getDatasetMeta(0);
      meta.data.forEach((bar, index) => {
        const val = chart.data.datasets[0].data[index];
        const text = `${val.toFixed(1)}%`;
        ctx.font = "bold 11px ui-monospace, monospace";
        ctx.textBaseline = "middle";

        const barWidth = bar.x - bar.base;
        if (barWidth > 140) {
          ctx.fillStyle = "#111115";
          ctx.textAlign = "right";
          ctx.fillText(text, bar.x - 10, bar.y);
        } else {
          ctx.fillStyle = "#e7e5e0";
          ctx.textAlign = "left";
          ctx.fillText(text, bar.x + 8, bar.y);
        }
      });
      ctx.restore();
    },
  };

  const chart = new Chart($("#chart-reference"), {
    type: "bar",
    data: {
      labels: k.map((x) => x.label),
      datasets: [{
        label: "% of PyTorch speed",
        data: getPcts("multi"),
        backgroundColor: getPcts("multi").map((pct) => (pct >= 100 ? "#f2b45a" : accent)),
      }],
    },
    options: {
      indexAxis: "y",
      layout: {
        padding: { top: 30, right: 55, left: 10, bottom: 5 },
      },
      scales: {
        x: {
          beginAtZero: true,
          suggestedMax: 150,
          title: { display: true, text: "% of PyTorch Performance" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.parsed.x.toFixed(1)}% of PyTorch speed`,
            afterLabel: (c) => {
              const item = k[c.dataIndex];
              const refTime = currentMode === "single" ? (item.reference_single_ns ?? item.reference_ns) : (item.reference_multi_ns ?? item.reference_ns);
              return ` mentats: ${fmtTime(item.mentats_ns)} vs PyTorch (${currentMode}): ${fmtTime(refTime)}`;
            },
          },
        },
      },
    },
    plugins: [parityPlugin],
  });

  const btnSingle = $("#btn-ref-single");
  const btnMulti = $("#btn-ref-multi");

  if (btnSingle && btnMulti) {
    btnSingle.addEventListener("click", () => {
      currentMode = "single";
      btnSingle.style.background = "var(--card-bg)";
      btnSingle.style.color = "var(--text)";
      btnMulti.style.background = "transparent";
      btnMulti.style.color = "var(--muted)";
      updateNote();
      const pcts = getPcts("single");
      chart.data.datasets[0].data = pcts;
      chart.data.datasets[0].backgroundColor = pcts.map((p) => (p >= 100 ? "#f2b45a" : accent));
      chart.update();
    });

    btnMulti.addEventListener("click", () => {
      currentMode = "multi";
      btnMulti.style.background = "var(--card-bg)";
      btnMulti.style.color = "var(--text)";
      btnSingle.style.background = "transparent";
      btnSingle.style.color = "var(--muted)";
      updateNote();
      const pcts = getPcts("multi");
      chart.data.datasets[0].data = pcts;
      chart.data.datasets[0].backgroundColor = pcts.map((p) => (p >= 100 ? "#f2b45a" : accent));
      chart.update();
    });
  }
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
