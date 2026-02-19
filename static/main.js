/* ──────────────────────────────────────────
   ベイズ最適化 可視化 – メインスクリプト
   ────────────────────────────────────────── */

let currentData = null;
let displayMode  = "true_f";
let initialized  = false;

// ────────────────────────────────────────────
// API ヘルパー
// ────────────────────────────────────────────
async function apiFetch(url, method = "GET") {
  setLoading(true);
  try {
    const res = await fetch(url, { method });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  document.getElementById("overlay").classList.toggle("hidden", !on);
  document.getElementById("btn-step").disabled  = on;
  document.getElementById("btn-reset").disabled = on;
}

// ────────────────────────────────────────────
// ボタン操作
// ────────────────────────────────────────────
async function resetFunction() {
  const data = await apiFetch("/api/reset");
  currentData = data;
  renderAll(data, true);   // true = 新規プロット
}

async function stepGP() {
  if (!currentData) return;
  const data = await apiFetch("/api/step", "POST");
  currentData = data;
  renderAll(data, false);  // false = 再描画
}

function setMode(mode) {
  displayMode = mode;
  document.querySelectorAll(".toggle-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === mode)
  );
  if (currentData) renderChart(currentData, false);
}

// ────────────────────────────────────────────
// UI 更新
// ────────────────────────────────────────────
function updateInfoPanel(d) {
  const fmt = (v, dec = 4) => Number(v).toFixed(dec);
  const pt  = (arr) => `(${fmt(arr[0], 3)}, ${fmt(arr[1], 3)})`;

  document.getElementById("i-step").textContent    = d.step;
  document.getElementById("i-obs").textContent     = d.obs_x.length;
  document.getElementById("i-best-f").textContent  = fmt(d.best_f);
  document.getElementById("i-best-pt").textContent = pt(d.best_point);
  document.getElementById("i-next").textContent    = pt(d.next_point);
  document.getElementById("formula-text").textContent = d.formula;
}

// ────────────────────────────────────────────
// Plotly レンダリング
// ────────────────────────────────────────────
const COLORSCALES = {
  true_f:   "Viridis",
  gp_mean:  "RdBu",
  gp_sigma: "Hot",
};

const LAYER_TITLES = {
  true_f:   "真の関数 f(x,y)",
  gp_mean:  "GP 平均 μ(x,y)",
  gp_sigma: "GP 不確かさ σ(x,y)",
};

function getZData(d) {
  switch (displayMode) {
    case "gp_mean":  return d.mu_grid;
    case "gp_sigma": return d.sigma_grid;
    default:         return d.f_grid;
  }
}

function buildTraces(d) {
  // ── ヒートマップ ──
  const heatmap = {
    type:        "heatmap",
    x:           d.xs,
    y:           d.ys,
    z:           getZData(d),
    colorscale:  COLORSCALES[displayMode],
    showscale:   true,
    colorbar:    { title: { text: LAYER_TITLES[displayMode], side: "right" },
                   len: 0.85, y: 0.5, thickness: 14 },
    hovertemplate: "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.4f}<extra></extra>",
    zsmooth: "best",
  };

  // ── 観測点 (カラーエンコード: f値) ──
  const obs = {
    type:   "scatter",
    x:      d.obs_x,
    y:      d.obs_y,
    mode:   "markers",
    name:   "観測点",
    marker: {
      color:     d.obs_f,
      colorscale: "Plasma",
      cmin:      Math.min(...d.obs_f),
      cmax:      Math.max(...d.obs_f),
      size:      10,
      symbol:    "circle",
      line:      { color: "white", width: 1.5 },
      showscale: false,
    },
    hovertemplate:
      "観測点<br>x: %{x:.3f}<br>y: %{y:.3f}<br>f: %{marker.color:.4f}<extra></extra>",
  };

  // ── 現在の最大点 (金星) ──
  const best = {
    type:   "scatter",
    x:      [d.best_point[0]],
    y:      [d.best_point[1]],
    mode:   "markers",
    name:   "★ 現在の最大点",
    marker: { color: "#FFD700", size: 20, symbol: "star",
              line: { color: "#333", width: 1.5 } },
    hovertemplate:
      "★ 現在の最大点<br>x: %{x:.3f}<br>y: %{y:.3f}<br>f*: " +
      d.best_f.toFixed(4) + "<extra></extra>",
  };

  // ── 次の提案点 (赤星) ──
  const next = {
    type:   "scatter",
    x:      [d.next_point[0]],
    y:      [d.next_point[1]],
    mode:   "markers",
    name:   "🎯 次の提案点",
    marker: { color: "#FF4455", size: 20, symbol: "star",
              line: { color: "white", width: 1.5 } },
    hovertemplate:
      "🎯 次の提案点<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
  };

  return [heatmap, obs, best, next];
}

function buildLayout() {
  return {
    paper_bgcolor: "#12122a",
    plot_bgcolor:  "#1a1a2e",
    font:          { color: "#ccd6f6", family: "'Segoe UI', sans-serif" },
    xaxis: {
      title:      { text: "x", font: { size: 13 } },
      gridcolor:  "#2a3060",
      zerolinecolor: "#4a5580",
      range: [-3, 3],
    },
    yaxis: {
      title:      { text: "y", font: { size: 13 } },
      gridcolor:  "#2a3060",
      zerolinecolor: "#4a5580",
      scaleanchor: "x",
      range: [-3, 3],
    },
    legend: {
      x: 1.02, y: 1,
      bgcolor:     "rgba(18,18,42,0.85)",
      bordercolor: "#3a4570",
      borderwidth: 1,
      font:        { size: 11 },
    },
    margin: { t: 30, b: 55, l: 60, r: 160 },
    hovermode: "closest",
  };
}

const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  displaylogo: false,
};

function renderChart(d, isNew) {
  const traces = buildTraces(d);
  const layout = buildLayout();

  if (isNew || !initialized) {
    Plotly.newPlot("chart", traces, layout, PLOTLY_CONFIG);
    initialized = true;
  } else {
    Plotly.react("chart", traces, layout, PLOTLY_CONFIG);
  }
}

function renderAll(d, isNew) {
  updateInfoPanel(d);
  renderChart(d, isNew);
}

// ────────────────────────────────────────────
// 初期化
// ────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  resetFunction();
});
