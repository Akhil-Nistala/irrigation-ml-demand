const API_BASE = "";

const STEP_OVERRIDES = {
  rainfall_mm: 0.5,
  temperature_C: 0.1,
  humidity_percent: 1,
  wind_speed_mps: 0.1,
  solar_radiation_MJ_m2_day: 0.1,
  previous_irrigation_mm: 0.1,
  crop_coefficient: 0.01,
};

const GROWTH_STAGE_LABELS = {
  1: "1 — Initial",
  2: "2 — Development",
  3: "3 — Mid-season",
  4: "4 — Late season",
};

const MODEL_COLORS = {
  "Linear Regression": "var(--series-1)",
  "Decision Tree": "var(--series-2)",
  "Random Forest": "var(--series-3)",
};

let META = null;

// ---------------- Tabs ----------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    document.getElementById(tab.dataset.target).classList.add("active");
  });
});

// ---------------- Load meta & build form ----------------
async function loadMeta() {
  const res = await fetch(`${API_BASE}/api/meta`);
  META = await res.json();
  buildForm(META.features);
  buildDatasetSummary(META);
  buildMetricsCharts(META.model_comparison);
  buildImportanceChart(META.feature_importance);
}

function buildForm(features) {
  const container = document.getElementById("field-list");
  container.innerHTML = "";

  features.forEach((f) => {
    const wrap = document.createElement("div");
    wrap.className = "field";

    if (f.name === "growth_stage") {
      const defaultVal = Math.min(4, Math.max(1, Math.round(f.mean)));
      wrap.innerHTML = `
        <div class="field-top">
          <label for="in-${f.name}">${f.label}</label>
        </div>
        <select id="in-${f.name}" data-feature="${f.name}" class="growth-select">
          ${[1, 2, 3, 4].map((v) => `<option value="${v}" ${v === defaultVal ? "selected" : ""}>${GROWTH_STAGE_LABELS[v]}</option>`).join("")}
        </select>
      `;
      container.appendChild(wrap);
      return;
    }

    const step = STEP_OVERRIDES[f.name] ?? 0.1;
    const defaultVal = Math.round(f.mean / step) * step;

    wrap.innerHTML = `
      <div class="field-top">
        <label for="in-${f.name}">${f.label}</label>
        <span class="unit">${f.unit}</span>
      </div>
      <div class="field-value-row">
        <input type="range" id="in-${f.name}" data-feature="${f.name}"
               min="${f.min}" max="${f.max}" step="${step}" value="${defaultVal}">
        <input type="number" data-feature="${f.name}" data-mirror="in-${f.name}"
               min="${f.min}" max="${f.max}" step="${step}" value="${defaultVal}">
      </div>
    `;
    container.appendChild(wrap);
  });

  // Keep slider <-> number input in sync
  container.querySelectorAll('input[type="range"]').forEach((range) => {
    const number = container.querySelector(`input[type="number"][data-mirror="${range.id}"]`);
    range.addEventListener("input", () => (number.value = range.value));
    number.addEventListener("input", () => (range.value = number.value));
  });
}

function buildDatasetSummary(meta) {
  document.getElementById("dataset-summary").textContent =
    `${meta.dataset_size.toLocaleString()} daily records · active model: ${meta.active_model}. ` +
    `Metrics below are computed on the held-out 20% test split.`;
}

// ---------------- Training: model comparison small multiples ----------------
function buildMetricsCharts(comparison) {
  const modelNames = Object.keys(comparison);
  const metricKeys = ["MAE", "RMSE", "R2"];
  const metricFmt = { MAE: (v) => v.toFixed(3), RMSE: (v) => v.toFixed(3), R2: (v) => v.toFixed(3) };

  const legend = document.getElementById("metric-legend");
  legend.innerHTML = modelNames
    .map((name) => `<span class="legend-item"><span class="legend-swatch" style="background:${MODEL_COLORS[name] || "var(--series-1)"}"></span>${name}</span>`)
    .join("");

  const chartsEl = document.getElementById("metrics-charts");
  chartsEl.innerHTML = "";
  metricKeys.forEach((metric) => {
    const values = modelNames.map((m) => comparison[m][metric]);
    const maxVal = Math.max(...values) || 1;

    const panel = document.createElement("div");
    panel.className = "metric-panel";
    panel.innerHTML = `<h4>${metric}${metric === "R2" ? " (higher is better)" : " mm (lower is better)"}</h4>
      <div class="metric-bars">
        ${modelNames
          .map((m) => {
            const v = comparison[m][metric];
            const heightPct = Math.max(4, (v / maxVal) * 100);
            return `<div class="metric-bar-col">
              <span class="metric-bar-value">${metricFmt[metric](v)}</span>
              <div class="metric-bar" style="height:${heightPct}%; background:${MODEL_COLORS[m] || "var(--series-1)"}"></div>
            </div>`;
          })
          .join("")}
      </div>`;
    chartsEl.appendChild(panel);
  });

  const table = document.getElementById("metrics-table");
  table.innerHTML = `
    <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R&sup2;</th></tr></thead>
    <tbody>
      ${modelNames
        .map(
          (m) =>
            `<tr><td>${m}</td><td>${comparison[m].MAE.toFixed(3)}</td><td>${comparison[m].RMSE.toFixed(3)}</td><td>${comparison[m].R2.toFixed(3)}</td></tr>`
        )
        .join("")}
    </tbody>`;
}

// ---------------- Training: feature importance bar chart ----------------
function buildImportanceChart(importance) {
  const rows = importance.features.map((name, i) => ({
    name,
    value: importance.rf_permutation_importance_mean[i],
  }));
  rows.sort((a, b) => b.value - a.value);
  const maxVal = Math.max(...rows.map((r) => r.value)) || 1;

  const el = document.getElementById("importance-chart");
  el.innerHTML = rows
    .map((r) => {
      const pct = Math.max(2, (r.value / maxVal) * 100);
      return `<div class="bar-row">
        <span class="bar-label">${labelFor(r.name)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <span class="bar-value">${r.value.toFixed(3)}</span>
      </div>`;
    })
    .join("");
}

function labelFor(featureName) {
  const match = META.features.find((f) => f.name === featureName);
  return match ? match.label : featureName;
}

// ---------------- Predict ----------------
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorEl = document.getElementById("predict-error");
  errorEl.hidden = true;

  const payload = {};
  document.querySelectorAll("#field-list [data-feature]").forEach((el) => {
    if (el.dataset.mirror) return; // skip mirrored number input, range already captured
    payload[el.dataset.feature] = el.value;
  });
  // number inputs (non-range) also carry data-feature for growth_stage select; ranges already included

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    renderResult(data);
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  }
});

function renderResult(data) {
  document.getElementById("result-empty").hidden = true;
  document.getElementById("result-content").hidden = false;
  document.getElementById("hero-value").textContent = data.prediction_mm.toFixed(2);
  document.getElementById("hero-model").textContent = `Model: ${data.model} · baseline ${data.bias_mm.toFixed(2)} mm`;

  const maxAbs = Math.max(...data.breakdown.map((b) => Math.abs(b.contribution)), 0.01);
  const chart = document.getElementById("contrib-chart");
  chart.innerHTML = data.breakdown
    .map((b) => {
      const pct = Math.min(100, (Math.abs(b.contribution) / maxAbs) * 50);
      const sign = b.contribution >= 0 ? "positive" : "negative";
      return `<div class="contrib-row">
        <span class="contrib-label">${b.label}</span>
        <div class="contrib-track">
          <div class="contrib-bar ${sign}" style="width:${pct}%"></div>
        </div>
        <span class="contrib-value">${b.contribution >= 0 ? "+" : ""}${b.contribution.toFixed(2)}</span>
      </div>`;
    })
    .join("");
}

loadMeta();
