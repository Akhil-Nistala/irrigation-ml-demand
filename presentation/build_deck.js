// Builds presentation/WRE_Project_Part3.pptx (Part 3 of the term-project presentation) from the results in this repository.
// Every chart reads its numbers from models/*.json, report/*.csv, report/showcase_summary.json and data/*.csv,
// so the deck follows the experiments if they are re-run (src/no_lag_experiment.py, src/showcase_analysis.py,
// src/part3_experiments.py).
//
//   npm install pptxgenjs          (once, anywhere on NODE_PATH)
//   node presentation/build_deck.js
//
// Optional: PPTX_THEME_HELPER=<path to apply_theme.js> writes the palette into the deck's theme.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(__dirname, "WRE_Project_Part3.pptx");

// ---------------------------------------------------------------- data
const readJSON = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
function readCSV(p) {
  const lines = fs.readFileSync(path.join(ROOT, p), "utf8").trim().split(/\r?\n/);
  const split = (l) => l.match(/("([^"]*)"|[^,]*)(,|$)/g).slice(0, -1).map((c) => c.replace(/,$/, "").replace(/^"|"$/g, ""));
  const head = split(lines[0]);
  return lines.slice(1).map((l) => {
    const cells = split(l);
    const row = {};
    head.forEach((h, i) => (row[h || "key"] = cells[i]));
    return row;
  });
}
const V1 = readJSON("models/metrics.json"); // first version: 8 inputs incl. yesterday's demand, synthetic weather
const SYN = readJSON("models/metrics_no_lag.json").results;
const NASA = readJSON("models/metrics_nasa.json").results;
const META = readJSON("report/nasa_power_dataset_meta.json");
const SHOW = readJSON("report/showcase_summary.json");
const P3 = readJSON("report/part3_summary.json");
const ROB = readJSON("report/robustness_checks.json"), VF = readJSON("report/viva_followups.json"), VR2 = readJSON("report/viva_round2_checks.json");
const impSyn = Object.fromEntries(readCSV("report/no_lag_feature_importance.csv").map((r) => [r.key, r]));
const impNasa = Object.fromEntries(readCSV("report/nasa_feature_importance.csv").map((r) => [r.key, r]));
const impV1 = Object.fromEntries(readCSV("report/feature_importance_summary.csv").map((r) => [r.key, r]));
const wx = Object.fromEntries(readCSV("report/nasa_vs_synthetic_weather.csv").map((r) => [r.key, r]));

function overfit(p) {
  const out = {};
  for (const r of readCSV(p)) {
    const o = (out[r.model] = out[r.model] || { train: [], val: [], test: [] });
    o.train.push(+r.train_MAE); o.val.push(+r.validation_MAE); o.test.push(+r.test_MAE);
  }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  for (const k in out) for (const m of ["train", "val", "test"]) out[k][m] = mean(out[k][m]);
  return out;
}
const OF_SYN = overfit("report/no_lag_overfitting_check.csv");
const OF_NASA = overfit("report/nasa_overfitting_check.csv");

function monthly(p, col) {
  const s = Array(12).fill(0), n = Array(12).fill(0);
  for (const r of readCSV(p)) { const m = +r.date.slice(5, 7) - 1; s[m] += +r[col]; n[m] += 1; }
  return s.map((v, i) => v / n[i]);
}
const DATA_SYN = "data/irrigation_dataset.csv", DATA_NASA = "data/irrigation_dataset_nasa_power.csv";
const demandSyn = monthly(DATA_SYN, "irrigation_demand_mm"), demandNasa = monthly(DATA_NASA, "irrigation_demand_mm");
const windSyn = monthly(DATA_SYN, "wind_speed_mps"), windNasa = monthly(DATA_NASA, "wind_speed_mps");
const nasaRows = readCSV(DATA_NASA).map((r) => +r.irrigation_demand_mm);
const nasaMean = nasaRows.reduce((a, v) => a + v, 0) / nasaRows.length;
const meanOf = (p, col) => { const r = readCSV(p); return r.reduce((a, x) => a + +x[col], 0) / r.length; };
const HYD_WIND = meanOf("data/irrigation_dataset_hyderabad.csv", "wind_speed_mps"), NAG_WIND = meanOf("data/irrigation_dataset_nasa_power.csv", "wind_speed_mps");
const nasaTrainMax = Math.max(...nasaRows.slice(0, 1600)), nasaTestMax = Math.max(...nasaRows.slice(1600));

const f1 = (v) => (+v).toFixed(1), f2 = (v) => (+v).toFixed(2), f3 = (v) => (+v).toFixed(3);
const pct = (a, b) => Math.round((1 - b / a) * 100);
const fmtInt = (v) => Math.round(v).toLocaleString("en-US");
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const NOISE_FLOOR = 0.18 * Math.sqrt(2 / Math.PI); // MAE of the N(0, 0.18 mm) noise added to the target

const MODELS = ["Linear Regression", "Decision Tree", "Random Forest", "RF + 7-day window", "MLP", "1D-CNN (7-day)", "LSTM (7-day)"];
const SHORT = { "Linear Regression": "Linear Regr.", "Decision Tree": "Decision Tree", "Random Forest": "Random Forest",
  "RF + 7-day window": "RF + 7-day", "MLP": "MLP", "1D-CNN (7-day)": "1D-CNN", "LSTM (7-day)": "LSTM" };
const FEATS = ["crop_coefficient", "solar_radiation_MJ_m2_day", "temperature_C", "rainfall_mm", "humidity_percent", "wind_speed_mps", "growth_stage"];
const FLAB = { crop_coefficient: "Crop coefficient", solar_radiation_MJ_m2_day: "Solar radiation", temperature_C: "Temperature",
  rainfall_mm: "Rainfall", humidity_percent: "Humidity", wind_speed_mps: "Wind speed", growth_stage: "Growth stage" };

// ---------------------------------------------------------------- design system
const HEX = { teal: "0F4C5C", panel: "EAF3F2", amber: "E36414", ink: "1E2B2F", muted: "5B6B70",
  mid: "7FA9B0", light: "B9D3D2", grid: "E1E8E9", white: "FFFFFF" };
const THEME = {
  name: "Irrigation Teal",
  headFontFace: "Avenir Next Demi Bold",
  bodyFontFace: "Avenir Next",
  colors: { dk1: HEX.ink, lt1: HEX.white, dk2: HEX.teal, lt2: HEX.panel, accent1: HEX.teal, accent2: HEX.amber,
    accent3: HEX.mid, accent4: HEX.light, accent5: HEX.muted, accent6: "9DBFC4", hlink: HEX.teal, folHlink: HEX.muted },
};
const BOLD = "+mj-lt"; // theme heading face (Avenir Next Demi Bold), used for emphasis inside body text

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Predicting Daily Irrigation Water Demand with Machine Learning";
pres.subject = "Term project, Part 3, Civil Engineering, IIT Hyderabad";
pres.author = "Akhil Nistala, Sai Nishant, Vaishnav Dhruv, Krishna Vardhan";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const S = pres.shapes;
const GREY = "93A1A5"; // a quiet grey for baselines
const W = 13.333, MX = 0.6, CW = W - 2 * MX;

// rich text: **demi bold**, _{subscript}, ^{superscript}; "\n" starts a new paragraph.
// Paragraph-level options (bullet) go on the first run only: pptxgenjs starts a new paragraph at every bulleted run.
function runs(str, baseIn = {}) {
  const { bullet, ...base } = baseIn;
  const out = [];
  const paras = str.split("\n");
  paras.forEach((p, pi) => {
    const pr = [];
    p.split(/(\*\*.+?\*\*)/).filter(Boolean).forEach((seg) => {
      const bold = /^\*\*.+\*\*$/.test(seg);
      const o = bold ? { ...base, fontFace: BOLD } : { ...base };
      const body = bold ? seg.slice(2, -2) : seg;
      const re = /_\{(.+?)\}|\^\{(.+?)\}|([^_^]+|[_^])/g;
      let m;
      while ((m = re.exec(body))) {
        if (m[1]) pr.push({ text: m[1], options: { ...o, subscript: true } });
        else if (m[2]) pr.push({ text: m[2], options: { ...o, superscript: true } });
        else pr.push({ text: m[3], options: { ...o } });
      }
    });
    if (!pr.length) pr.push({ text: " ", options: { ...base } });
    if (bullet) pr[0].options.bullet = bullet;
    if (pi < paras.length - 1) pr[pr.length - 1].options.breakLine = true;
    out.push(...pr);
  });
  return out;
}
const plain = (s) => s.replace(/\*\*|_\{|\^\{|\}/g, "");

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.teal },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.45, w: 11.4, h: 1.9, fontFace: THEME.headFontFace,
      fontSize: 40, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 3.55, w: 11.2, h: 1.0, fontSize: 20,
      color: C.accent4, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "byline", type: "body", x: 0.8, y: 5.2, w: 11.2, h: 1.2, fontSize: 16,
      color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: HEX.teal },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 2.75, y: 2.35, w: 9.6, h: 1.0, fontFace: THEME.headFontFace,
      fontSize: 38, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 2.75, y: 3.5, w: 9.0, h: 1.4, fontSize: 18,
      color: C.accent4, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.38, w: 9.4, h: 0.75, fontFace: THEME.headFontFace,
      fontSize: 30, color: C.accent1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Irrigation water demand prediction  ·  Civil Engineering, IIT Hyderabad",
      options: { x: MX, y: 7.02, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, valign: "top" } } },
  ],
  slideNumber: { x: 12.03, y: 7.02, w: 0.7, h: 0.3, fontSize: 10, color: C.accent5, align: "right", margin: 0 },
});

// ---------------------------------------------------------------- slide helpers
let curSection = "";
function section(title) { curSection = title; pres.addSection({ title }); }
const TAGS = { 1: "New site", 2: "New crop", 3: "Days ahead", 4: "Imperfect records", 5: "Practical use", A: "Backup" };
function content(title, tag) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: curSection });
  s.addText(title, { placeholder: "title" });
  if (tag !== undefined) {
    const w = 2.35, x = W - MX - w, y = 0.535, h = 0.44;
    s.addShape(S.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2 },
      rectRadius: 0.22, objectName: "Component tag" });
    badge(s, tag, x + 0.07, y + 0.06, 0.32, { size: 12 });
    s.addText(TAGS[tag], { x: x + 0.47, y, w: w - 0.55, h, fontSize: 12, fontFace: BOLD, color: C.accent1,
      valign: "middle", margin: 0, isTextBox: true, objectName: "Component tag text" });
  }
  return s;
}
function takeaway(s, str, y = 6.2) {
  s.addShape(S.ROUNDED_RECTANGLE, { x: MX, y, w: CW, h: 0.65, fill: { color: C.background2 }, line: { color: C.background2 },
    rectRadius: 0.08, objectName: "Takeaway panel" });
  s.addText(runs(str, { fontSize: 15, color: C.text1 }), { x: MX + 0.25, y, w: CW - 0.5, h: 0.65, valign: "middle",
    margin: 0, isTextBox: true, objectName: "Takeaway" });
}
function badge(s, n, x, y, d, opts = {}) {
  s.addShape(S.OVAL, { x, y, w: d, h: d, fill: { color: opts.fill || C.accent1 }, line: { color: opts.fill || C.accent1 },
    objectName: `Badge ${n}` });
  s.addText(String(n), { x, y, w: d, h: d, fontSize: opts.size || Math.round(d * 30), fontFace: BOLD,
    color: opts.color || C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `Badge ${n} number` });
}
function panel(s, x, y, w, h, name, fill) {
  s.addShape(S.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill || C.background2 }, line: { color: fill || C.background2 },
    rectRadius: 0.1, objectName: name });
}
function text(s, str, o) {
  const { size = 15, color = C.text1, paraSpaceAfter, align, x, y, w, h, valign = "top", bullet } = o;
  const base = { fontSize: size, color };
  if (paraSpaceAfter) base.paraSpaceAfter = paraSpaceAfter;
  if (align) base.align = align;
  if (bullet) base.bullet = bullet;
  s.addText(runs(str, base), { x, y, w, h, valign, margin: 0, isTextBox: true });
}
function statCard(s, x, y, w, h, big, label, opts = {}) {
  const bigSize = opts.bigSize || 32, bigH = (bigSize / 72) * 1.3;
  panel(s, x, y, w, h, `Stat card ${big}`, opts.fill);
  s.addText(big, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: bigH, fontSize: bigSize, fontFace: BOLD,
    color: opts.bigColor || C.accent1, valign: "bottom", margin: 0, isTextBox: true, objectName: `Stat ${big}` });
  s.addText(runs(label, { fontSize: opts.labelSize || 14, color: C.text1 }), { x: x + 0.25, y: y + 0.18 + bigH + 0.08,
    w: w - 0.5, h: h - bigH - 0.38, valign: "top", margin: 0, isTextBox: true, objectName: `Stat ${big} label` });
}
// one style for every chart: same fonts, label colours and grid
function chartOpts(extra) {
  return {
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLineShow: true, catAxisLineColor: HEX.light, valAxisLineShow: false,
    valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
    dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.ink,
    showTitle: false, legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.ink,
    valAxisTitleFontFace: "+mn-lt", valAxisTitleFontSize: 12, valAxisTitleColor: HEX.muted,
    catAxisTitleFontFace: "+mn-lt", catAxisTitleFontSize: 12, catAxisTitleColor: HEX.muted,
    ...extra,
  };
}
function chartTitle(s, str, x, y, w) {
  s.addText(runs(str, { fontSize: 15, fontFace: BOLD, color: C.text1 }), { x, y, w, h: 0.4, align: "center", valign: "middle",
    margin: 0, isTextBox: true, objectName: `Chart title: ${plain(str)}` });
}
function legend(s, items, x, y) {
  let cx = x;
  for (const [label, color] of items) {
    s.addShape(S.RECTANGLE, { x: cx, y: y + 0.09, w: 0.22, h: 0.22, fill: { color }, line: { color }, objectName: `Legend ${label}` });
    const w = 0.25 + label.length * 0.095;
    s.addText(label, { x: cx + 0.3, y, w, h: 0.4, fontSize: 12, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    cx += 0.3 + w + 0.3;
  }
}
function bullets(s, items, x, y, w, h, size = 14, space = 8) {
  s.addText(items.flatMap((t, i) => runs(t, { fontSize: size, color: C.text1, bullet: true, paraSpaceAfter: space })
    .map((r, j, a) => (j === a.length - 1 && i < items.length - 1 ? { ...r, options: { ...r.options, breakLine: true } } : r))),
    { x, y, w, h, valign: "top", margin: 0, isTextBox: true });
}
function hexMix(t) { // white -> teal, for heatmap cells
  const a = [255, 255, 255], b = [0x0f, 0x4c, 0x5c];
  return a.map((c, i) => Math.round(c + (b[i] - c) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
}
const RF = NASA["Random Forest"], LS = NASA["LSTM (7-day)"], RFs = SYN["Random Forest"], LSs = SYN["LSTM (7-day)"];

// ================================================================= 1. Title
section("Where we left off");
const lagDrop = f2(impV1.previous_irrigation_mm.rf_permutation_importance_mean); // drop in test R² when yesterday's demand is shuffled
const SITE = P3.new_site, CROP = P3.new_crop, AHEAD = P3.days_ahead, NOISY = P3.noisy_records, CTX = P3.context;
const hydT = SITE["Hyderabad, Nagpur model"], hydR = SITE["Hyderabad, retrained"], nagT = SITE["Nagpur (trained here)"];
const cotT = CROP["Cotton, maize model"], cotR = CROP["Cotton, retrained"];
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: curSection });
  s.addText("Predicting Irrigation Water Demand with Machine Learning", { placeholder: "title" });
  s.addText("Part 3: open questions, practical use and final results", { placeholder: "subtitle" });
  s.addText([
    { text: "Akhil Nistala  ·  Sai Nishant  ·  Vaishnav Dhruv  ·  Krishna Vardhan", options: { fontFace: BOLD, breakLine: true } },
    { text: "Civil Engineering (Water Resources Engineering) Term Project  ·  IIT Hyderabad", options: { breakLine: true } },
    { text: "November 2026", options: { color: C.accent4 } },
  ], { placeholder: "byline" });
  s.addNotes("Part 1 (27 September) built the first model; Part 2 (5 October) removed yesterday's demand as an input, added deep learning and real weather. " +
    "Part 3 closes the open questions Part 2 ended with, shows what the model offers in practice, and summarises the whole project.");
}

// ================================================================= 2. Parts 1 and 2
{
  const s = content("Parts 1 and 2 in brief");
  const lw = 6.2;
  const parts = [
    ["Part 1  ·  27 September", [
      "**Data** · made-up weather for maize at Nagpur, 2,000 days",
      `**Result** · Random Forest R^{2} ${f3(V1["Random Forest"].R2)} with 8 inputs`,
      `**Problem** · it mostly copied yesterday's demand: scrambling that input cut R^{2} by ${lagDrop}`]],
    ["Part 2  ·  5 October", [
      `**Yesterday's demand removed** · 7 inputs, R^{2} still ${f3(RFs.R2)}`,
      `**Neural networks** · LSTM error ${f2(LSs.MAE)} mm/day, ${pct(RFs.MAE, LSs.MAE)}% lower than Random Forest`,
      `**Real weather** · NASA data, LSTM error ${f3(LS.MAE)} mm/day`]],
  ];
  parts.forEach(([t, items], i) => {
    const y = 1.4 + i * 2.35, h = 2.15;
    panel(s, MX, y, lw, h, `${t} panel`);
    s.addText(t, { x: MX + 0.3, y: y + 0.15, w: lw - 0.6, h: 0.4, fontSize: 16, fontFace: BOLD, color: C.accent1, margin: 0, isTextBox: true });
    text(s, items.join("\n"), { x: MX + 0.3, y: y + 0.62, w: lw - 0.6, h: h - 0.75, size: 14, paraSpaceAfter: 5 });
  });
  const cx = MX + lw + 0.35, cw = W - MX - cx;
  const stages = [["Forest, 8 inputs", V1["Random Forest"].MAE], ["Forest, 7 inputs", RFs.MAE], ["LSTM", LSs.MAE], ["LSTM, real weather", LS.MAE]];
  chartTitle(s, "Average error at each stage (mm/day, lower is better)", cx, 1.35, cw);
  s.addChart(pres.charts.BAR, [{ name: "MAE", labels: stages.map((r) => r[0]), values: stages.map((r) => r[1]) }], chartOpts({
    x: cx, y: 1.75, w: cw, h: 4.25, barDir: "col", chartColors: [HEX.light, HEX.mid, HEX.mid, HEX.teal], barGapWidthPct: 55,
    showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12,
    valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 0.5,
  }));
  takeaway(s, "**Part 2 ended with six open problems**; Part 3 tests the four we can check with our data, then looks at practical use.");
  s.addNotes("A one-slide recap. Part 1 used yesterday's demand as an input and reached R² 0.970, but the model leaned on that input through day-to-day persistence (shuffling it cut test R² by 0.58). " +
    "Part 2 removed it (almost no loss), showed that an LSTM beats Random Forest by 35%, and confirmed the result on real NASA POWER weather. " +
    "The first bar is Part 1, the other three are Part 2. The first three use the synthetic target, whose ET0 is capped at 8 mm/day; the last uses the real-weather target, which has no cap, so the bars show the order of work rather than a like-for-like improvement.");
}

// ================================================================= 3. Part 3 overview
section("Open questions");
{
  const s = content("Part 3 overview");
  text(s, "**Where Part 2 stopped.** One site, one crop, predictions for today only, and almost error-free demand values.\n" +
    "**Part 3.** We test each of these with real weather, then ask what the model can do for a farm.",
    { x: MX, y: 1.3, w: CW, h: 0.8, size: 16, paraSpaceAfter: 2 });
  const comps = [
    ["New site", "IIT Hyderabad, with and without retraining", "Part 2: Nagpur only"],
    ["New crop", "cotton, the main crop around Nagpur", "Part 2: maize only"],
    ["Days ahead", "1, 3 and 7 days ahead, no weather forecast", "Part 2: today only"],
    ["Imperfect records", "random errors added to the demand values", "Part 2: almost error-free values"],
    ["Practical use", "water totals, cheaper sensors, data needs", "new in Part 3"],
  ];
  const y0 = 2.3, rh = 0.78, gap = 0.12, lw = 7.75;
  comps.forEach(([t, d, from], i) => {
    const y = y0 + i * (rh + gap);
    panel(s, MX, y, lw, rh, `Component ${i + 1} panel`);
    badge(s, i + 1, MX + 0.2, y + (rh - 0.5) / 2, 0.5, { size: 15 });
    s.addText(t, { x: MX + 0.9, y, w: 2.2, h: rh, fontSize: 16, fontFace: BOLD, color: C.accent1, margin: 0, valign: "middle", isTextBox: true });
    text(s, `${d}\n${from}`, { x: MX + 3.15, y: y + 0.08, w: lw - 3.35, h: rh - 0.16, size: 14, valign: "middle" });
  });
  const x = 8.75, w = W - MX - x, y = y0, h = 5 * rh + 4 * gap;
  panel(s, x, y, w, h, "Headline card");
  s.addText("New site, no retraining", { x: x + 0.3, y: y + 0.3, w: w - 0.6, h: 0.35, fontSize: 14, color: C.accent5, margin: 0, isTextBox: true });
  s.addText([
    { text: f2(hydT.LSTM), options: { fontSize: 66, fontFace: BOLD, color: C.accent2 } },
    { text: " mm/day", options: { fontSize: 20, fontFace: BOLD, color: C.accent2 } },
  ], { x: x + 0.3, y: y + 0.7, w: w - 0.6, h: 1.15, valign: "middle", margin: 0, isTextBox: true, objectName: "Headline number" });
  text(s, `error of the Nagpur LSTM at IIT Hyderabad\nretrained there: ${f2(hydR.LSTM)}\nRandom Forest: ${f2(hydT.RF)} (retrained ${f2(hydR.RF)})`,
    { x: x + 0.3, y: y + 2.05, w: w - 0.6, h: 2.0, size: 15, paraSpaceAfter: 8 });
  s.addNotes("Part 2 listed six limitations. Four can be tested with data we already have or can download: a second site, a second crop, predictions for future days, " +
    "and noisier targets. The other two (gridded instead of station weather, calculated instead of measured demand) need field data; they stay on the limitations slide. " +
    "The five numbered components return on the conclusions slide in the same order. Headline: the LSTM trained at Nagpur predicts demand at IIT Hyderabad almost as well as one trained there.");
}

// ================================================================= Baselines and unseen data
{
  const s = content("Tougher comparisons, new data");
  const GB = ROB.gradient_boosting["NASA POWER, 7 inputs"].GB_500.MAE, U = VF.unseen_period, BS = VF.bootstrap_rf_minus_lstm;
  const base = [["Copy yesterday", VF.persistence_mae.nasa, GREY], ["Linear Regression", NASA["Linear Regression"].MAE, HEX.light],
    ["Random Forest", RF.MAE, HEX.mid], ["Gradient Boosting", GB, HEX.mid], ["MLP", NASA.MLP.MAE, HEX.teal], ["LSTM", LS.MAE, HEX.amber]];
  const order = [...base].reverse(); // horizontal bars draw the first category at the bottom
  chartTitle(s, "Average error on the test year, May 2023 – Jun 2024 (mm/day)", MX, 1.35, 8.1);
  s.addChart(pres.charts.BAR, [{ name: "MAE", labels: order.map((r) => r[0]), values: order.map((r) => r[1]) }], chartOpts({
    x: MX, y: 1.75, w: 8.1, h: 4.25, barDir: "bar", chartColors: order.map((r) => r[2]), barGapWidthPct: 45, showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisHidden: true, valGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 1.1, catAxisLabelFontSize: 13,
  }));
  const x = 9.1, w = W - MX - x;
  statCard(s, x, 1.75, w, 2.0, f3(U.LSTM), `LSTM error on ${U.days} later days we never used (Jun 2024 – Nov 2026); Random Forest ${f2(U.RF)}, copying yesterday ${f2(U.persistence)}`,
    { bigSize: 30, bigColor: C.accent2 });
  statCard(s, x, 4.0, w, 2.0, `${f2(BS.ci95[0])}–${f2(BS.ci95[1])}`, "how much larger Random Forest's error is than the LSTM's, mm/day (95% confidence range)", { bigSize: 30 });
  takeaway(s, `**The LSTM stays best against tougher rivals and new data**: ${pct(GB, LS.MAE)}% below Gradient Boosting; ${f3(U.LSTM)} on ${U.days} new days.`);
  s.addNotes("Two checks prompted by the mock viva. First, stronger baselines on the original test year: persistence (today's demand carried to tomorrow) and Gradient Boosting " +
    `(500 trees, rate 0.05, depth 4: one setting we picked, not a search). The LSTM beats Gradient Boosting by ${pct(GB, LS.MAE)}% and Random Forest by ${pct(RF.MAE, LS.MAE)}%. ` +
    "Second, a period nobody had looked at: NASA POWER weather for Nagpur after the original data, demand computed with the same FAO-56 code, models trained on all 2,000 original days. " +
    `On those ${U.days} days the LSTM scores ${f3(U.LSTM)}, Random Forest ${f2(U.RF)}, Linear Regression ${f2(U.LR)}, persistence ${f2(U.persistence)}. ` +
    `The bootstrap resamples 14-day blocks of the original test year 10,000 times; the LSTM is closer to the truth on ${Math.round(BS.share_of_days_lstm_closer * 100)}% of days. ` +
    "Sources: src/robustness_checks.py and src/viva_followups.py.");
}

// ================================================================= 4. New site
{
  const s = content("New site: IIT Hyderabad", 1);
  const cats = ["Nagpur, trained there", "Hyderabad, Nagpur model", "Hyderabad, retrained"];
  const rows = [nagT, hydT, hydR];
  chartTitle(s, "Average error, real weather (mm/day)", MX, 1.35, 8.1);
  s.addChart(pres.charts.BAR, [
    { name: "Random Forest", labels: cats, values: rows.map((r) => r.RF) },
    { name: "LSTM", labels: cats, values: rows.map((r) => r.LSTM) },
  ], chartOpts({
    x: MX, y: 1.75, w: 8.1, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [HEX.mid, HEX.teal], barGapWidthPct: 60,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisHidden: true, valGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 0.75, showLegend: true, legendPos: "t",
  }));
  const x = 9.1, w = W - MX - x;
  statCard(s, x, 1.75, w, 2.0, `${fmtInt(CTX.hyderabad.annual_rain_mm)} mm`, `annual rain at Hyderabad (Nagpur ${fmtInt(CTX.nagpur.annual_rain_mm)}): a drier, windier site`, { bigSize: 30 });
  statCard(s, x, 4.0, w, 2.0, `${f1(CTX.hyderabad.mean_demand)} mm/day`, `average daily demand at Hyderabad (Nagpur ${f1(CTX.nagpur.mean_demand)})`, { bigSize: 30 });
  takeaway(s, `**The LSTM works at a new site**: trained at Nagpur, it scores ${f2(hydT.LSTM)} mm/day at Hyderabad; retrained there, ${f2(hydR.LSTM)}.`);
  s.addNotes(`We download NASA POWER weather for the IIT Hyderabad campus (${CTX.hyderabad.lat}°N, ${CTX.hyderabad.lon}°E, ${Math.round(CTX.hyderabad.elevation_m)} m) for the same 2,000 days ` +
    "and compute demand with the same FAO-56 pipeline and maize calendar. Then we test the Nagpur models on Hyderabad's last 400 days without retraining, " +
    `and compare with models trained on Hyderabad's first 1,600 days. Random Forest loses more when moved (${f2(hydT.RF)} against ${f2(hydR.RF)} retrained), ` +
    `plausibly because its trees carve the inputs into Nagpur's usual combinations and Hyderabad combines them differently (windier: mean ${f1(HYD_WIND)} against ${f1(NAG_WIND)} m/s), although 99% of Hyderabad's test days lie inside Nagpur's range for every input. ` +
    `Relative to mean demand at Hyderabad (${f1(CTX.hyderabad.test_mean_demand)} mm/day in the test period), the transferred LSTM error is about ${Math.round(hydT.LSTM / CTX.hyderabad.test_mean_demand * 100)}%. ` +
    `Part 3 re-trains the Nagpur LSTM with a newer PyTorch version: ${f3(nagT.LSTM)} mm/day against Part 2's ${f3(LS.MAE)}, within the spread between seeds.`);
}

// ================================================================= 5. New crop
{
  const s = content("New crop: cotton", 2);
  const half = (CW - 0.4) / 2, x2 = MX + half + 0.4;
  const kcAt = (d, L, k) => {
    const [l1, l2, l3, l4] = L, [ini, mid, end] = k, c = d % (l1 + l2 + l3 + l4);
    return c < l1 ? ini : c < l1 + l2 ? ini + (c - l1) / l2 * (mid - ini) : c < l1 + l2 + l3 ? mid : mid + (c - l1 - l2 - l3) / l4 * (end - mid);
  };
  const days = Array.from({ length: 39 }, (_, i) => i * 5); // 0-190: one cotton season
  chartTitle(s, "Crop water factor K_{c}, by day after sowing", MX, 1.35, half);
  s.addChart(pres.charts.LINE, [
    { name: "Maize (125-day seasons)", labels: days.map((d) => (d % 50 ? "" : String(d))), values: days.map((d) => kcAt(d, [20, 35, 40, 30], [0.3, 1.2, 0.6])) },
    { name: "Cotton (195-day season)", labels: days.map((d) => (d % 50 ? "" : String(d))), values: days.map((d) => kcAt(d, [30, 50, 60, 55], CTX.cotton.kc)) },
  ], chartOpts({
    x: MX, y: 1.75, w: half, h: 4.25, chartColors: [HEX.mid, HEX.teal], lineSize: 2.5, lineDataSymbol: "none",
    valAxisMinVal: 0, valAxisMaxVal: 1.4, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "t", catAxisLabelRotate: 0,
  }));
  const cats = ["Maize, maize model", "Cotton, maize model", "Cotton, retrained"];
  const rows = [nagT, cotT, cotR];
  chartTitle(s, "Average error, real weather (mm/day)", x2, 1.35, half);
  s.addChart(pres.charts.BAR, [
    { name: "Random Forest", labels: cats, values: rows.map((r) => r.RF) },
    { name: "LSTM", labels: cats, values: rows.map((r) => r.LSTM) },
  ], chartOpts({
    x: x2, y: 1.75, w: half, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [HEX.mid, HEX.teal], barGapWidthPct: 60,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisHidden: true, valGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 0.7, showLegend: true, legendPos: "t", catAxisLabelFontSize: 11,
  }));
  takeaway(s, `**A maize model works for cotton**: cotton's K_{c} stays inside maize's range (LSTM ${f2(cotT.LSTM)}; cotton-trained ${f2(cotR.LSTM)}).`);
  s.addNotes("Cotton is the main crop of the Vidarbha region around Nagpur. We keep Nagpur's real weather and swap the maize calendar for the FAO-56 cotton calendar " +
    `(30/50/60/55 days, Kc ${CTX.cotton.kc.join(" → ")}). Cotton's Kc range lies inside maize's (0.3–1.2), so the maize-trained models only need to read a new Kc curve. ` +
    "Cotton seasons are longer, so 1,600 training days hold about 8 cotton seasons against 13 maize seasons; fewer examples plausibly explain why the cotton-trained " +
    `models do slightly worse than the maize-trained ones. Random Forest: ${f2(cotT.RF)} transferred, ${f2(cotR.RF)} retrained. ` +
    `This is interpolation in Kc, not a new range: asked about Kc 1.3, the forest returns exactly its Kc 1.2 answer (${f2(VR2.rf_at_kc['1.3'])} against ${f2(VR2.rf_at_kc['1.2'])} mm/day).`);
}

// ================================================================= 6. Days ahead
{
  const s = content("Predicting days ahead", 3);
  const hs = ["1", "3", "7"], cats = hs.map((h) => `${h} ${h === "1" ? "day" : "days"} ahead`);
  const CLIM = VF.climatology_mae.typical_et0_x_kc_no_rain; // typical ET0 for the date x Kc on the target day, no rain
  chartTitle(s, "Average error with no weather forecast (mm/day)", MX, 1.35, 8.1);
  s.addChart(pres.charts.BAR, [
    { name: "Copy today's demand", labels: cats, values: hs.map((h) => AHEAD[h].persistence) },
    { name: "Usual value for the date", labels: cats, values: hs.map(() => CLIM) },
    { name: "Random Forest", labels: cats, values: hs.map((h) => AHEAD[h].RF) },
    { name: "LSTM", labels: cats, values: hs.map((h) => AHEAD[h].LSTM) },
  ], chartOpts({
    x: MX, y: 1.75, w: 8.1, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [HEX.light, GREY, HEX.mid, HEX.teal], barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10, valAxisHidden: true, valGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 2.2, showLegend: true, legendPos: "t",
  }));
  const x = 9.1, w = W - MX - x;
  statCard(s, x, 1.75, w, 2.0, f2(AHEAD["0"].LSTM), "LSTM for today, using today's weather: about what a perfect forecast would allow", { bigSize: 30, bigColor: C.accent2 });
  statCard(s, x, 4.0, w, 2.0, f2(CLIM), "usual value for the date (typical weather × K_{c}): beats both models at 7 days", { bigSize: 30 });
  takeaway(s, "**Without a forecast, the models barely beat simple rules**: tied with copying today at 1 day, behind the usual value at 7.");
  const gain = (h) => pct(AHEAD[h].persistence, Math.min(AHEAD[h].RF, AHEAD[h].LSTM));
  s.addNotes("We predict the demand h days ahead from the weather of the last seven days plus the crop stage on the target day, which is known from the sowing date. " +
    "Persistence carries today's FAO-56 demand forward. Rain is the problem: past weather cannot tell whether it will rain tomorrow, and one rainy day changes demand sharply. " +
    `At 3 and 7 days the models beat persistence by ${gain("3")}% and ${gain("7")}%, because they know the crop stage and the season, but a calendar climatology ` +
    `(typical ET0 for the date times Kc on the target day) scores ${f2(CLIM)}: it ties the models at 3 days and beats them at 7. The models did not get today's demand as an input; persistence has it. ` +
    "The gap between the same-day error and these numbers is the value of a weather forecast: a deployed scheduler should take IMD forecasts as inputs.");
}

// ================================================================= 7. Noisier records
{
  const s = content("Imperfect records", 4);
  const keys = Object.keys(NOISY), cats = keys.map((k) => `Typical error ${k} mm/day`);
  chartTitle(s, "Average error against the true FAO-56 demand (mm/day)", MX, 1.35, 8.1);
  s.addChart(pres.charts.BAR, [
    { name: "The records themselves", labels: cats, values: keys.map((k) => NOISY[k].record_error) },
    { name: "Random Forest", labels: cats, values: keys.map((k) => NOISY[k].RF) },
    { name: "LSTM", labels: cats, values: keys.map((k) => NOISY[k].LSTM) },
  ], chartOpts({
    x: MX, y: 1.75, w: 8.1, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [HEX.light, HEX.mid, HEX.teal], barGapWidthPct: 55,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisHidden: true, valGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 0.9, showLegend: true, legendPos: "t",
  }));
  const x = 9.1, w = W - MX - x, lo = NOISY[keys[0]], hi = NOISY[keys[keys.length - 1]];
  statCard(s, x, 1.75, w, 2.0, `${f2(hi.LSTM)}`, `LSTM error when records are off by 1.0 mm/day; the records' own error is ${f2(hi.record_error)}`, { bigSize: 30 });
  statCard(s, x, 4.0, w, 2.0, `×${f1(lo.RF / lo.LSTM)} → ×${f1(hi.RF / hi.LSTM)}`, "how many times smaller the LSTM's error is than Random Forest's, cleanest to messiest records", { bigSize: 30 });
  takeaway(s, `**Messier records shrink the LSTM's lead** (×${f1(lo.RF / lo.LSTM)} → ×${f1(hi.RF / hi.LSTM)}), yet both models land closer to the truth than the records.`);
  s.addNotes("Measured irrigation would not follow the FAO-56 formula exactly. We add noise to every target (0.18 mm/day as in Parts 1–2, then 0.5 and 1.0 mm/day, about 10% and 20% of mean demand), " +
    "train on the noisy targets and score against the noise-free FAO-56 demand. The 'noisy record' bars show how far the records themselves are from the truth. " +
    "Both models average the noise out, so their predictions sit closer to the true demand than the records do. The LSTM's advantage, which came partly from fitting a smooth formula, " +
    "shrinks as the noise grows, as Part 2 expected.");
}

// ================================================================= 12. Week by week
section("Practical use");
const B = SHOW.water_budget_mm, SM = SHOW.season_mae, PK = SHOW.peak_days, SE = SHOW.cheap_sensors, HI = SHOW.history_length;
{
  const s = content("The test year, week by week", 5);
  const wk = SHOW.weekly;
  const labels = wk.week_start.map((d, i) => {
    const day = +d.slice(8, 10), m = +d.slice(5, 7) - 1;
    return i === 0 || day <= 7 ? `${MONTHS[m]}${m === 0 || i === 0 ? " " + d.slice(2, 4) : ""}` : "";
  });
  chartTitle(s, "Average demand each week over the 400 test days (mm/day)", MX, 1.35, CW);
  s.addChart(pres.charts.LINE, [
    { name: "FAO-56 (true value)", labels, values: wk.actual },
    { name: "Random Forest", labels, values: wk["Random Forest"] },
    { name: "LSTM", labels, values: wk.LSTM },
  ], chartOpts({
    x: MX, y: 1.75, w: CW, h: 4.3, chartColors: [HEX.ink, HEX.mid, HEX.amber], lineSize: 2.25, lineDataSymbol: "none",
    valAxisMinVal: 0, valAxisMaxVal: 16, valAxisMajorUnit: 4, valAxisLabelFormatCode: "0", showLegend: true, legendPos: "t",
    catAxisLabelFrequency: 1, catAxisLabelRotate: 0, catAxisLabelFontSize: 11,
  }));
  const may = (k) => { // mean over the May 2024 test days
    const rows = wk.week_start.map((d, i) => [d, wk[k][i]]).filter(([d]) => d >= "2024-04-29" && d < "2024-06-03");
    return rows.reduce((a, r) => a + r[1], 0) / rows.length;
  };
  const wmae = (k) => wk[k].reduce((a, v, i) => a + Math.abs(v - wk.actual[i]), 0) / wk.actual.length;
  takeaway(s, `**Both models follow the seasons**, but at the May 2024 peak Random Forest is ${f1(may("actual") - may("Random Forest"))} mm/day too low; the LSTM is not.`);
  s.addNotes(`Weekly means smooth out day-to-day noise. Mean weekly error: Random Forest ${f2(wmae("Random Forest"))}, LSTM ${f2(wmae("LSTM"))} mm/day. ` +
    "Demand is highest in the hot, dry weeks before the monsoon (April to early June), lowest in the July monsoon and in November. " +
    "Random Forest predictions are clipped at its training range, which shows at the May 2024 peak.");
}

// ================================================================= 13. Water budget
{
  const s = content("Seasonal water totals", 5);
  const seasons = ["Monsoon", "Post-monsoon and winter", "Pre-monsoon", "Whole test period"];
  const slab = { "Monsoon": "Monsoon", "Post-monsoon and winter": "Post-monsoon, winter", "Pre-monsoon": "Pre-monsoon", "Whole test period": "All 400 days" };
  const err = (m, se) => Math.round((B[se][m] / B[se].actual_mm - 1) * 1000) / 10;
  chartTitle(s, "Error in the total water predicted for each season (%)", MX, 1.35, 8.1);
  s.addChart(pres.charts.BAR, [
    { name: "Linear Regression", labels: seasons.map((x) => slab[x]), values: seasons.map((x) => err("Linear Regression", x)) },
    { name: "Decision Tree", labels: seasons.map((x) => slab[x]), values: seasons.map((x) => err("Decision Tree", x)) },
    { name: "Random Forest", labels: seasons.map((x) => slab[x]), values: seasons.map((x) => err("Random Forest", x)) },
    { name: "LSTM", labels: seasons.map((x) => slab[x]), values: seasons.map((x) => err("LSTM", x)) },
  ], chartOpts({
    x: MX, y: 1.75, w: 8.1, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [HEX.light, GREY, HEX.mid, HEX.amber], barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0;0.0", dataLabelFontSize: 10,
    valAxisMinVal: -10, valAxisMaxVal: 10, valAxisMajorUnit: 5, valAxisLabelFormatCode: "+0;-0;0", showLegend: true, legendPos: "t",
    catAxisLabelPos: "low",
  }));
  const x = 9.1, w = W - MX - x, tot = B["Whole test period"].actual_mm;
  statCard(s, x, 1.75, w, 2.0, `${fmtInt(tot)} mm`, `water needed over the 400 test days, about ${fmtInt(Math.round(tot * 10 / 100) * 100)} m³ per hectare`, { bigSize: 30 });
  const worst = Math.max(...seasons.map((x) => Math.abs(err("LSTM", x))));
  statCard(s, x, 4.0, w, 2.0, `±${Math.ceil(worst)}%`, "LSTM's error in the water total, in every season", { bigSize: 30, bigColor: C.accent2 });
  const maxTot = Math.max(...["Linear Regression", "Decision Tree", "Random Forest", "LSTM"].map((m) => Math.abs(err(m, "Whole test period"))));
  takeaway(s, `**Totals are easy, seasons are harder**: all within ${f1(maxTot)}% over 400 days, but Linear Regression is ${f1(-err("Linear Regression", "Pre-monsoon"))}% short before the monsoon.`);
  s.addNotes("We sum the daily predictions over each season of the test year and compare with the FAO-56 total. One millimetre over one hectare is 10 cubic metres. " +
    `Pre-monsoon: actual ${fmtInt(B["Pre-monsoon"].actual_mm)} mm, Linear Regression ${fmtInt(B["Pre-monsoon"]["Linear Regression"])}, Random Forest ${fmtInt(B["Pre-monsoon"]["Random Forest"])}, LSTM ${fmtInt(B["Pre-monsoon"].LSTM)}. ` +
    "Errors on individual days cancel over a season, so totals hide what matters for scheduling: the linear model over-irrigates in the monsoon and under-irrigates in the peak. " +
    "Predictions are clipped at zero; the LSTM is the seed-averaged prediction from src/showcase_analysis.py.");
}

// ================================================================= 14. Error heatmap
{
  const s = content("Where the errors fall", 5);
  const cols = [["Monsoon", "Monsoon"], ["Post-monsoon and winter", "Post-monsoon, winter"], ["Pre-monsoon", "Pre-monsoon"], ["peak", `Peak days (${PK.days})`]];
  const models = ["Linear Regression", "Decision Tree", "Random Forest", "LSTM"];
  const val = (m, c) => c === "peak" ? PK.mae[m] : SM[c][m];
  const maxV = 2.0;
  const hdr = (t) => ({ text: t, options: { fontFace: BOLD, color: HEX.teal, fill: { color: HEX.white }, align: "center" } });
  const rows = [[{ text: "Average error, mm/day", options: { fontFace: BOLD, color: HEX.teal, fill: { color: HEX.white } } }, ...cols.map((c) => hdr(c[1]))]];
  for (const m of models) {
    rows.push([{ text: m, options: { fill: { color: HEX.white }, fontFace: m === "LSTM" ? BOLD : THEME.bodyFontFace } }, ...cols.map(([c]) => {
      const v = val(m, c), t = Math.min(1, v / maxV) * 0.92;
      return { text: f2(v), options: { align: "center", fill: { color: hexMix(t) }, color: t > 0.5 ? HEX.white : HEX.ink, fontFace: t > 0.5 ? BOLD : THEME.bodyFontFace } };
    })]);
  }
  s.addTable(rows, { x: MX, y: 1.5, w: 8.1, colW: [2.1, 1.5, 1.5, 1.5, 1.5], rowH: 0.8, fontSize: 15, color: HEX.ink,
    fontFace: THEME.bodyFontFace, valign: "middle", border: { type: "solid", color: HEX.white, pt: 2 }, margin: [0, 0.12, 0, 0.12] });
  const x = 9.1, w = W - MX - x;
  panel(s, x, 1.5, w, 4.0, "Definitions");
  text(s, `**Peak days** · the ${PK.days} days with the highest demand (above ${f1(PK.threshold_mm)} mm/day), when water matters most\n` +
    "**Shading** · the darker the cell, the larger the error\n**Seasons** · monsoon Jun–Sep, post-monsoon and winter Oct–Feb, pre-monsoon Mar–May\n" +
    `**LSTM** · the only model below 0.3 mm/day in every column`,
    { x: x + 0.3, y: 1.75, w: w - 0.6, h: 3.6, size: 14, paraSpaceAfter: 10 });
  takeaway(s, `**The peak days show the real gap**: there the LSTM is off by ${f2(PK.mae.LSTM)} mm/day, Random Forest ${f2(PK.mae["Random Forest"])}, Linear Regression ${f2(PK.mae["Linear Regression"])}.`);
  s.addNotes("All models are most accurate in the post-monsoon and winter months, when demand is moderate and steady. The pre-monsoon peak is hardest for the classical models: " +
    `Linear Regression's error there is ${f2(SM["Pre-monsoon"]["Linear Regression"])} mm/day. The LSTM error stays between ${f2(Math.min(...cols.map(([c]) => val("LSTM", c))))} and ${f2(Math.max(...cols.map(([c]) => val("LSTM", c))))} mm/day in every column, ` +
    "including the peak days. LSTM values are means over three seeds.");
}

// ================================================================= 15. Cheap sensors
{
  const s = content("Cheaper weather stations", 5);
  const names = Object.keys(SE);
  const lab = { "All 7 inputs": "Full station (7 inputs)", "No wind or solar sensor": "No wind or solar sensor",
    "Rain gauge + max/min thermometer": "Rain gauge + max/min thermometer", "Rain gauge + thermometer": "Rain gauge + thermometer" };
  const order = [...names].reverse();
  chartTitle(s, "Average error by set of sensors (mm/day)", MX, 1.35, 8.1);
  s.addChart(pres.charts.BAR, [
    { name: "Random Forest", labels: order.map((n) => lab[n]), values: order.map((n) => SE[n].RF) },
    { name: "LSTM", labels: order.map((n) => lab[n]), values: order.map((n) => SE[n].LSTM) },
  ], chartOpts({
    x: MX, y: 1.75, w: 8.1, h: 4.25, barDir: "bar", barGrouping: "clustered", chartColors: [HEX.mid, HEX.teal], barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 11, valAxisHidden: true,
    valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 1.1, showLegend: true, legendPos: "t", catAxisLabelFontSize: 12,
  }));
  // right: the same cheap sensors, against FAO-56's own temperature-only method (Hargreaves)
  const HG = SHOW.hargreaves_baseline, mm = SE["Rain gauge + max/min thermometer"];
  const x = 9.1, w = W - MX - x;
  chartTitle(s, "Rain gauge + max/min thermometer", x, 1.35, w);
  const hcats = ["LSTM", "Random Forest", "Hargreaves, tuned", "Hargreaves, as printed"]; // horizontal bars draw the first at the bottom
  s.addChart(pres.charts.BAR, [{ name: "MAE", labels: hcats, values: [mm.LSTM, mm.RF, HG.calibrated, HG.raw] }], chartOpts({
    x, y: 1.75, w, h: 4.25, barDir: "bar", chartColors: [HEX.teal, HEX.mid, HEX.light, HEX.light], barGapWidthPct: 45, showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 11, valAxisHidden: true,
    valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 1.0, catAxisLabelFontSize: 11,
  }));
  takeaway(s, `**With cheap sensors, ML only ties FAO-56's simple formula** (both ${f2(mm.LSTM)}); it pulls ahead with a full station.`);
  s.addNotes("Left: we retrain Random Forest and LSTM with fewer inputs, as if the farm had a cheaper station; the crop coefficient and growth stage are always known. " +
    "Wind and solar radiation carry real information: without them the error roughly triples. A max/min thermometer does better than a mean-only one, " +
    "because a large daily range means clear skies and strong sunshine. Right: FAO-56 has its own method for exactly this case, the Hargreaves equation (Eq. 52), " +
    `which estimates ET0 from Tmax, Tmin and extraterrestrial radiation. Used as printed it gives ${f2(HG.raw)} mm/day; calibrated on the same 1,600 training days ` +
    `(ET0 = ${f2(HG.calibration.intercept)} + ${f2(HG.calibration.slope)} × ET0_Hargreaves), it gives ${f2(HG.calibrated)}, the same as our models (${f2(mm.LSTM)}). ` +
    "So with cheap sensors the models bring no gain over the established method; their advantage appears only with the full set of readings.");
}

// ================================================================= 16. Years of records
{
  const s = content("How many years of records", 5);
  const keys = Object.keys(HI);
  const labels = keys.map((k) => `${HI[k].years} ${HI[k].years === 1 ? "year" : "years"}`);
  chartTitle(s, "Average error by years of training data (mm/day)", MX, 1.35, 8.1);
  s.addChart(pres.charts.BAR, [
    { name: "Random Forest", labels, values: keys.map((k) => HI[k].RF) },
    { name: "LSTM", labels, values: keys.map((k) => HI[k].LSTM) },
  ], chartOpts({
    x: MX, y: 1.75, w: 8.1, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [HEX.mid, HEX.teal], barGapWidthPct: 60,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisHidden: true, valGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 0.75, showLegend: true, legendPos: "t",
  }));
  const x = 9.1, w = W - MX - x;
  statCard(s, x, 1.75, w, 2.0, f2(HI["730"].LSTM), `LSTM with 2 years of records\n(Random Forest with 4.4 years: ${f2(HI["1600"].RF)})`, { bigSize: 30, bigColor: C.accent2 });
  statCard(s, x, 4.0, w, 2.0, `−${pct(HI["365"].LSTM, HI["1600"].LSTM)}%`, "drop in LSTM error from 1 year to 4.4 years of data", { bigSize: 30 });
  takeaway(s, `**The LSTM needs less data**: with 2 years (${f2(HI["730"].LSTM)}) it beats Random Forest with 4.4 years (${f2(HI["1600"].RF)}), useful where records are short.`);
  s.addNotes("We train on the most recent 1, 2, 3 or 4.4 years before the test period and test on the same 400 days. Both models keep improving with more history, " +
    `the LSTM faster: ${f2(HI["365"].LSTM)} → ${f2(HI["1600"].LSTM)} mm/day, Random Forest ${f2(HI["365"].RF)} → ${f2(HI["1600"].RF)}. ` +
    "With one year the LSTM varies more between seeds (± " + f2(HI["365"].LSTM_std) + "), since one year holds a single pre-monsoon peak.");
}

// ================================================================= Project summary
section("Conclusions");
{
  const s = content("Project summary: three parts");
  const cols = [
    ["Part 1  ·  27 Sep", `R² ${f3(V1["Random Forest"].R2)}`, "Random Forest, 8 inputs",
      ["**Data** · made-up weather, FAO-56 demand", "**Models** · Linear Regression, Decision Tree, Random Forest", "**Found** · the model mostly copied yesterday's demand"]],
    ["Part 2  ·  5 Oct", `${f3(LS.MAE)} mm/day`, "LSTM on real weather",
      ["**Fixed** · that input removed, 7 inputs left", "**Models** · three neural networks: MLP, 1D-CNN, LSTM", "**Checked** · with real NASA weather"]],
    ["Part 3  ·  Nov", `${f2(hydT.LSTM)} mm/day`, "LSTM at a new site, no retraining",
      ["**Tested** · new site, new crop, days ahead, imperfect records", "**Measured** · water totals, peak days, cheaper sensors", "**Checked** · tougher rivals and 871 new days"]],
  ];
  const cw = (CW - 2 * 0.3) / 3, y = 1.4, h = 4.55;
  cols.forEach(([t, big, sub, items], i) => {
    const x = MX + i * (cw + 0.3);
    panel(s, x, y, cw, h, `${t} panel`, i === 2 ? C.accent1 : C.background2);
    const fg = i === 2 ? C.background1 : C.accent1, body = i === 2 ? C.background1 : C.text1;
    s.addText(t, { x: x + 0.3, y: y + 0.2, w: cw - 0.6, h: 0.4, fontSize: 16, fontFace: BOLD, color: fg, margin: 0, isTextBox: true });
    const [num, unit] = big.split(" mm/day");
    s.addText([{ text: num, options: { fontSize: 36, fontFace: BOLD } }, ...(unit !== undefined ? [{ text: " mm/day", options: { fontSize: 18, fontFace: BOLD } }] : [])],
      { x: x + 0.3, y: y + 0.65, w: cw - 0.6, h: 0.75, color: i === 2 ? C.background1 : C.accent1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(sub, { x: x + 0.3, y: y + 1.4, w: cw - 0.6, h: 0.4, fontSize: 14, color: i === 2 ? C.accent4 : C.accent5, margin: 0, isTextBox: true });
    text(s, items.join("\n"), { x: x + 0.3, y: y + 2.0, w: cw - 0.6, h: h - 2.15, size: 14, color: body, paraSpaceAfter: 8 });
  });
  takeaway(s, "**Each part fixed a weakness of the one before**: copying yesterday's demand, made-up weather, one site and crop.");
  s.addNotes("The whole project on one slide. Part 1 built the pipeline and found that the model leaned on yesterday's demand. Part 2 removed that input, added deep learning and real weather. " +
    "Part 3 tested whether the result carries over to a new site and crop, what happens days ahead and with noisy records, and measured what the model offers in practice.");
}

// ================================================================= Conclusions
{
  const s = content("Conclusions");
  const items = [
    [`**It works at a new site**, as expected when the target is a formula: the Nagpur LSTM scores ${f2(hydT.LSTM)} mm/day at Hyderabad; retrained there, ${f2(hydR.LSTM)}.`, f2(hydT.LSTM)],
    [`**A maize model works for cotton**, because cotton's K_{c} stays inside maize's range: LSTM ${f2(cotT.LSTM)} mm/day, no worse than one trained on cotton.`, f2(cotT.LSTM)],
    [`**Predicting days ahead needs a weather forecast**: without one, no model beats copying today at 1 day (${f2(AHEAD["1"].persistence)}) or the usual value at 7 days (${f2(VF.climatology_mae.typical_et0_x_kc_no_rain)}).`, f2(AHEAD["1"].LSTM)],
    [`**Messier records shrink the LSTM's lead**: from ×${f1(NOISY["0.18"].RF / NOISY["0.18"].LSTM)} to ×${f1(NOISY["1.0"].RF / NOISY["1.0"].LSTM)} when records are off by 1 mm/day; records that are always too high or too low are untested.`, f2(NOISY["1.0"].LSTM)],
    ["**Useful with a full weather station**: season totals within ±2%, and two years of data are enough; with cheap sensors, FAO-56's simple Hargreaves formula does as well.", "±2%"],
  ];
  const y0 = 1.4, rh = 0.92, gap = 0.15;
  items.forEach(([d, k], i) => {
    const y = y0 + i * (rh + gap);
    panel(s, MX, y, CW, rh, `Conclusion ${i + 1}`);
    badge(s, i + 1, MX + 0.22, y + (rh - 0.56) / 2, 0.56, { size: 16 });
    text(s, d, { x: MX + 1.05, y, w: 9.0, h: rh, size: 15, valign: "middle" });
    s.addText(k, { x: MX + 10.1, y, w: CW - 10.1 - 0.25, h: rh, fontSize: 26, fontFace: BOLD, color: i === 0 ? C.accent2 : C.accent1,
      align: "right", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addNotes("Each conclusion answers the Part 3 component with the same number. Claim first, then the evidence. " +
    "Numbers on the right: transferred LSTM error at Hyderabad, maize-trained LSTM error on cotton, LSTM error one day ahead without a forecast, " +
    "LSTM error at the noisiest records, and the LSTM's worst seasonal volume error.");
}

// ================================================================= What we learned
{
  const s = content("What we learned");
  const lessons = [
    ["Check what the model relies on", `Part 1's model mostly copied yesterday's demand (scrambling it cut R² by ${lagDrop}); without it, R² only fell from ${f3(V1["Random Forest"].R2)} to ${f3(RFs.R2)}.`],
    ["Test outside the training data", "Real weather, a new site and a new crop each showed what still works and what does not."],
    ["Measure what the farmer needs", "Every model got the season totals right; only the peak days showed the real differences."],
    ["Know the model's limits", "Predicting days ahead needs a forecast; with cheap sensors, FAO-56's simple formula does as well."],
  ];
  const cw = (CW - 0.35) / 2, ch = 2.2;
  lessons.forEach(([t, d], i) => {
    const x = MX + (i % 2) * (cw + 0.35), y = 1.45 + Math.floor(i / 2) * (ch + 0.35);
    panel(s, x, y, cw, ch, `Lesson ${i + 1}`);
    badge(s, i + 1, x + 0.3, y + 0.3, 0.5, { size: 15 });
    s.addText(t, { x: x + 1.0, y: y + 0.3, w: cw - 1.3, h: 0.5, fontSize: 18, fontFace: BOLD, color: C.accent1, valign: "middle", margin: 0, isTextBox: true });
    text(s, d, { x: x + 1.0, y: y + 0.95, w: cw - 1.3, h: ch - 1.1, size: 15 });
  });
  s.addNotes("Four lessons from the three parts. The first came from Part 1, the second from Parts 2 and 3, the last two from Part 3. " +
    "The last one matters for anyone deploying this: the model is limited by what the weather inputs know, not by the network.");
}

// ================================================================= Limitations
{
  const s = content("Limitations and next steps");
  const half = (CW - 0.4) / 2;
  const col = (x, title, items) => {
    panel(s, x, 1.4, half, 4.55, `${title} panel`);
    s.addText(title, { x: x + 0.3, y: 1.6, w: half - 0.6, h: 0.45, fontSize: 18, fontFace: BOLD, color: C.accent1, margin: 0, isTextBox: true });
    bullets(s, items, x + 0.3, 2.2, half - 0.6, 3.6, 15, 12);
  };
  col(MX, "Still open", [
    "**No soil water store**: rain that soaks into the soil is not carried over to the next days (FAO-56 Ch. 8)",
    "**Rainfall formula**: made for monthly totals but used daily, so any rain under 16.7 mm counts as zero",
    "**Back-to-back seasons**: maize grows all year, even in April and May, with no rest period",
    "**Calculated, not measured** demand; weather from satellite and model grids, not a farm station",
  ]);
  col(MX + half + 0.4, "Next steps", [
    "Rebuild the target as a daily soil water balance, with real local sowing dates",
    "Check our ET0 against FAO's CLIMWAT data, then against field measurements",
    "Add IMD weather forecasts to predict 1–7 days ahead",
    "Train and test on measured irrigation or soil-moisture records from a research farm",
  ]);
  takeaway(s, "**The real test for ML is measured data**: there it can learn what the formula misses; here it can only copy the formula.");
  s.addNotes("The biggest open issue is the target itself: it has no root-zone store, and the rainfall formula (the FAO/AGLW dependable-rain formula, which our earlier decks wrongly called USDA-SCS) " +
    "was built for monthly totals; used daily it counts 522 of 659 real rain days as zero. Rebuilding the target as an FAO-56 Chapter 8 water balance, with irrigation triggered at RAW, " +
    "would turn the output into irrigation dates and depths and give the LSTM's memory a real job. Measured data and an on-farm station remain out of reach for this project.");
}

// ================================================================= 21. References
{
  const s = content("References");
  const refs = [
    "Allen, R. G., Pereira, L. S., Raes, D., Smith, M. (1998). Crop evapotranspiration: guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56, FAO, Rome.",
    "Smith, M. (1992). CROPWAT: a computer program for irrigation planning and management. FAO Irrigation and Drainage Paper 46, FAO, Rome (effective-rainfall methods).",
    "Hargreaves, G. H., Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture, 1(2), 96–99.",
    "NASA Langley Research Center. POWER Project, daily agroclimatology data (AG community). https://power.larc.nasa.gov",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
    "Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.",
    "Fisher, A., Rudin, C., Dominici, F. (2019). All models are wrong, but many are useful. JMLR, 20(177), 1–81.",
    "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. JMLR, 12, 2825–2830.",
    "Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. NeurIPS 32.",
  ];
  s.addText(refs.map((r, i) => ({ text: `[${i + 1}]  ${r}`, options: { fontSize: 14, color: C.text1, paraSpaceAfter: 8,
    breakLine: i < refs.length - 1 } })), { x: MX, y: 1.4, w: CW, h: 4.85, valign: "top", margin: 0, isTextBox: true });
  text(s, "Code, data and results: github.com/Akhil-Nistala/irrigation-ml-demand", { x: MX, y: 6.35, w: CW, h: 0.4, size: 14, color: C.accent5 });
  s.addNotes("All numbers in this deck come from the scripts and notebooks in the repository: src/no_lag_experiment.py for the model comparison and src/showcase_analysis.py " +
    "for the practical-use slides, src/part3_experiments.py, src/robustness_checks.py and src/viva_followups.py for the Part 3 tests and checks (random_state 42, seeds 0 to 2).");
}

// ================================================================= 22. Questions
{
  const s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: curSection });
  s.addText("Thank you", { placeholder: "title" });
  s.addText("Questions and discussion", { placeholder: "body" });
  badge(s, "?", 0.95, 2.45, 1.35, { fill: C.background2, color: C.accent1, size: 48 });
  s.addNotes("Likely questions. Why synthetic data first? No field dataset had all the variables; Experiment 3 replaces it with real weather. " +
    "Why drop yesterday's demand? The model leaned on it through persistence and it is not available for forecasting; models without it do almost as well. " +
    "Why a chronological split? To stop near-copies of test days leaking into training. Is the LSTM overfitting? No: see the backup slide; train-test gap about 0.06 mm/day.");
}

// ================================================================= Backup slides
section("Backup");
{
  const s = content("Where our target differs from FAO-56", "A");
  const hdr = (t) => ({ text: t, options: { fontFace: BOLD, color: HEX.white, fill: { color: HEX.teal } } });
  const rowsD = [
    ["No soil water store", "all data", "rain is not carried over; monsoon demand too high"],
    ["Monthly rainfall formula (FAO/AGLW) used daily; earlier decks called it USDA-SCS", "all data", "rain under 16.7 mm counts as zero; only 15% of real rain counts"],
    ["Humidity term simplified (FAO-56 Eq. 19 not used)", "all data", "ET0 2.5% too high, same day-to-day pattern"],
    ["ET0 capped at 8 mm/day", "made-up weather only", "demand cannot go above 12.8 mm/day in April–May"],
    ["Kc not adjusted for local climate", "all data", "mid-season Kc off by about 0.1 either way"],
    ["Back-to-back seasons, no rest period", "all data", "maize in April–May; yearly totals too high"],
    ["Random error of 0.18 mm added, never below zero", "all data", "zero-demand days get small positive values"],
  ];
  const rows = [[hdr("What differs"), hdr("Where"), hdr("Effect")],
    ...rowsD.map((r, i) => r.map((c) => ({ text: c, options: { fill: { color: i % 2 ? HEX.white : HEX.panel } } })))];
  s.addTable(rows, { x: MX, y: 1.4, w: CW, colW: [5.0, 1.8, CW - 6.8], rowH: 0.56, fontSize: 14, color: HEX.ink,
    fontFace: THEME.bodyFontFace, valign: "middle", border: { type: "none" }, margin: [0, 0.12, 0, 0.12] });
  takeaway(s, "**None of these change which model is best**; the first two change what the demand numbers mean for a farmer.");
  s.addNotes("Listed after the mock viva so that nothing comes out only under questioning. Sources: report/robustness_checks.json (vapour pressure and rainfall audit), " +
    "report/viva_followups.json (cap counts). Eq. 62 would raise maize Kc mid by about 0.1 in the dry pre-monsoon and lower it by about 0.1 in the humid monsoon.");
}
{
  const s = content("New site and crop: data", "A");
  const cx = MX, cw = 7.3;
  chartTitle(s, "Average monthly demand for maize (mm/day)", cx, 1.35, cw);
  s.addChart(pres.charts.BAR, [
    { name: "Nagpur", labels: MONTHS.map((m) => m[0]), values: CTX.nagpur_monthly_demand },
    { name: "IIT Hyderabad", labels: MONTHS.map((m) => m[0]), values: CTX.hyderabad_monthly_demand },
  ], chartOpts({
    x: cx, y: 1.75, w: cw, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [HEX.light, HEX.teal], barGapWidthPct: 40,
    valAxisMinVal: 0, valAxisMaxVal: 12, valAxisMajorUnit: 3, valAxisLabelFormatCode: "0", showLegend: true, legendPos: "t",
  }));
  const x = cx + cw + 0.4, w = W - MX - x;
  panel(s, x, 1.4, w, 4.6, "Data facts");
  text(s, `**IIT Hyderabad** · ${CTX.hyderabad.lat}°N ${CTX.hyderabad.lon}°E, ${Math.round(CTX.hyderabad.elevation_m)} m, NASA POWER daily data, same 2,000 days\n` +
    `**Climate** · ${fmtInt(CTX.hyderabad.annual_rain_mm)} mm rain a year (Nagpur ${fmtInt(CTX.nagpur.annual_rain_mm)}), mean wind ${f1(HYD_WIND)} m/s (Nagpur ${f1(NAG_WIND)})\n` +
    `**Cotton** · FAO-56 growth stages of 30/50/60/55 days (195 in all), K_{c} ${CTX.cotton.kc.join(" → ")}\n` +
    "**Method** · same FAO-56 code; it rebuilds the Part 2 Nagpur data exactly",
    { x: x + 0.3, y: 1.65, w: w - 0.6, h: 4.2, size: 14, paraSpaceAfter: 12 });
  const jja = (a) => (a[6] + a[7]) / 2;
  takeaway(s, `**Hyderabad needs more water in July–August** (${f1(jja(CTX.hyderabad_monthly_demand))} against ${f1(jja(CTX.nagpur_monthly_demand))} mm/day), partly due to the daily rainfall formula.`);
  s.addNotes("Built by src/part3_experiments.py. The Hyderabad weather comes from the same NASA POWER service; the elevation (565 m) comes from the API. " +
    "The script checks that its FAO-56 builder reproduces the Part 2 Nagpur dataset before building the new ones.");
}

// ---------------------------------------------------------------- write
// pptxgenjs writes an <a:pPr> before every run of a multi-run paragraph; OOXML allows one, first.
// Keep the leading one and drop the rest, so bullets and spacing come from the paragraph's first run.
async function dedupeParagraphProps(file) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const pPr = /<a:pPr\b[^>]*?(?:\/>|>[\s\S]*?<\/a:pPr>)/g;
  let removed = 0;
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    zip.file(name, xml.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (m, body) => {
      let first = true;
      const out = body.replace(pPr, (tag, offset) => {
        if (first && offset === 0) { first = false; return tag; }
        first = false; removed += 1; return "";
      });
      return `<a:p>${out}</a:p>`;
    }));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  return removed;
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  const removed = await dedupeParagraphProps(OUT);
  if (process.env.PPTX_THEME_HELPER) await require(process.env.PPTX_THEME_HELPER).applyTheme(OUT, THEME);
  console.log("wrote", OUT, `(removed ${removed} duplicate paragraph-property tags)`);
})();
