// LDA, QDA, Gaussian naive Bayes and logistic regression on two Gaussian classes in 2-D
// (ISL 4.4–4.5, ESL 4.3, 4.4.5).
// Truth: X | Y = k ~ N(mu_k, Sigma_k), Pr(Y = 1) = pi1. Every method returns
// g(x) = (class-2 discriminant) − (class-1 discriminant) and says "class 2" when g(x) > 0.
//   LDA   pooled covariance with divisor n − K
//   QDA   class sample covariances, divisor n_k − 1 (sklearn 1.9 divides by n_k, which moves
//         the boundary slightly at small n_k)
//   NB    per-class, per-feature means and variances (divisor n_k, as sklearn's GaussianNB)
//   LR    Newton–Raphson on the log-likelihood with a tiny ridge (λ = 1e-4) on the slopes,
//         i.e. sklearn's LogisticRegression(C = 1e4)
// Training and test points are fixed standard normals pushed through x = mu_k + A_k z, so
// dragging a mean moves the same points smoothly.

// ---------- seeded RNG (mulberry32 + Box–Muller) ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function normalSampler(rand) {
  let spare = null;
  return function () {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0;
    while (u === 0) u = rand();
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}
// n draws of (z1, z2, u): two standard normals and a uniform for the class label.
function draws(n, seed) {
  const rand = mulberry32(seed);
  const rnorm = normalSampler(rand);
  const z1 = new Float64Array(n), z2 = new Float64Array(n), u = new Float64Array(n);
  for (let i = 0; i < n; i++) { z1[i] = rnorm(); z2[i] = rnorm(); u[i] = rand(); }
  return { z1, z2, u };
}

const N_VALUES = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000];
const N_MAX = 1000;
const N_TEST = 50000;
const N_AVG = 100;
const LR_LAMBDA = 1e-4;
const DEG = Math.PI / 180;
const AVG_DELAY = 250;   // ms after the last change before the 100-sample averages run
const TEST = draws(N_TEST, 99991);

const METHODS = [
  { key: "bayes", name: "Bayes (truth)", color: "#111", width: 3.2, dash: null },
  { key: "lda", name: "LDA", color: "#1a5fb4", width: 2.6, dash: "10 5" },
  { key: "qda", name: "QDA", color: "#a51d2d", width: 3, dash: "1 5" },
  { key: "nb", name: "Naive Bayes", color: "#26a269", width: 2.6, dash: "9 3 2 3" },
  { key: "lr", name: "Logistic", color: "#613583", width: 2.6, dash: "4 3" },
];
const FITTED = METHODS.filter(m => m.key !== "bayes");
const CLASS_COLOR = ["#3d3846", "#c64600"];

// ---------- state ----------
const PRESETS = {
  // Shared covariance (centred in the plot): unit variances, correlation
  // 0.7, so the long axis is at 45° with s.d. √1.7 and the short axis has s.d. √0.3.
  lda: { shared: true, cls: [
    { mx: -0.5, my: -0.75, th: 45, a: Math.sqrt(1.7), w: 0.42 },
    { mx: 0.5, my: 0.75, th: 45, a: Math.sqrt(1.7), w: 0.42 }] },
  // Different covariances: as above, but class 2 has correlation −0.7.
  qda: { shared: false, cls: [
    { mx: -0.5, my: -0.75, th: 45, a: Math.sqrt(1.7), w: 0.42 },
    { mx: 0.5, my: 0.75, th: -45, a: Math.sqrt(1.7), w: 0.42 }] },
  // Axis-aligned ellipses, so the features really are independent within each class.
  nb: { shared: false, cls: [
    { mx: -0.9, my: 0, th: 0, a: 1.5, w: 0.33 },
    { mx: 0.9, my: 0, th: 90, a: 1.2, w: 0.5 }] },
};
let S;            // { preset, shared, pi1, nIdx, cls: [{mx, my, th, a, w}, ...] }
let seed = 1;
let train = null; // draws for the current seed
let averages = null;

function setPreset(name) {
  const p = PRESETS[name];
  S = {
    preset: name,
    shared: p.shared,
    pi1: 0.5,
    nIdx: S ? S.nIdx : 5,
    cls: p.cls.map(c => ({ ...c })),
  };
}

// Covariance of class k as { A (2x2 square root), S11, S12, S22 }.
function classShape(k) {
  const c = S.cls[S.shared ? 0 : k];
  const th = c.th * DEG, a = c.a, b = c.a * c.w;
  const ct = Math.cos(th), st = Math.sin(th);
  return {
    A: [ct * a, -st * b, st * a, ct * b],     // row-major [A11, A12, A21, A22]
    S11: a * a * ct * ct + b * b * st * st,
    S12: (a * a - b * b) * ct * st,
    S22: a * a * st * st + b * b * ct * ct,
  };
}
function mean(k) { return [S.cls[k].mx, S.cls[k].my]; }

// Points of a sample: arrays x1, x2, y (0 = class 1, 1 = class 2).
function makePoints(d, n) {
  const shapes = [classShape(0), classShape(1)];
  const x1 = new Float64Array(n), x2 = new Float64Array(n), y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const k = d.u[i] < S.pi1 ? 0 : 1;
    const A = shapes[k].A, c = S.cls[k];
    y[i] = k;
    x1[i] = c.mx + A[0] * d.z1[i] + A[1] * d.z2[i];
    x2[i] = c.my + A[2] * d.z1[i] + A[3] * d.z2[i];
  }
  return { x1, x2, y, n };
}

// ---------- 2x2 helpers ----------
function inv2(s11, s12, s22) {
  const det = s11 * s22 - s12 * s12;
  return { i11: s22 / det, i12: -s12 / det, i22: s11 / det, det };
}
// Quadratic discriminant term: −½ log|Σ| − ½ (x − μ)ᵀ Σ⁻¹ (x − μ) + log π
function gaussDisc(mu, s11, s12, s22, logPrior) {
  const iv = inv2(s11, s12, s22);
  if (!(iv.det > 1e-12)) return null;
  const c0 = -0.5 * Math.log(iv.det) + logPrior;
  return (x, y) => {
    const dx = x - mu[0], dy = y - mu[1];
    return c0 - 0.5 * (iv.i11 * dx * dx + 2 * iv.i12 * dx * dy + iv.i22 * dy * dy);
  };
}

// ---------- the classifiers ----------
function bayesRule() {
  const d = [0, 1].map(k => {
    const s = classShape(k);
    return gaussDisc(mean(k), s.S11, s.S12, s.S22, Math.log(k === 0 ? S.pi1 : 1 - S.pi1));
  });
  return (x, y) => d[1](x, y) - d[0](x, y);
}

function classStats(P) {
  const st = [0, 1].map(() => ({ n: 0, m1: 0, m2: 0, s11: 0, s12: 0, s22: 0 }));
  for (let i = 0; i < P.n; i++) { const c = st[P.y[i]]; c.n++; c.m1 += P.x1[i]; c.m2 += P.x2[i]; }
  for (const c of st) if (c.n) { c.m1 /= c.n; c.m2 /= c.n; }
  for (let i = 0; i < P.n; i++) {
    const c = st[P.y[i]], d1 = P.x1[i] - c.m1, d2 = P.x2[i] - c.m2;
    c.s11 += d1 * d1; c.s12 += d1 * d2; c.s22 += d2 * d2;   // sums of squares, divided later
  }
  return st;
}

function fitLDA(P, st) {
  if (st[0].n < 1 || st[1].n < 1 || P.n <= 2) return null;
  const dv = P.n - 2;
  const iv = inv2((st[0].s11 + st[1].s11) / dv, (st[0].s12 + st[1].s12) / dv, (st[0].s22 + st[1].s22) / dv);
  if (!(iv.det > 1e-12)) return null;
  const dm1 = st[1].m1 - st[0].m1, dm2 = st[1].m2 - st[0].m2;
  const w1 = iv.i11 * dm1 + iv.i12 * dm2, w2 = iv.i12 * dm1 + iv.i22 * dm2;   // Σ⁻¹(μ2 − μ1)
  const c = Math.log(st[1].n / st[0].n) - w1 * (st[0].m1 + st[1].m1) / 2 - w2 * (st[0].m2 + st[1].m2) / 2;
  return (x, y) => c + w1 * x + w2 * y;
}

function fitQDA(P, st) {
  if (st[0].n < 3 || st[1].n < 3) return null;
  const d = st.map(c => gaussDisc([c.m1, c.m2], c.s11 / (c.n - 1), c.s12 / (c.n - 1), c.s22 / (c.n - 1),
    Math.log(c.n / P.n)));
  if (!d[0] || !d[1]) return null;
  return (x, y) => d[1](x, y) - d[0](x, y);
}

function fitNB(P, st) {
  if (st[0].n < 2 || st[1].n < 2) return null;
  // sklearn's var_smoothing: add 1e-9 × the largest feature variance.
  let maxVar = 0;
  for (const j of [0, 1]) {
    let m = 0, v = 0;
    const xs = j === 0 ? P.x1 : P.x2;
    for (let i = 0; i < P.n; i++) m += xs[i];
    m /= P.n;
    for (let i = 0; i < P.n; i++) v += (xs[i] - m) ** 2;
    maxVar = Math.max(maxVar, v / P.n);
  }
  const eps = 1e-9 * maxVar;
  const d = st.map(c => {
    const v1 = c.s11 / c.n + eps, v2 = c.s22 / c.n + eps;
    const c0 = Math.log(c.n / P.n) - 0.5 * Math.log(v1) - 0.5 * Math.log(v2);
    return (x, y) => c0 - (x - c.m1) ** 2 / (2 * v1) - (y - c.m2) ** 2 / (2 * v2);
  });
  return (x, y) => d[1](x, y) - d[0](x, y);
}

function fitLR(P, st) {
  if (st[0].n < 1 || st[1].n < 1) return null;
  // Newton–Raphson on ℓ(β) − (λ/2)(β1² + β2²), with step halving so the objective never falls.
  let b = [0, 0, 0];
  const obj = bb => {
    let ll = 0;
    for (let i = 0; i < P.n; i++) {
      const eta = bb[0] + bb[1] * P.x1[i] + bb[2] * P.x2[i];
      // log Pr(Y = y_i) = y η − log(1 + e^η), computed stably
      ll += (P.y[i] ? eta : 0) - (eta > 0 ? eta + Math.log1p(Math.exp(-eta)) : Math.log1p(Math.exp(eta)));
    }
    return ll - LR_LAMBDA / 2 * (bb[1] * bb[1] + bb[2] * bb[2]);
  };
  let f = obj(b);
  for (let it = 0; it < 100; it++) {
    const g = [0, -LR_LAMBDA * b[1], -LR_LAMBDA * b[2]];
    const H = [0, 0, 0, 0, LR_LAMBDA, 0, 0, 0, LR_LAMBDA];   // negative Hessian, row-major 3x3
    for (let i = 0; i < P.n; i++) {
      const xi = [1, P.x1[i], P.x2[i]];
      const p = 1 / (1 + Math.exp(-(b[0] + b[1] * xi[1] + b[2] * xi[2])));
      const r = P.y[i] - p, w = p * (1 - p);
      for (let j = 0; j < 3; j++) {
        g[j] += r * xi[j];
        for (let l = 0; l < 3; l++) H[3 * j + l] += w * xi[j] * xi[l];
      }
    }
    const step = solve3(H, g);
    if (!step) break;
    let t = 1, nb, fn;
    for (let h = 0; h < 30; h++) {
      nb = [b[0] + t * step[0], b[1] + t * step[1], b[2] + t * step[2]];
      fn = obj(nb);
      if (fn >= f - 1e-12) break;
      t /= 2;
    }
    const moved = Math.hypot(nb[0] - b[0], nb[1] - b[1], nb[2] - b[2]);
    b = nb; f = fn;
    if (moved < 1e-9 * (1 + Math.hypot(...b))) break;
  }
  return (x, y) => b[0] + b[1] * x + b[2] * y;
}
function solve3(M, v) {
  const [a, b, c, d, e, f, g, h, i] = M;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!(Math.abs(det) > 1e-300)) return null;
  const inv = [e * i - f * h, c * h - b * i, b * f - c * e,
               f * g - d * i, a * i - c * g, c * d - a * f,
               d * h - e * g, b * g - a * h, a * e - b * d];
  return [0, 1, 2].map(r => (inv[3 * r] * v[0] + inv[3 * r + 1] * v[1] + inv[3 * r + 2] * v[2]) / det);
}

function fitAll(P) {
  const st = classStats(P);
  return { bayes: bayesRule(), lda: fitLDA(P, st), qda: fitQDA(P, st), nb: fitNB(P, st), lr: fitLR(P, st), st };
}
function testError(g, T) {
  if (!g) return NaN;
  let wrong = 0;
  for (let i = 0; i < T.n; i++) if ((g(T.x1[i], T.x2[i]) > 0 ? 1 : 0) !== T.y[i]) wrong++;
  return wrong / T.n;
}

// ---------- plot ----------
const W = 520, H = 520;
const M = { top: 10, right: 10, bottom: 40, left: 44 };
const LIM = 4.5;
const xS = d3.scaleLinear().domain([-LIM, LIM]).range([M.left, W - M.right]);
const yS = d3.scaleLinear().domain([-LIM, LIM]).range([H - M.bottom, M.top]);
// Grid for contours, wider than the plot so the contour polygons' edges fall outside the clip.
const G = 110, GLIM = 5.5, CELL = 2 * GLIM / G;
const gridX = d3.range(G).map(i => -GLIM + (i + 0.5) * CELL);
const gridY = d3.range(G).map(j => GLIM - (j + 0.5) * CELL);
const contours = d3.contours().size([G, G]).thresholds([0]);
const cellPx = xS(-GLIM + CELL) - xS(-GLIM);
const contourPath = d3.geoPath(d3.geoIdentity().scale(cellPx).translate([xS(-GLIM), yS(GLIM)]));

const svg = d3.select("#plot").append("svg")
  .attr("viewBox", `0 0 ${W} ${H}`)
  .attr("role", "group")
  .attr("aria-labelledby", "plot-title")
  .attr("aria-describedby", "plot-desc");
svg.append("title").attr("id", "plot-title").text("Training sample, class distributions and decision boundaries");
const descEl = d3.select("#plot").append("p").attr("id", "plot-desc").attr("class", "sr-only");

svg.append("defs").append("clipPath").attr("id", "clip")
  .append("rect").attr("x", M.left).attr("y", M.top)
  .attr("width", W - M.left - M.right).attr("height", H - M.top - M.bottom);

const deco = svg.append("g").attr("aria-hidden", "true");
deco.append("g").attr("class", "axis").attr("transform", `translate(0,${H - M.bottom})`)
  .call(d3.axisBottom(xS).ticks(9));
deco.append("g").attr("class", "axis").attr("transform", `translate(${M.left},0)`)
  .call(d3.axisLeft(yS).ticks(9));
deco.append("text").attr("class", "axis-label").attr("x", (M.left + W - M.right) / 2).attr("y", H - 6)
  .attr("text-anchor", "middle").text("x₁");
deco.append("text").attr("class", "axis-label")
  .attr("transform", `translate(12,${(M.top + H - M.bottom) / 2}) rotate(-90)`)
  .attr("text-anchor", "middle").text("x₂");

const area = deco.append("g").attr("clip-path", "url(#clip)");
const regionPath = area.append("path").attr("fill", "#fbe3d3").attr("fill-opacity", 0.7);
const ellipseLayer = area.append("g");
const pointLayer = area.append("g");
const boundaryLayer = area.append("g");
const boundaryPaths = {};
for (const m of METHODS) {
  boundaryPaths[m.key] = boundaryLayer.append("path").attr("fill", "none")
    .attr("stroke", m.color).attr("stroke-width", m.width)
    .attr("stroke-dasharray", m.dash).attr("stroke-linecap", m.key === "qda" ? "round" : "butt");
}

// Handles (focusable; outside the aria-hidden group).
const handleLayer = svg.append("g");
const triangle = d3.symbol().type(d3.symbolTriangle);
function makeMeanHandle(k) {
  const g = handleLayer.append("g").attr("class", "handle").attr("tabindex", 0)
    .attr("role", "button").attr("aria-roledescription", "draggable point");
  g.append("circle").attr("class", "ring").attr("r", 16);
  if (k === 0) g.append("circle").attr("r", 9);
  else g.append("path").attr("d", triangle.size(300)());
  g.selectAll("circle:not(.ring), path").attr("fill", "#fff").attr("stroke", CLASS_COLOR[k]).attr("stroke-width", 3);
  g.append("text").attr("text-anchor", "middle").attr("dy", k === 0 ? "0.35em" : "0.6em")
    .attr("font-size", 10).attr("font-weight", 700).attr("fill", CLASS_COLOR[k]).text(k + 1);
  return g;
}
const meanHandles = [0, 1].map(makeMeanHandle);

// ---------- formatting ----------
const fmt3 = d3.format(".3f");
const fmt2 = v => d3.format(".2f")(v).replace("-", "−");
const fmtErr = v => (Number.isNaN(v) ? "—" : fmt3(v));

// ---------- error table ----------
const rows = d3.select("#error-rows").selectAll("tr").data(METHODS).join("tr");
rows.append("th").attr("scope", "row").text(m => m.name);
rows.append("td").attr("class", "key").append("svg").attr("width", 44).attr("height", 12)
  .attr("aria-hidden", "true")
  .append("line").attr("x1", 2).attr("x2", 42).attr("y1", 6).attr("y2", 6)
  .attr("stroke", m => m.color).attr("stroke-width", m => m.width)
  .attr("stroke-dasharray", m => m.dash).attr("stroke-linecap", m => (m.key === "qda" ? "round" : "butt"));
rows.select("td.key").append("span").attr("class", "sr-only")
  .text(m => ({ bayes: "solid black", lda: "long dashes", qda: "dots", nb: "dash-dot", lr: "short dashes" })[m.key]);
const cellNow = rows.append("td");
const cellAvg = rows.append("td");

// ---------- update ----------
const nSlider = document.getElementById("n-slider");
const presetButtons = document.querySelectorAll("button[data-preset]");
const statsEl = document.getElementById("stats");
const show = { bayes: true };   // the Bayes boundary and region are always drawn
document.querySelectorAll("input[data-method]").forEach(cb => {
  show[cb.dataset.method] = cb.checked;
  cb.addEventListener("change", () => { show[cb.dataset.method] = cb.checked; schedule(); });
});
let avgTimer = null;
let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; update(); });
}
// Any change clears the averages; they are recomputed once things have been still for AVG_DELAY.
function changed() {
  averages = null;
  schedule();
  clearTimeout(avgTimer);
  avgTimer = setTimeout(runAverages, AVG_DELAY);
}

function syncControls() {
  nSlider.value = S.nIdx;
  presetButtons.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.preset === S.preset)));
}

function ellipsePoints(k, c) {
  const s = classShape(k), mu = mean(k);
  return d3.range(73).map(t => {
    const a = t * 5 * DEG, u = c * Math.cos(a), v = c * Math.sin(a);
    return [mu[0] + s.A[0] * u + s.A[1] * v, mu[1] + s.A[2] * u + s.A[3] * v];
  });
}
const lineXY = d3.line().x(d => xS(d[0])).y(d => yS(d[1]));

function update() {
  const n = N_VALUES[S.nIdx];
  const P = makePoints(train, n);
  const T = makePoints(TEST, N_TEST);
  const fits = fitAll(P);
  const err = {};
  for (const m of METHODS) err[m.key] = testError(fits[m.key], T);

  // Region and boundaries
  const values = new Float64Array(G * G);
  for (const m of METHODS) {
    const g = fits[m.key];
    if (!g || (!show[m.key] && m.key !== "bayes")) { boundaryPaths[m.key].attr("d", null); continue; }
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) values[i + j * G] = g(gridX[i], gridY[j]);
    const shape = contours(values)[0];
    const d = contourPath(shape);
    if (m.key === "bayes") regionPath.attr("d", d);
    boundaryPaths[m.key].attr("d", show[m.key] ? d : null);
  }

  // Ellipses (1 and 2 s.d.)
  ellipseLayer.selectAll("path").data([[0, 1], [0, 2], [1, 1], [1, 2]]).join("path")
    .attr("fill", "none").attr("stroke", ([k]) => CLASS_COLOR[k]).attr("stroke-opacity", 0.45)
    .attr("stroke-width", ([, c]) => (c === 1 ? 1.4 : 1)).attr("stroke-dasharray", ([k]) => (k === 0 ? null : "5 3"))
    .attr("d", ([k, c]) => lineXY(ellipsePoints(k, c)));

  // Points: class 1 open circles, class 2 filled triangles
  const idx = d3.range(n);
  const r = n > 300 ? 2 : 3;
  pointLayer.selectAll("circle").data(idx.filter(i => P.y[i] === 0)).join("circle")
    .attr("cx", i => xS(P.x1[i])).attr("cy", i => yS(P.x2[i])).attr("r", r)
    .attr("fill", "none").attr("stroke", CLASS_COLOR[0]).attr("stroke-width", 1.1).attr("stroke-opacity", 0.8);
  const tri = triangle.size(n > 300 ? 18 : 34)();
  pointLayer.selectAll("path").data(idx.filter(i => P.y[i] === 1)).join("path")
    .attr("d", tri).attr("transform", i => `translate(${xS(P.x1[i])},${yS(P.x2[i])})`)
    .attr("fill", CLASS_COLOR[1]).attr("fill-opacity", 0.75);

  // Handles
  for (const k of [0, 1]) {
    const mu = mean(k);
    meanHandles[k].attr("transform", `translate(${xS(mu[0])},${yS(mu[1])})`)
      .attr("aria-label", `Class ${k + 1} mean at x₁ = ${fmt2(mu[0])}, x₂ = ${fmt2(mu[1])}. Arrow keys move it.`);
  }

  // Table: bold marks the lowest average (at displayed precision, so ties are all bold).
  const best = averages ? d3.min(FITTED.map(m => averages[m.key]).filter(v => !Number.isNaN(v)).map(fmt3)) : null;
  rows.classed("best", m => m.key !== "bayes" && best !== null && fmtErr(averages[m.key]) === best);
  cellNow.text(m => fmtErr(err[m.key]));
  cellAvg.text(m => (averages ? fmtErr(averages[m.key]) : "…"));

  document.getElementById("n-out").textContent = n;
  nSlider.setAttribute("aria-valuetext", `n = ${n}`);

  // Description and live summary (announced once the averages are in)
  const errText = METHODS.map(m => `${m.name} ${fmtErr(err[m.key])}`).join(", ");
  const scenario = { lda: "shared covariance", qda: "different covariances", nb: "independent features" }[S.preset];
  descEl.text(
    `Scatter plot, x₁ and x₂ from −4.5 to 4.5, scenario: ${scenario}. ${n} training points: ` +
    `${fits.st[0].n} in class 1 (open circles), ${fits.st[1].n} in class 2 (filled triangles). ` +
    `Class 1 is centred at (${fmt2(S.cls[0].mx)}, ${fmt2(S.cls[0].my)}), class 2 at ` +
    `(${fmt2(S.cls[1].mx)}, ${fmt2(S.cls[1].my)}). The shaded region is where the Bayes classifier picks class 2. ` +
    `Test errors for this sample: ${errText}.`);
  if (averages) {
    const bestNames = FITTED.filter(m => fmtErr(averages[m.key]) === best).map(m => m.name).join(" and ");
    statsEl.textContent = `n = ${n}, ${scenario}. Average test error over ${N_AVG} samples: ` +
      METHODS.map(m => `${m.name} ${fmtErr(averages[m.key])}`).join(", ") + `. Lowest: ${bestNames || "none"}.`;
  }
}

function runAverages() {
  const n = N_VALUES[S.nIdx];
  const T = makePoints(TEST, N_TEST);
  const sums = Object.fromEntries(FITTED.map(m => [m.key, 0]));
  const counts = Object.fromEntries(FITTED.map(m => [m.key, 0]));
  for (let s = 1; s <= N_AVG; s++) {
    const fits = fitAll(makePoints(draws(n, 100000 + s), n));
    for (const m of FITTED) {
      const e = testError(fits[m.key], T);
      if (!Number.isNaN(e)) { sums[m.key] += e; counts[m.key]++; }
    }
  }
  averages = { bayes: testError(bayesRule(), T) };
  // A method that can't be fitted on some samples is reported as "—" rather than averaged
  // over the samples where it happened to work.
  for (const m of FITTED) averages[m.key] = counts[m.key] === N_AVG ? sums[m.key] / N_AVG : NaN;
  update();
}

// ---------- interaction ----------
const clampMean = v => Math.max(-4, Math.min(4, v));
function invX(px) { return xS.invert(px); }
function invY(py) { return yS.invert(py); }

meanHandles.forEach((g, k) => {
  g.call(d3.drag().on("drag", event => {
    const [px, py] = d3.pointer(event, svg.node());
    S.cls[k].mx = clampMean(invX(px));
    S.cls[k].my = clampMean(invY(py));
    changed();
  }));
  g.on("keydown", event => {
    const step = event.shiftKey ? 0.5 : 0.1;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[event.key];
    if (!d) return;
    event.preventDefault();
    S.cls[k].mx = clampMean(+(S.cls[k].mx + d[0]).toFixed(2));
    S.cls[k].my = clampMean(+(S.cls[k].my + d[1]).toFixed(2));
    changed();
  });
});
nSlider.addEventListener("input", () => { S.nIdx = +nSlider.value; changed(); });
// Shift+arrow moves a slider by 5 steps.
document.querySelectorAll("input[type=range]").forEach(sl => sl.addEventListener("keydown", event => {
  if (!event.shiftKey) return;
  const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[event.key];
  if (!dir) return;
  event.preventDefault();
  const step = +sl.step, lo = +sl.min, hi = +sl.max;
  sl.value = Math.min(hi, Math.max(lo, +sl.value + 5 * dir * step));
  sl.dispatchEvent(new Event("input"));
}));

presetButtons.forEach(b => b.addEventListener("click", () => {
  setPreset(b.dataset.preset);
  syncControls();
  changed();
}));
document.getElementById("resample").addEventListener("click", () => {
  seed += 1;
  train = draws(N_MAX, seed);
  schedule();   // the averages don't depend on this sample, so keep them
});

train = draws(N_MAX, seed);
setPreset("lda");
syncControls();
changed();
