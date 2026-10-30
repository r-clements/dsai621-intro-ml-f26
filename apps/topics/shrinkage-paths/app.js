// Shrinkage: ridge, lasso and elastic net (ISL 6.2, ESL 3.4).
//
// Paths view: the elastic-net problem in scikit-learn's ElasticNet scaling,
//   minimize (1/2n)||y − Xb||² + λ[ρ||b||₁ + (1−ρ)/2 ||b||²],
// with X standardized (mean 0, sd 1 using 1/n) and y centred. ρ = 1 is the lasso, ρ = 0 is ridge
// (equal to sklearn Ridge(alpha = nλ)). Solved by coordinate descent with warm starts down a λ grid,
// then polished by an exact solve on the active set and a KKT check, so every point is exact.
//
// Geometry view: minimize (β − b̂)ᵀA(β − b̂) subject to P(β) ≤ P(t, 0), with A = [[1, r], [r, 1]] and
// P(β) = ρ|β|₁ + (1−ρ)/2 |β|². Solved through the penalized form: for a multiplier μ the 2-D problem
// is solved exactly by enumerating supports and signs, and μ is found by bisection.
//

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

const fmt2 = d3.format(".2f");
const fmt3 = d3.format(".3f");
const fmtLam = v => (v >= 100 ? d3.format(".0f")(v) : v >= 1 ? d3.format(".2f")(v) : d3.format(".3~g")(v));
const signed = v => (Math.abs(v) < 5e-4 ? "0" : fmt2(v)).replace("-", "−");
const minus = s => String(s).replace(/-/g, "−");
// Live regions update at once on first render, then debounced so dragging doesn't flood a screen reader.
let started = false;
const liveDelay = () => (started ? 300 : 0);

// ---------- small dense linear algebra ----------
// Solve M x = b for symmetric positive definite M (Gaussian elimination with partial pivoting).
function solve(M, b) {
  const k = b.length;
  const A = M.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < k; c++) {
    let piv = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    for (let r = c + 1; r < k; r++) {
      const f = A[r][c] / A[c][c];
      for (let j = c; j <= k; j++) A[r][j] -= f * A[c][j];
    }
  }
  const x = new Array(k).fill(0);
  for (let r = k - 1; r >= 0; r--) {
    let s = A[r][k];
    for (let j = r + 1; j < k; j++) s -= A[r][j] * x[j];
    x[r] = s / A[r][r];
  }
  return x;
}
function inverse(M) {
  const k = M.length;
  return d3.range(k).map(j => solve(M, d3.range(k).map(i => (i === j ? 1 : 0))))
    .reduce((cols, col, j) => { col.forEach((v, i) => { cols[i][j] = v; }); return cols; },
      d3.range(k).map(() => new Array(k).fill(0)));
}
// The mix slider only applies to the elastic net: keep it visible but disabled (greyed) otherwise.
function setMixControl(prefix, method) {
  const on = method === "enet";
  document.getElementById(`${prefix}-slider`).disabled = !on;
  document.getElementById(`${prefix}-row`).classList.toggle("off", !on);
}
const soft = (z, g) => (z > g ? z - g : z < -g ? z + g : 0);

// =====================================================================
// Paths view
// =====================================================================
const N = 60, N_TEST = 2000, P_BASE = 6, SIGMA = 3, CORR = 0.5;
const BETA = [3, -2, 1.5, 0, 0, 0.5];
const N_GRID = 121;                // slider positions; index 0 = largest λ

// Generate train and test sets from one seeded stream. x7 = x1 + 0.1·noise is always drawn
// (so toggling it does not change the other columns) but only used when the toggle is on.
function makeData(seed) {
  const rnorm = normalSampler(mulberry32(seed));
  const draw = m => {
    const X = [], y = [];
    for (let i = 0; i < m; i++) {
      const row = new Array(P_BASE + 1);
      row[0] = rnorm();
      for (let j = 1; j < P_BASE; j++) row[j] = CORR * row[j - 1] + Math.sqrt(1 - CORR * CORR) * rnorm();
      row[P_BASE] = row[0] + 0.1 * rnorm();
      let mu = 0;
      for (let j = 0; j < P_BASE; j++) mu += BETA[j] * row[j];
      X.push(row);
      y.push(mu + SIGMA * rnorm());
    }
    return { X, y };
  };
  return { train: draw(N), test: draw(N_TEST) };
}

const P_SEED0 = 649;   // chosen so the default fit is readable: lasso zeroes x4 and x5 at its best λ
let pSeed = P_SEED0;
let raw = null;           // { train, test } unstandardized
let D = null;             // prepared data for the current column set
let path = null;          // { lams, coefs[g][j], nz, df, train, test }
let ols = null;

function prepare(useDup) {
  const p = useDup ? P_BASE + 1 : P_BASE;
  const cols = d3.range(p);
  const Xr = raw.train.X, yr = raw.train.y;
  const mean = cols.map(j => d3.mean(Xr, r => r[j]));
  const sd = cols.map(j => Math.sqrt(d3.mean(Xr, r => (r[j] - mean[j]) ** 2)));
  const ybar = d3.mean(yr);
  const X = Xr.map(r => cols.map(j => (r[j] - mean[j]) / sd[j]));
  const y = yr.map(v => v - ybar);
  const G = cols.map(j => cols.map(k => d3.sum(X, r => r[j] * r[k]) / N));
  const c = cols.map(j => d3.sum(X, (r, i) => r[j] * y[i]) / N);
  const yy = d3.sum(y, v => v * v) / N;
  const Xt = raw.test.X.map(r => cols.map(j => (r[j] - mean[j]) / sd[j]));
  const yt = raw.test.y;
  // True coefficients on the standardized scale: β_j · sd_j (x7 has true coefficient 0).
  const trueStd = cols.map(j => (j < P_BASE ? BETA[j] * sd[j] : 0));
  const names = cols.map(j => `x${j + 1}`);
  return { p, G, c, yy, Xt, yt, ybar, trueStd, names };
}

const trainMSE = b => {
  const { G, c, yy, p } = D;
  let s = yy;
  for (let j = 0; j < p; j++) {
    s -= 2 * c[j] * b[j];
    for (let k = 0; k < p; k++) s += b[j] * G[j][k] * b[k];
  }
  return s;
};
const testMSE = b => {
  let s = 0;
  for (let i = 0; i < D.Xt.length; i++) {
    let pred = D.ybar;
    const r = D.Xt[i];
    for (let j = 0; j < D.p; j++) pred += r[j] * b[j];
    s += (D.yt[i] - pred) ** 2;
  }
  return s / D.Xt.length;
};

// Exact elastic-net solution at (λ, ρ), warm-started from b0.
function enet(lam, rho, b0) {
  const { G, c, p } = D;
  const l1 = lam * rho, l2 = lam * (1 - rho);
  const b = b0.slice();
  for (let sweep = 0; sweep < 3000; sweep++) {
    let maxd = 0;
    for (let j = 0; j < p; j++) {
      let z = c[j];
      for (let k = 0; k < p; k++) if (k !== j) z -= G[j][k] * b[k];
      const nb = soft(z, l1) / (G[j][j] + l2);
      maxd = Math.max(maxd, Math.abs(nb - b[j]));
      b[j] = nb;
    }
    if (maxd < 1e-9) break;
  }
  // Polish: solve the stationarity equations exactly on the active set, then check KKT.
  for (let attempt = 0; attempt < 5; attempt++) {
    const A = d3.range(p).filter(j => b[j] !== 0);
    if (A.length === 0) break;
    const s = A.map(j => Math.sign(b[j]));
    const M = A.map(j => A.map(k => G[j][k] + (j === k ? l2 : 0)));
    const bA = solve(M, A.map((j, i) => c[j] - l1 * s[i]));
    const cand = new Array(p).fill(0);
    A.forEach((j, i) => { cand[j] = bA[i]; });
    const signsOk = A.every((j, i) => Math.sign(bA[i]) === s[i]);
    const kktOk = d3.range(p).every(j => {
      if (cand[j] !== 0) return true;
      let g = c[j];
      for (let k = 0; k < p; k++) g -= G[j][k] * cand[k];
      return Math.abs(g) <= l1 + 1e-9;
    });
    if (signsOk && kktOk) return cand;
    // Otherwise run more coordinate descent from the candidate and try again.
    for (let sweep = 0; sweep < 20000; sweep++) {
      let maxd = 0;
      for (let j = 0; j < p; j++) {
        let z = c[j];
        for (let k = 0; k < p; k++) if (k !== j) z -= G[j][k] * b[k];
        const nb = soft(z, l1) / (G[j][j] + l2);
        maxd = Math.max(maxd, Math.abs(nb - b[j]));
        b[j] = nb;
      }
      if (maxd < 1e-13) break;
    }
  }
  return b;
}

// Effective degrees of freedom (Zou & Hastie 2005): tr[G_AA (G_AA + λ(1−ρ)I)⁻¹] over the active
// set. Ridge: Σ e/(e + λ); lasso: number of non-zero coefficients.
function dfOf(b, lam, rho) {
  const A = d3.range(D.p).filter(j => b[j] !== 0);
  if (A.length === 0) return 0;
  const GA = A.map(j => A.map(k => D.G[j][k]));
  const Minv = inverse(GA.map((row, i) => row.map((v, k) => v + (i === k ? lam * (1 - rho) : 0))));
  let tr = 0;
  for (let i = 0; i < A.length; i++) for (let k = 0; k < A.length; k++) tr += GA[i][k] * Minv[k][i];
  return tr;
}

let lamMaxL1 = 1;
function computePath(rho) {
  lamMaxL1 = d3.max(D.c, Math.abs);   // sklearn's alpha_max for the lasso
  const hi = Math.log10(lamMaxL1) + 1.5, lo = Math.log10(lamMaxL1) - 3;
  const lams = d3.range(N_GRID).map(g => 10 ** (hi - (hi - lo) * g / (N_GRID - 1)));
  let b = new Array(D.p).fill(0);
  const coefs = [], nz = [], df = [], tr = [], te = [];
  for (const lam of lams) {
    b = enet(lam, rho, b);
    coefs.push(b.slice());
    nz.push(b.filter(v => v !== 0).length);
    df.push(dfOf(b, lam, rho));
    tr.push(trainMSE(b));
    te.push(testMSE(b));
  }
  return { lams, coefs, nz, df, train: tr, test: te, rho };
}

// ---------- paths chart ----------
const PW = 720, PH = 330;
const mp = { top: 14, right: 46, bottom: 40, left: 46 };
const pathsSvg = d3.select("#paths").append("svg").attr("viewBox", `0 0 ${PW} ${PH}`).attr("role", "img");
const pathsTitle = pathsSvg.append("title").attr("id", "paths-title");
const pathsDesc = pathsSvg.append("desc").attr("id", "paths-desc");
pathsSvg.attr("aria-labelledby", "paths-title paths-desc");
const gx = d3.scaleLinear().domain([0, N_GRID - 1]).range([mp.left, PW - mp.right]);
const py = d3.scaleLinear().range([PH - mp.bottom, mp.top]);
const pxAxis = pathsSvg.append("g").attr("class", "axis").attr("transform", `translate(0,${PH - mp.bottom})`);
const pyAxis = pathsSvg.append("g").attr("class", "axis").attr("transform", `translate(${mp.left},0)`);
pathsSvg.append("text").attr("class", "axis-label").attr("x", (mp.left + PW - mp.right) / 2).attr("y", PH - 6)
  .attr("text-anchor", "middle").text("λ (log scale)");
pathsSvg.append("text").attr("class", "axis-label")
  .attr("transform", `translate(12,${(mp.top + PH - mp.bottom) / 2}) rotate(-90)`)
  .attr("text-anchor", "middle").text("coefficient");
const zeroLine = pathsSvg.append("line").attr("stroke", "#bbb");
const pathLayer = pathsSvg.append("g");
const labelLayer = pathsSvg.append("g");
const lamLine = pathsSvg.append("line").attr("stroke", "#333").attr("stroke-width", 1)
  .attr("y1", mp.top).attr("y2", PH - mp.bottom);
const dotLayer = pathsSvg.append("g");

const COLORS = ["#1a5fb4", "#c01c28", "#26a269", "#c64600", "#813d9c", "#5e5c64", "#986a44"];
const DASHES = ["", "7 4", "2 3", "10 3 2 3", "4 2", "12 4", "1 4"];

// ---------- error chart ----------
const EW = 720, EH = 210;
const me = { top: 12, right: 46, bottom: 34, left: 46 };
const errSvg = d3.select("#errors").append("svg").attr("viewBox", `0 0 ${EW} ${EH}`).attr("role", "img");
const errTitle = errSvg.append("title").attr("id", "err-title");
const errDesc = errSvg.append("desc").attr("id", "err-desc");
errSvg.attr("aria-labelledby", "err-title err-desc");
const ey = d3.scaleLinear().range([EH - me.bottom, me.top]);
const exAxis = errSvg.append("g").attr("class", "axis").attr("transform", `translate(0,${EH - me.bottom})`);
const eyAxis = errSvg.append("g").attr("class", "axis").attr("transform", `translate(${me.left},0)`);
errSvg.append("text").attr("class", "axis-label")
  .attr("transform", `translate(12,${(me.top + EH - me.bottom) / 2}) rotate(-90)`)
  .attr("text-anchor", "middle").text("MSE");
const sigLine = errSvg.append("line").attr("stroke", "#777").attr("stroke-width", 1.5).attr("stroke-dasharray", "2 3");
const testPath = errSvg.append("path").attr("fill", "none").attr("stroke", "#c01c28").attr("stroke-width", 2.5);
const trainPath = errSvg.append("path").attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 2.5)
  .attr("stroke-dasharray", "6 4");
const testLbl = errSvg.append("text").attr("class", "line-label").attr("text-anchor", "end").text("test");
const trainLbl = errSvg.append("text").attr("class", "line-label").attr("text-anchor", "end").text("training");
const sigLbl = errSvg.append("text").attr("class", "line-label").attr("fill", "#555").text("σ²");
const errLam = errSvg.append("line").attr("stroke", "#333").attr("y1", me.top).attr("y2", EH - me.bottom);
const errTestDot = errSvg.append("circle").attr("r", 5).attr("fill", "#c01c28").attr("stroke", "#fff").attr("stroke-width", 1.5);
const errTrainDot = errSvg.append("rect").attr("width", 9).attr("height", 9).attr("fill", "#1a5fb4")
  .attr("stroke", "#fff").attr("stroke-width", 1.5);

function lamTicks() {
  const lams = path.lams;
  const hi = Math.log10(lams[0]), lo = Math.log10(lams[N_GRID - 1]);
  const ticks = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) ticks.push(e);
  const toG = e => (hi - e) / (hi - lo) * (N_GRID - 1);
  return { values: ticks.map(toG), format: g => { const e = Math.round(hi - g / (N_GRID - 1) * (hi - lo)); return minus(e === 0 ? "1" : e === 1 ? "10" : `10^${e}`); } };
}
const superscript = s => s.replace(/10\^(−?\d+)/, (m, e) => "10" + [...e].map(ch => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["−0123456789".indexOf(ch)]).join(""));

const lamSlider = document.getElementById("lam-slider");
const rhoSlider = document.getElementById("rho-slider");
const dupToggle = document.getElementById("dup-toggle");
const methodOf = () => document.querySelector("input[name=method]:checked").value;
const rhoOf = () => ({ ridge: 0, lasso: 1, enet: +rhoSlider.value }[methodOf()]);

function rebuildPaths() {
  D = prepare(dupToggle.checked);
  ols = solve(D.G, D.c);
  path = computePath(rhoOf());
  const all = path.coefs.flat().concat(ols, D.trueStd);
  const ext = d3.extent(all);
  const pad = 0.08 * (ext[1] - ext[0]);
  py.domain([ext[0] - pad, ext[1] + pad]).nice();
  const t = lamTicks();
  pxAxis.call(d3.axisBottom(gx).tickValues(t.values).tickFormat(g => superscript(t.format(g))));
  pyAxis.call(d3.axisLeft(py).ticks(6).tickFormat(v => minus(d3.format("~g")(v))));
  zeroLine.attr("x1", mp.left).attr("x2", PW - mp.right).attr("y1", py(0)).attr("y2", py(0));

  const line = d3.line().x((d, g) => gx(g)).y(d => py(d));
  pathLayer.selectAll("path").data(d3.range(D.p)).join("path")
    .attr("fill", "none").attr("stroke", j => COLORS[j]).attr("stroke-width", 2)
    .attr("stroke-dasharray", j => DASHES[j])
    .attr("d", j => line(path.coefs.map(b => b[j])));

  // Direct labels at the right end, nudged apart so they don't overlap.
  const ends = d3.range(D.p).map(j => ({ j, y: py(path.coefs[N_GRID - 1][j]) })).sort((a, b) => a.y - b.y);
  for (let it = 0; it < 50; it++) {
    for (let i = 1; i < ends.length; i++) {
      const gap = ends[i].y - ends[i - 1].y;
      if (gap < 11) { const d = (11 - gap) / 2; ends[i].y += d; ends[i - 1].y -= d; }
    }
  }
  labelLayer.selectAll("text").data(ends).join("text").attr("class", "line-label")
    .attr("x", PW - mp.right + 4).attr("y", d => d.y).attr("dy", "0.35em")
    .attr("fill", d => COLORS[d.j]).style("font-weight", 600).text(d => D.names[d.j]);

  const ext2 = [0, Math.max(d3.max(path.test), d3.max(path.train), SIGMA * SIGMA) * 1.05];
  ey.domain(ext2).nice();
  exAxis.call(d3.axisBottom(gx).tickValues(t.values).tickFormat(g => superscript(t.format(g))));
  eyAxis.call(d3.axisLeft(ey).ticks(5));
  sigLine.attr("x1", mp.left).attr("x2", EW - me.right).attr("y1", ey(SIGMA * SIGMA)).attr("y2", ey(SIGMA * SIGMA));
  const eline = arr => d3.line().x((d, g) => gx(g)).y(d => ey(d))(arr);
  testPath.attr("d", eline(path.test));
  trainPath.attr("d", eline(path.train));
  testLbl.attr("x", EW - me.right - 2).attr("y", ey(path.test[N_GRID - 1]) - 6);
  trainLbl.attr("x", EW - me.right - 2).attr("y", ey(path.train[N_GRID - 1]) + 14);
  sigLbl.attr("x", me.left + 4).attr("y", ey(SIGMA * SIGMA) + 14);
  updatePaths();
}

const bestIndex = () => d3.minIndex(path.test);
let pathsLive = null;
function updatePaths() {
  const g = +lamSlider.value;
  const lam = path.lams[g], b = path.coefs[g], rho = path.rho, method = methodOf();
  const x = gx(g);
  lamLine.attr("x1", x).attr("x2", x);
  errLam.attr("x1", x).attr("x2", x);
  errTestDot.attr("cx", x).attr("cy", ey(path.test[g]));
  errTrainDot.attr("x", x - 4.5).attr("y", ey(path.train[g]) - 4.5);
  dotLayer.selectAll("circle").data(d3.range(D.p)).join("circle").attr("r", 3.5)
    .attr("fill", j => (b[j] === 0 ? "#fff" : COLORS[j])).attr("stroke", j => COLORS[j]).attr("stroke-width", 1.5)
    .attr("cx", x).attr("cy", j => py(b[j]));

  const nz = path.nz[g];
  const sk = method === "ridge" ? `sklearn: Ridge(alpha=${fmtLam(N * lam)})`
    : method === "lasso" ? `sklearn: Lasso(alpha=${fmtLam(lam)})`
      : `sklearn: ElasticNet(alpha=${fmtLam(lam)}, l1_ratio=${rho})`;
  document.getElementById("lam-out").textContent = `λ = ${fmtLam(lam)}`;
  document.getElementById("rho-out").textContent = `ρ = ${rho}`;
  document.getElementById("paths-extra").textContent =
    `λ = ${fmtLam(lam)}, effective df = ${fmt2(path.df[g])}. Penalty (1/2n)·RSS + λ[ρ|β|₁ + (1−ρ)/2·|β|₂²]; ${sk}.`;
  lamSlider.setAttribute("aria-valuetext", `λ = ${fmtLam(lam)}, ${nz} of ${D.p} coefficients non-zero`);

  const best = bestIndex();
  const zeros = d3.range(D.p).filter(j => b[j] === 0).map(j => D.names[j]);
  const methodName = { ridge: "Ridge", lasso: "Lasso", enet: `Elastic net (ρ = ${rho})` }[method];
  pathsTitle.text(`${methodName} coefficient paths, current λ = ${fmtLam(lam)}`);
  pathsDesc.text(
    `Line chart of the ${D.p} standardized coefficients against λ on a log scale, from λ = ${fmtLam(path.lams[0])} ` +
    `at the left, where ${path.nz[0] === 0 ? "every coefficient is exactly zero" : "every coefficient is shrunk close to zero"}, ` +
    `to λ = ${fmtLam(path.lams[N_GRID - 1])} at the right, where they are close to OLS. ` +
    `${method === "ridge" ? "Ridge never sets a coefficient exactly to zero; the curves approach zero smoothly." :
      "Coefficients leave zero one at a time as λ decreases."} ` +
    `At the current λ: ${D.names.map((n, j) => `${n} ${signed(b[j])}`).join(", ")}. ` +
    (zeros.length ? `Exactly zero: ${zeros.join(", ")}. ` : "No coefficient is exactly zero. ") +
    `OLS: ${D.names.map((n, j) => `${n} ${signed(ols[j])}`).join(", ")}.`);
  errTitle.text(`Training and test error against λ, current λ = ${fmtLam(lam)}`);
  errDesc.text(
    `Line chart. Training MSE rises steadily as λ grows (leftward), from ${fmt2(path.train[N_GRID - 1])} near OLS to ` +
    `${fmt2(path.train[0])} at the largest λ. Test MSE on 2000 new points is lowest, ${fmt2(path.test[best])}, at ` +
    `λ = ${fmtLam(path.lams[best])}, compared with ${fmt2(path.test[N_GRID - 1])} near OLS and ${fmt2(path.test[0])} ` +
    `at the largest λ. Noise variance σ² = 9 is the floor. Current λ: training ${fmt2(path.train[g])}, test ${fmt2(path.test[g])}.`);

  d3.select("#coef-body").selectAll("tr").data(d3.range(D.p)).join("tr")
    .html(j => `<th scope="row">${D.names[j]}</th><td>${signed(D.trueStd[j])}</td><td>${signed(ols[j])}</td>` +
      `<td class="${b[j] === 0 ? "zero" : ""}">${b[j] === 0 ? "0 (exactly)" : signed(b[j])}</td>`);

  clearTimeout(pathsLive);
  pathsLive = setTimeout(() => {
    document.getElementById("s-nz").textContent = `${nz} of ${D.p}`;
    document.getElementById("s-train").textContent = fmt2(path.train[g]);
    document.getElementById("s-test").textContent = fmt2(path.test[g]);
  }, liveDelay());
}

function setMethodControls() {
  setMixControl("rho", methodOf());
}
document.querySelectorAll("input[name=method]").forEach(el =>
  el.addEventListener("change", () => { setMethodControls(); rebuildPaths(); }));
rhoSlider.addEventListener("input", rebuildPaths);
lamSlider.addEventListener("input", updatePaths);
dupToggle.addEventListener("change", rebuildPaths);
document.getElementById("new-data").addEventListener("click", () => {
  pSeed += 1; raw = makeData(pSeed); rebuildPaths();
});
document.getElementById("best-lam").addEventListener("click", () => {
  lamSlider.value = bestIndex(); updatePaths();
});
document.getElementById("reset-paths").addEventListener("click", () => {
  pSeed = P_SEED0; raw = makeData(pSeed);
  document.querySelector("input[name=method][value=lasso]").checked = true;
  rhoSlider.value = 0.5; dupToggle.checked = false; setMethodControls();
  rebuildPaths(); lamSlider.value = bestIndex(); updatePaths();
});

// =====================================================================
// Geometry view
// =====================================================================
const GW = 520, GH = 520, GM = 34, GLIM = 3.2;
const geomSvg = d3.select("#geom").append("svg").attr("viewBox", `0 0 ${GW} ${GH}`).attr("role", "img")
  .style("max-width", "520px").style("margin", "0 auto");
const geomTitle = geomSvg.append("title").attr("id", "geom-title");
const geomDesc = geomSvg.append("desc").attr("id", "geom-desc");
geomSvg.attr("aria-labelledby", "geom-title geom-desc");
const bx = d3.scaleLinear().domain([-GLIM, GLIM]).range([GM, GW - GM]);
const by = d3.scaleLinear().domain([-GLIM, GLIM]).range([GH - GM, GM]);
geomSvg.append("defs").append("clipPath").attr("id", "geom-clip").append("rect")
  .attr("x", GM).attr("y", GM).attr("width", GW - 2 * GM).attr("height", GH - 2 * GM);
geomSvg.append("g").attr("class", "axis").attr("transform", `translate(0,${by(0)})`)
  .call(d3.axisBottom(bx).ticks(6).tickFormat(v => (v === 0 ? "" : minus(v))));
geomSvg.append("g").attr("class", "axis").attr("transform", `translate(${bx(0)},0)`)
  .call(d3.axisLeft(by).ticks(6).tickFormat(v => (v === 0 ? "" : minus(v))));
geomSvg.append("text").attr("class", "axis-label").attr("x", GW - GM).attr("y", by(0) - 6).attr("text-anchor", "end").text("β₁");
geomSvg.append("text").attr("class", "axis-label").attr("x", bx(0) + 6).attr("y", GM + 4).text("β₂");
const gClip = geomSvg.append("g").attr("clip-path", "url(#geom-clip)");
const regionPath = gClip.append("path").attr("fill", "#d9e3f2").attr("fill-opacity", 0.85)
  .attr("stroke", "#1a5fb4").attr("stroke-width", 2);
const contourLayer = gClip.append("g");
const segLine = gClip.append("line").attr("stroke", "#888").attr("stroke-width", 1).attr("stroke-dasharray", "1 3");
const solMark = geomSvg.append("rect").attr("width", 11).attr("height", 11).attr("fill", "#111");
const solLabel = geomSvg.append("text").attr("class", "line-label").style("font-weight", 600);
const olsG = geomSvg.append("g").attr("class", "ols-point").attr("tabindex", 0)
  .attr("role", "button").attr("aria-roledescription", "draggable point");
olsG.append("circle").attr("r", 16).attr("fill", "transparent");
olsG.append("circle").attr("class", "ring").attr("r", 7).attr("fill", "#fff").attr("stroke", "#c01c28").attr("stroke-width", 2.5);
const olsLabel = geomSvg.append("text").attr("class", "line-label").style("font-weight", 600).text("OLS");

const G_DEFAULT = { bhat: [1.0, 2.6], t: 1.2, r: 0.5 };
let bhat = G_DEFAULT.bhat.slice();
const gMethodOf = () => document.querySelector("input[name=gmethod]:checked").value;
const grhoSlider = document.getElementById("grho-slider");
const tSlider = document.getElementById("t-slider");
const rSlider = document.getElementById("r-slider");
const gRhoOf = () => ({ ridge: 0, lasso: 1, enet: +grhoSlider.value }[gMethodOf()]);

const penalty = (b, rho) => rho * (Math.abs(b[0]) + Math.abs(b[1])) + (1 - rho) / 2 * (b[0] ** 2 + b[1] ** 2);
const quadAt = (b, bh, r) => {
  const d0 = b[0] - bh[0], d1 = b[1] - bh[1];
  return d0 * d0 + 2 * r * d0 * d1 + d1 * d1;
};
const quad = (b, r) => quadAt(b, bhat, r);

// Exact minimizer of ½(β−b̂)ᵀA(β−b̂) + μ P(β) in 2-D, for OLS point bh: try every support and sign
// pattern, keep the candidates whose signs agree, and take the one with the smallest objective
// (the problem is convex). Used by the Geometry view (with bisection on μ) and the Loss view.
function penalized2(mu, rho, r, bh = bhat) {
  const A = [[1, r], [r, 1]];
  const Ab = [bh[0] + r * bh[1], r * bh[0] + bh[1]];
  const l1 = mu * rho, l2 = mu * (1 - rho);
  const obj = b => 0.5 * quadAt(b, bh, r) + mu * penalty(b, rho);
  let best = [0, 0], bestVal = obj(best);
  for (const j of [0, 1]) {
    for (const s of [1, -1]) {
      const v = (Ab[j] - l1 * s) / (A[j][j] + l2);
      if (Math.sign(v) === s) {
        const b = [0, 0]; b[j] = v;
        const val = obj(b);
        if (val < bestVal) { best = b; bestVal = val; }
      }
    }
  }
  for (const s0 of [1, -1]) for (const s1 of [1, -1]) {
    const v = solve([[1 + l2, r], [r, 1 + l2]], [Ab[0] - l1 * s0, Ab[1] - l1 * s1]);
    if (Math.sign(v[0]) === s0 && Math.sign(v[1]) === s1) {
      const val = obj(v);
      if (val < bestVal) { best = v; bestVal = val; }
    }
  }
  return best;
}

const capOf = (t, rho) => rho * t + (1 - rho) / 2 * t * t;   // region passes through (t, 0)

function constrained(t, rho, r) {
  const cap = capOf(t, rho);
  if (penalty(bhat, rho) <= cap) return { b: bhat.slice(), active: false };
  let lo = 0, hi = 1;
  while (penalty(penalized2(hi, rho, r), rho) > cap && hi < 1e8) hi *= 2;
  for (let it = 0; it < 80; it++) {
    const mid = (lo + hi) / 2;
    if (penalty(penalized2(mid, rho, r), rho) > cap) lo = mid; else hi = mid;
  }
  const b = penalized2(hi, rho, r);
  for (const j of [0, 1]) if (Math.abs(b[j]) < 1e-9) b[j] = 0;
  return { b, active: true };
}

// Boundary of {P(β) ≤ cap} traced by angle: for direction (cos θ, sin θ), the radius s solves
// ρ s (|cos|+|sin|) + (1−ρ)/2 s² = cap.
function regionPoints(t, rho) {
  const cap = rho * t + (1 - rho) / 2 * t * t;
  return d3.range(0, 721).map(k => {
    const th = (k / 720) * 2 * Math.PI, cs = Math.cos(th), sn = Math.sin(th);
    const a = (1 - rho) / 2, bq = rho * (Math.abs(cs) + Math.abs(sn));
    const s = a === 0 ? cap / bq : (-bq + Math.sqrt(bq * bq + 4 * a * cap)) / (2 * a);
    return [s * cs, s * sn];
  });
}
// Ellipse (β−b̂)ᵀA(β−b̂) = q. A has eigenvalues 1+r along (1,1)/√2 and 1−r along (1,−1)/√2.
function ellipsePoints(q, r) {
  const a1 = Math.sqrt(q / (1 + r)), a2 = Math.sqrt(q / (1 - r));
  return d3.range(0, 361).map(k => {
    const th = (k / 360) * 2 * Math.PI;
    const u = a1 * Math.cos(th), v = a2 * Math.sin(th);
    return [bhat[0] + (u + v) / Math.SQRT2, bhat[1] + (u - v) / Math.SQRT2];
  });
}
const toPath = d3.line().x(d => bx(d[0])).y(d => by(d[1]));

let geomLive = null;
function updateGeom() {
  const method = gMethodOf(), rho = gRhoOf(), t = +tSlider.value, r = +rSlider.value;
  setMixControl("grho", method);
  document.getElementById("grho-out").textContent = `ρ = ${rho}`;
  document.getElementById("t-out").textContent = `t = ${fmt2(t)}`;
  document.getElementById("r-out").textContent = `r = ${minus(fmt2(r))}`;
  regionPath.attr("d", toPath(regionPoints(t, rho)) + "Z");
  const { b, active } = constrained(t, rho, r);
  const qStar = quad(b, r);
  const levels = active ? [0.12, 0.45, 1].map(f => f * qStar) : [0.25, 1, 2.5];
  contourLayer.selectAll("path").data(levels).join("path")
    .attr("fill", "none").attr("stroke", "#555").attr("stroke-dasharray", "5 4")
    .attr("stroke-width", (d, i) => (active && i === 2 ? 2 : 1.2))
    .attr("d", q => toPath(ellipsePoints(q, r)) + "Z");
  segLine.attr("x1", bx(bhat[0])).attr("y1", by(bhat[1])).attr("x2", bx(b[0])).attr("y2", by(b[1]))
    .attr("display", active ? null : "none");
  solMark.attr("x", bx(b[0]) - 5.5).attr("y", by(b[1]) - 5.5);
  const sx = bx(b[0]), sy = by(b[1]);
  solLabel.attr("x", sx + (b[0] >= 0 ? 9 : -9)).attr("y", sy + 18).attr("text-anchor", b[0] >= 0 ? "start" : "end")
    .text("solution");
  olsG.attr("transform", `translate(${bx(bhat[0])},${by(bhat[1])})`);
  olsLabel.attr("x", bx(bhat[0]) + 10).attr("y", by(bhat[1]) - 10);

  const pair = p => `(${signed(p[0])}, ${signed(p[1])})`;
  const zeros = [0, 1].filter(j => b[j] === 0);
  const regionName = { ridge: "disk", lasso: "diamond", enet: "elastic-net region" }[method];
  let msg;
  if (!active) msg = "OLS is inside the region: solution = OLS";
  else if (zeros.length === 2) msg = "Both exactly 0";
  else if (zeros.length === 1) msg = `Corner: β̂${zeros[0] === 0 ? "₁" : "₂"} = 0 exactly`;
  else msg = "Both non-zero";
  olsG.attr("aria-label", `OLS estimate at β1 = ${signed(bhat[0])}, β2 = ${signed(bhat[1])}. Arrow keys move it.`);
  geomTitle.text(`Constraint region and RSS contours: ${regionName}, t = ${fmt2(t)}, correlation ${minus(fmt2(r))}`);
  geomDesc.text(
    `Plot of β1 against β2 from −3.2 to 3.2. The shaded ${regionName} passes through (±${fmt2(t)}, 0) and (0, ±${fmt2(t)})` +
    `${method === "lasso" ? " and has corners on the axes" : method === "ridge" ? " and has no corners" : ", with rounded edges and corners on the axes"}. ` +
    `The OLS estimate is at ${pair(bhat)}. Elliptical RSS contours are centred on it, tilted ` +
    `${r > 0 ? "along the line β1 = −β2" : r < 0 ? "along the line β1 = β2" : "with no tilt (circles)"}` +
    `${Math.abs(r) >= 0.9 ? ", and very elongated because the predictors are highly correlated" : ""}. ` +
    `Solution: ${pair(b)}. ${msg}.`);

  clearTimeout(geomLive);
  geomLive = setTimeout(() => {
    document.getElementById("g-sol").textContent = pair(b);
    document.getElementById("g-msg").textContent = msg;
  }, liveDelay());
}

const clampB = v => Math.max(-GLIM + 0.05, Math.min(GLIM - 0.05, v));
olsG.call(d3.drag()
  .on("start", () => olsG.style("cursor", "grabbing"))
  .on("drag", event => {
    const [mx, my] = d3.pointer(event, geomSvg.node());
    bhat = [clampB(bx.invert(mx)), clampB(by.invert(my))];
    updateGeom();
  })
  .on("end", () => olsG.style("cursor", null)));
olsG.on("keydown", event => {
  const step = event.shiftKey ? 0.5 : 0.05;
  const d = { ArrowRight: [step, 0], ArrowLeft: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[event.key];
  if (!d) return;
  event.preventDefault();
  bhat = [clampB(Math.round((bhat[0] + d[0]) * 100) / 100), clampB(Math.round((bhat[1] + d[1]) * 100) / 100)];
  updateGeom();
});
document.querySelectorAll("input[name=gmethod]").forEach(el => el.addEventListener("change", updateGeom));
[grhoSlider, tSlider, rSlider].forEach(el => el.addEventListener("input", updateGeom));
document.getElementById("reset-geom").addEventListener("click", () => {
  bhat = G_DEFAULT.bhat.slice(); tSlider.value = G_DEFAULT.t; rSlider.value = G_DEFAULT.r; grhoSlider.value = 0.5;
  document.querySelector("input[name=gmethod][value=lasso]").checked = true;
  updateGeom();
});

// =====================================================================
// One-predictor view: the loss RSS(β) + λ·penalty as a curve in β, in ISL's scaling.
// One standardized predictor and centred y, n = 20, so Σx² = 20; the least-squares slope is
// β̂ = 2 and RSS(β̂) = 20. Then RSS(β) = 20 + 20(β − 2)², and the minimizer is
//   ridge: β̂_λ = Σxy / (Σx² + λ) = 40 / (20 + λ)
//   lasso: β̂_λ = (Σxy − λ/2)₊ / Σx² = (40 − λ/2)₊ / 20, which is exactly 0 for λ ≥ 80.
// =====================================================================
const O_SXX = 20, O_BHAT = 2, O_RSS0 = 20, O_SXY = O_SXX * O_BHAT;
const oRSS = b => O_RSS0 + O_SXX * (b - O_BHAT) ** 2;
const oPen = (b, lam, method) => (method === "ridge" ? lam * b * b : lam * Math.abs(b));
const oMin = (lam, method) => (method === "ridge" ? O_SXY / (O_SXX + lam) : Math.max(O_SXY - lam / 2, 0) / O_SXX);

const OW = 720, OH = 380, om = { top: 14, right: 20, bottom: 42, left: 52 };
const oneSvg = d3.select("#one").append("svg").attr("viewBox", `0 0 ${OW} ${OH}`).attr("role", "img");
const oneTitle = oneSvg.append("title").attr("id", "one-title");
const oneDesc = oneSvg.append("desc").attr("id", "one-desc");
oneSvg.attr("aria-labelledby", "one-title one-desc");
const ox = d3.scaleLinear().domain([-1, 3]).range([om.left, OW - om.right]);
const oy = d3.scaleLinear().domain([0, 200]).range([OH - om.bottom, om.top]);   // fixed, so a rising loss shows
oneSvg.append("defs").append("clipPath").attr("id", "one-clip").append("rect")
  .attr("x", om.left).attr("y", om.top).attr("width", OW - om.left - om.right).attr("height", OH - om.top - om.bottom);
oneSvg.append("g").attr("class", "axis").attr("transform", `translate(0,${OH - om.bottom})`)
  .call(d3.axisBottom(ox).ticks(8).tickFormat(v => minus(v)));
oneSvg.append("g").attr("class", "axis").attr("transform", `translate(${om.left},0)`).call(d3.axisLeft(oy).ticks(5));
oneSvg.append("text").attr("class", "axis-label").attr("x", (om.left + OW - om.right) / 2).attr("y", OH - 6)
  .attr("text-anchor", "middle").text("coefficient β");
oneSvg.append("text").attr("class", "axis-label").attr("transform", `translate(14,${(om.top + OH - om.bottom) / 2}) rotate(-90)`)
  .attr("text-anchor", "middle").text("loss");
oneSvg.append("line").attr("x1", ox(0)).attr("x2", ox(0)).attr("y1", om.top).attr("y2", OH - om.bottom).attr("stroke", "#bbb");
oneSvg.append("line").attr("x1", ox(O_BHAT)).attr("x2", ox(O_BHAT)).attr("y1", oy(O_RSS0)).attr("y2", OH - om.bottom)
  .attr("stroke", "#888").attr("stroke-dasharray", "2 3");
oneSvg.append("text").attr("class", "line-label").attr("x", ox(O_BHAT) + 5).attr("y", OH - om.bottom - 5).text("least squares, β̂ = 2");

const oClip = oneSvg.append("g").attr("clip-path", "url(#one-clip)");
const oGrid = d3.range(0, 401).map(k => -1 + (4 * k) / 400);
const oLine = d3.line().x(d => ox(d[0])).y(d => oy(d[1]));
const rssCurve = oClip.append("path").attr("fill", "none").attr("stroke", "#5e5c64").attr("stroke-width", 2).attr("stroke-dasharray", "7 4");
const penCurve = oClip.append("path").attr("fill", "none").attr("stroke", "#c64600").attr("stroke-width", 2).attr("stroke-dasharray", "2 3");
const lossCurve = oClip.append("path").attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 3);
// Stacked bar under the minimum: the RSS part from 0, then the penalty part on top.
const rssBar = oneSvg.append("rect").attr("width", 12).attr("fill", "#d4d3d8").attr("stroke", "#5e5c64");
const penBar = oneSvg.append("rect").attr("width", 12).attr("fill", "#f3c9a8").attr("stroke", "#c64600");
const oDot = oneSvg.append("circle").attr("r", 6).attr("fill", "#1a5fb4").attr("stroke", "#fff").attr("stroke-width", 2);
const oLabels = oneSvg.append("g");

let oneLive = null;
function updateOne() {
  const method = document.querySelector("input[name=omethod]:checked").value;
  const lam = +document.getElementById("olam-slider").value;
  document.getElementById("olam-out").textContent = `λ = ${lam}`;

  const b = oMin(lam, method), rss = oRSS(b), pen = oPen(b, lam, method), loss = rss + pen;
  rssCurve.attr("d", oLine(oGrid.map(t => [t, oRSS(t)])));
  penCurve.attr("d", oLine(oGrid.map(t => [t, oPen(t, lam, method)])));
  lossCurve.attr("d", oLine(oGrid.map(t => [t, oRSS(t) + oPen(t, lam, method)])));

  const bx0 = ox(b) - 6;
  rssBar.attr("x", bx0).attr("y", oy(rss)).attr("height", oy(0) - oy(rss));
  penBar.attr("x", bx0).attr("y", oy(loss)).attr("height", Math.max(oy(rss) - oy(loss), 0));
  oDot.attr("cx", ox(b)).attr("cy", oy(loss));

  // Labels beside the bar: the two parts and the total. Put them on whichever side has room.
  const right = b < 1.6, ax = ox(b) + (right ? 12 : -12), anchor = right ? "start" : "end";
  const lbl = [
    { y: oy(loss) - 10, text: `minimum loss ${fmt2(loss)} at β = ${fmt3(b)}`, bold: true },
    { y: (oy(rss) + oy(loss)) / 2 + 4, text: `penalty ${fmt2(pen)}` },
    { y: (oy(0) + oy(rss)) / 2 + 4, text: `RSS ${fmt2(rss)}` },
  ].filter(d => pen > 0.005 || !d.text.startsWith("penalty"));   // no penalty slice to label at λ = 0 or β = 0
  // Keep labels at least 14px apart (the penalty slice can be very thin).
  for (let i = 1; i < lbl.length; i++) lbl[i].y = Math.max(lbl[i].y, lbl[i - 1].y + 14);
  oLabels.selectAll("text").data(lbl).join("text").attr("class", "line-label")
    .attr("x", ax).attr("y", d => d.y).attr("text-anchor", anchor).style("font-weight", d => (d.bold ? 600 : 400))
    .style("paint-order", "stroke").style("stroke", "#fff").style("stroke-width", "3px").text(d => d.text);

  document.getElementById("o-pen-key").textContent = method === "ridge" ? "penalty λβ²" : "penalty λ|β|";

  const name = method === "ridge" ? "Ridge" : "Lasso";
  const zero = method === "lasso" && b === 0;
  oneTitle.text(`${name} loss for one predictor at λ = ${lam}`);
  oneDesc.text(
    `Loss against the coefficient β from −1 to 3, on a fixed vertical scale from 0 to 200. ` +
    `A dashed grey parabola is the RSS, 20 + 20(β − 2)², lowest at the least-squares value β = 2. ` +
    (method === "ridge"
      ? `A dotted orange parabola is the penalty λβ². The solid blue loss is their sum, also a parabola, narrower than the RSS. `
      : `A dotted orange V is the penalty λ|β|. The solid blue loss is their sum, with a corner at β = 0. `) +
    `At λ = ${lam} its minimum is at β = ${fmt3(b)}${zero ? ", exactly zero" : ""}, with loss ${fmt2(loss)}: ` +
    `RSS ${fmt2(rss)} plus penalty ${fmt2(pen)}, shown as a stacked bar under the minimum. ` +
    `As λ grows the minimum moves toward 0 and rises.`);

  document.getElementById("o-pen-name").innerHTML =
    method === "ridge" ? "penalty, λβ̂<sub>λ</sub>²" : "penalty, λ|β̂<sub>λ</sub>|";
  clearTimeout(oneLive);
  oneLive = setTimeout(() => {
    document.getElementById("o-beta").textContent = fmt3(b) + (zero ? " (exactly 0)" : "");
    document.getElementById("o-rss").textContent = fmt2(rss);
    document.getElementById("o-pen").textContent = fmt2(pen);
    document.getElementById("o-loss").textContent = fmt2(loss);
  }, liveDelay());
}
document.querySelectorAll("input[name=omethod]").forEach(el => el.addEventListener("change", updateOne));
document.getElementById("olam-slider").addEventListener("input", updateOne);
document.getElementById("reset-one").addEventListener("click", () => {
  document.getElementById("olam-slider").value = 20;
  document.querySelector("input[name=omethod][value=ridge]").checked = true;
  updateOne();
});

// =====================================================================
// Loss view: level curves of the penalized loss
//   L(β) = ½(β − b̂)ᵀR(β − b̂) + λ[ρ|β|₁ + (1−ρ)/2 |β|²],   R = [[1, r], [r, 1]].
// For two standardized predictors this is the Paths view's loss (1/2n)RSS + λ·penalty up to an
// additive constant, so λ means the same thing in both views.
// =====================================================================
const LW = 520, LH = 520, LM = 34, LLIM = 3.2, LGRID = 161, LAM_MAX = 4;
const L_OLS = [1.0, 2.6];
const lossSvg = d3.select("#loss").append("svg").attr("viewBox", `0 0 ${LW} ${LH}`).attr("role", "img")
  .style("max-width", "520px").style("margin", "0 auto");
const lossTitle = lossSvg.append("title").attr("id", "loss-title");
const lossDesc = lossSvg.append("desc").attr("id", "loss-desc");
lossSvg.attr("aria-labelledby", "loss-title loss-desc");
const lx = d3.scaleLinear().domain([-LLIM, LLIM]).range([LM, LW - LM]);
const ly = d3.scaleLinear().domain([-LLIM, LLIM]).range([LH - LM, LM]);
lossSvg.append("defs").append("clipPath").attr("id", "loss-clip").append("rect")
  .attr("x", LM).attr("y", LM).attr("width", LW - 2 * LM).attr("height", LH - 2 * LM);
const lossClip = lossSvg.append("g").attr("clip-path", "url(#loss-clip)");
const fillLayer = lossClip.append("g");
const lineLayer = lossClip.append("g");
lossSvg.append("g").attr("class", "axis").attr("transform", `translate(0,${ly(0)})`)
  .call(d3.axisBottom(lx).ticks(6).tickFormat(v => (v === 0 ? "" : minus(v))));
lossSvg.append("g").attr("class", "axis").attr("transform", `translate(${lx(0)},0)`)
  .call(d3.axisLeft(ly).ticks(6).tickFormat(v => (v === 0 ? "" : minus(v))));
lossSvg.append("text").attr("class", "axis-label").attr("x", LW - LM).attr("y", ly(0) - 6).attr("text-anchor", "end").text("β₁");
lossSvg.append("text").attr("class", "axis-label").attr("x", lx(0) + 6).attr("y", LM + 4).text("β₂");
const lossPath = lossSvg.append("path").attr("fill", "none").attr("stroke", "#c64600").attr("stroke-width", 2)
  .attr("stroke-dasharray", "6 4");
lossSvg.append("circle").attr("cx", lx(L_OLS[0])).attr("cy", ly(L_OLS[1])).attr("r", 7)
  .attr("fill", "#fff").attr("stroke", "#c01c28").attr("stroke-width", 2.5);
lossSvg.append("text").attr("class", "line-label").style("font-weight", 600)
  .attr("x", lx(L_OLS[0]) + 10).attr("y", ly(L_OLS[1]) - 10).text("OLS (λ = 0)");
const lossMin = lossSvg.append("rect").attr("width", 11).attr("height", 11).attr("fill", "#111")
  .attr("stroke", "#fff").attr("stroke-width", 1.5);
const lossMinLabel = lossSvg.append("text").attr("class", "line-label").style("font-weight", 600).text("minimum");

const lMethodOf = () => document.querySelector("input[name=lmethod]:checked").value;
const lrhoSlider = document.getElementById("lrho-slider");
const llamSlider = document.getElementById("llam-slider");
const lrSlider = document.getElementById("lr-slider");
const lRhoOf = () => ({ ridge: 0, lasso: 1, enet: +lrhoSlider.value }[lMethodOf()]);
const lossAt = (b, lam, rho, r) => 0.5 * quadAt(b, L_OLS, r) + lam * penalty(b, rho);
// One fixed colour scale for every λ, so a rising loss shows up as a lighter surface (not just a
// shifted one). Levels 0, 0.25, 1, 2.25, … , 90.25 are evenly spaced on the square-root scale used
// for the colours, so each band is one equal colour step. (The loss stays below about 91 on this
// plot for every method, λ ≤ 4 and |r| ≤ 0.99.)
const LVMAX = 90.25;
const LLEVELS = d3.range(0, 20).map(k => 0.25 * k * k);
const LCOLOR = v => d3.interpolateBlues(0.8 - 0.75 * Math.sqrt(Math.min(v, LVMAX) / LVMAX));   // darker = lower

// Colour legend: a bar on the same square-root scale, with a marker at the current minimum loss.
const KW = 520, KH = 72, KM = { left: 34, right: 34 };
const keySvg = d3.select("#loss-legend").append("svg").attr("viewBox", `0 0 ${KW} ${KH}`).attr("role", "img")
  .style("max-width", "520px").style("margin", "0 auto");
const keyTitle = keySvg.append("title").attr("id", "key-title");
keySvg.attr("aria-labelledby", "key-title");
const kx = d3.scaleSqrt().domain([0, LVMAX]).range([KM.left, KW - KM.right]);
keySvg.selectAll("rect.band").data(LLEVELS).join("rect").attr("class", "band")
  .attr("x", v => kx(v)).attr("y", 20).attr("height", 14)
  .attr("width", (v, i) => kx(i + 1 < LLEVELS.length ? LLEVELS[i + 1] : LVMAX) - kx(v))
  .attr("fill", v => LCOLOR(v)).attr("stroke", "#334").attr("stroke-width", 0.4);
keySvg.append("g").attr("class", "axis").attr("transform", "translate(0,34)")
  .call(d3.axisBottom(kx).tickValues([0, 1, 4, 9, 16, 25, 36, 49, 64, 81]).tickSize(4));
keySvg.append("text").attr("class", "axis-label").attr("x", KW / 2).attr("y", 68).attr("text-anchor", "middle")
  .text("loss, square-root scale (darker = lower)");
const keyMark = keySvg.append("path").attr("d", "M0,0 L-6,-9 L6,-9 Z").attr("fill", "#111");
const keyMarkLabel = keySvg.append("text").attr("class", "line-label").style("font-weight", 600).attr("y", 12);

let lossLive = null;
function updateLoss() {
  const method = lMethodOf(), rho = lRhoOf(), lam = +llamSlider.value, r = +lrSlider.value;
  setMixControl("lrho", method);
  document.getElementById("lrho-out").textContent = `ρ = ${rho}`;
  document.getElementById("llam-out").textContent = `λ = ${fmt2(lam)}`;
  document.getElementById("lr-out").textContent = `r = ${minus(fmt2(r))}`;

  // Loss on a grid, then filled level sets at the fixed levels LLEVELS (absolute loss values).
  const b = penalized2(lam, rho, r, L_OLS).map(v => (Math.abs(v) < 1e-9 ? 0 : v));
  const lmin = lossAt(b, lam, rho, r);
  const vals = new Float64Array(LGRID * LGRID);
  for (let i = 0; i < LGRID; i++) {
    const b2 = LLIM - (2 * LLIM * i) / (LGRID - 1);
    for (let j = 0; j < LGRID; j++) {
      const b1 = -LLIM + (2 * LLIM * j) / (LGRID - 1);
      vals[i * LGRID + j] = lossAt([b1, b2], lam, rho, r);
    }
  }
  const gscale = (LW - 2 * LM) / (LGRID - 1);
  // d3.contours places sample j at grid coordinate j + 0.5, so shift back half a cell.
  const proj = d3.geoTransform({ point(x, y) { this.stream.point(LM + (x - 0.5) * gscale, LM + (y - 0.5) * gscale); } });
  const geoPath = d3.geoPath(proj);
  // A filled contour at level v covers the region where loss ≥ v; drawing them in increasing order
  // leaves each band [v_k, v_k+1) in the colour of v_k.
  const conts = d3.contours().size([LGRID, LGRID]).thresholds(LLEVELS)(vals);
  fillLayer.selectAll("path").data(conts).join("path")
    .attr("d", geoPath).attr("fill", d => LCOLOR(d.value)).attr("stroke", "none");
  lineLayer.selectAll("path").data(conts.slice(1)).join("path")
    .attr("d", geoPath).attr("fill", "none").attr("stroke", "#334").attr("stroke-width", 0.8)
    .attr("stroke-opacity", 0.55);

  // Path of the minimizer as λ runs from 0 to LAM_MAX (recomputed each time; 201 tiny solves).
  const pathPts = d3.range(0, 201).map(k => penalized2((LAM_MAX * k) / 200, rho, r, L_OLS));
  lossPath.attr("d", d3.line().x(d => lx(d[0])).y(d => ly(d[1]))(pathPts));
  lossMin.attr("x", lx(b[0]) - 5.5).attr("y", ly(b[1]) - 5.5);
  lossMinLabel.attr("x", lx(b[0]) + (b[0] >= 0.3 ? 10 : -10)).attr("y", ly(b[1]) + 18)
    .attr("text-anchor", b[0] >= 0.3 ? "start" : "end");
  const mx = kx(lmin);
  keyMark.attr("transform", `translate(${mx},20)`);
  keyMarkLabel.attr("x", mx).attr("text-anchor", mx > KW - 120 ? "end" : mx < 150 ? "start" : "middle")
    .text(`minimum loss ${fmt2(lmin)}`);
  keyTitle.text(`Colour scale for loss from 0 (dark blue) to 90 (pale), on a square-root scale. ` +
    `The current minimum loss, ${fmt2(lmin)}, is marked on it.`);

  const pair = p => `(${signed(p[0])}, ${signed(p[1])})`;
  const zeros = [0, 1].filter(j => b[j] === 0);
  const name = { ridge: "Ridge", lasso: "Lasso", enet: `Elastic net (ρ = ${rho})` }[method];
  let msg;
  if (lam === 0) msg = "λ = 0: the minimum is OLS";
  else if (zeros.length === 2) msg = "Both exactly 0";
  else if (zeros.length === 1) msg = `β̂${zeros[0] === 0 ? "₁" : "₂"} = 0 exactly`;
  else msg = "Both non-zero";
  lossTitle.text(`${name} loss level curves at λ = ${fmt2(lam)}, correlation ${minus(fmt2(r))}`);
  lossDesc.text(
    `Contour plot of the penalized loss over β1 and β2 from −3.2 to 3.2, on a fixed colour scale from 0 (dark) to 90 (pale), so the surface gets paler as λ raises the loss. ` +
    `OLS, the minimum at λ = 0, is at ${pair(L_OLS)}. ` +
    (method === "ridge" ? "The ridge penalty is smooth, so the level curves stay smooth ellipses as they move toward the origin. "
      : "The absolute-value penalty puts corners in the level curves where they cross the axes. ") +
    `At λ = ${fmt2(lam)} the minimum is at ${pair(b)}, loss ${fmt2(lmin)}. ${msg}. ` +
    `A dashed curve traces the minimum from OLS at λ = 0 to ${pair(pathPts[pathPts.length - 1])} at λ = ${LAM_MAX}.`);

  clearTimeout(lossLive);
  lossLive = setTimeout(() => {
    document.getElementById("l-min").textContent = pair(b);
    document.getElementById("l-loss").textContent = fmt2(lmin);
    document.getElementById("l-msg").textContent = msg;
  }, liveDelay());
}
document.querySelectorAll("input[name=lmethod]").forEach(el => el.addEventListener("change", updateLoss));
[lrhoSlider, llamSlider, lrSlider].forEach(el => el.addEventListener("input", updateLoss));
document.getElementById("reset-loss").addEventListener("click", () => {
  llamSlider.value = 0.5; lrSlider.value = 0.5; lrhoSlider.value = 0.5;
  document.querySelector("input[name=lmethod][value=lasso]").checked = true;
  updateLoss();
});

// =====================================================================
// Shared: Shift+arrow steps sliders by 10; tabs
// =====================================================================
document.querySelectorAll("input[type=range]").forEach(sl => sl.addEventListener("keydown", event => {
  if (!event.shiftKey) return;
  const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[event.key];
  if (dir === undefined) return;
  event.preventDefault();
  const step = +sl.step, lo = +sl.min, hi = +sl.max;
  const v = Math.min(hi, Math.max(lo, +sl.value + 10 * step * dir));
  sl.value = Math.round(v / step) * step;
  sl.dispatchEvent(new Event("input"));
}));

const tabs = [...document.querySelectorAll("[role=tab]")];
function selectTab(tab, focus) {
  tabs.forEach(t => {
    const on = t === tab;
    t.setAttribute("aria-selected", on);
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
  });
  if (focus) tab.focus();
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t, false));
  t.addEventListener("keydown", event => {
    const k = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (event.key === "Home") { event.preventDefault(); selectTab(tabs[0], true); }
    else if (event.key === "End") { event.preventDefault(); selectTab(tabs[tabs.length - 1], true); }
    else if (k) { event.preventDefault(); selectTab(tabs[(i + k + tabs.length) % tabs.length], true); }
  });
});

// ---------- start ----------
raw = makeData(pSeed);
setMethodControls();
rebuildPaths();
lamSlider.value = bestIndex();
updatePaths();
updateGeom();
updateOne();
updateLoss();
started = true;
