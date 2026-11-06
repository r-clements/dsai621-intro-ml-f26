// Basis expansions (ISL 7.1–7.4, ESL 5.1–5.2). Every fit is OLS of y on the M = df columns of a
// basis matrix that spans the constant (so no separate intercept column; ISL's β0 + Σ βj bj(x) is
// the same fit by a change of basis), solved by QR (modified Gram–Schmidt, reorthogonalized). Only the columns
// change between methods:
//   polynomial    Legendre P_0..P_d on 2x − 1 (same column space as 1, x, …, x^d)   M = d + 1
//   piecewise     Legendre P_0..P_3 on each region, zero outside it                   M = 4(K + 1)
//   cubic spline  cubic B-splines, boundary knots 0 and 1 repeated four times         M = K + 4
//   natural       the B-spline space with f''(0) = f''(1) = 0 (ESL 5.2.1)             M = K + 2
// K = number of interior knots. Natural-spline basis for K ≥ 2: keep the interior B-splines and
// replace the three at each end by two combinations with zero second derivative at the boundary;
// for K ≤ 1, ESL eq. 5.4–5.5. SE band: Var f̂(x0) = σ̂² b(x0)ᵀ(BᵀB)⁻¹b(x0) = σ̂² ‖R⁻ᵀb(x0)‖²,
// with b(x0) the row of basis values at x0.

const N_MAX = 500;
const GRID_N = 1001;          // grid on [0, 1] for the test-error integral
const PLOT_N = 600;           // plotting points, split across regions
const KNOT_MIN_GAP = 0.02;
const KNOT_LO = 0.02, KNOT_HI = 0.98;

const PRESETS = {
  esl:  { f: x => Math.sin(12 * (x + 0.2)) / (x + 0.2), sigma: 1, y: [-5.5, 6.5],
          name: "ESL wave" },
  sine: { f: x => Math.sin(2 * Math.PI * x), sigma: 0.3, y: [-2, 2], name: "sine" },
  kink: { f: x => 1 - 4 * Math.abs(x - 0.6), sigma: 0.3, y: [-2.6, 2], name: "kink" },
  flat: { f: x => 2 / (1 + Math.exp(-25 * (x - 0.5))) - 1, sigma: 0.3, y: [-2, 2],
          name: "flat tails" },
};

const METHODS = {
  poly:    { label: "Polynomial", param: "Degree d", sym: "d", min: 1, max: 15, def: 3,
             df: p => p + 1, formula: p => `d + 1 = ${p + 1}` },
  pwcubic: { label: "Piecewise cubic", param: "Knots K", sym: "K", min: 0, max: 8, def: 2,
             df: p => 4 * (p + 1), formula: p => `4(K + 1) = ${4 * (p + 1)}` },
  cubic:   { label: "Cubic spline", param: "Knots K", sym: "K", min: 0, max: 15, def: 3,
             df: p => p + 4, formula: p => `K + 4 = ${p + 4}` },
  natural: { label: "Natural cubic spline", param: "Knots K", sym: "K", min: 0, max: 15, def: 4,
             df: p => p + 2, formula: p => `K + 2 = ${p + 2}` },
};

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
// x ~ U(0, 1) and standard-normal noise z, drawn once per seed; y = f(x) + σz. Changing the
// preset keeps the draws, and increasing n adds points to the same sample.
function draws(seed) {
  const rand = mulberry32(1000 * seed + 621);
  const norm = normalSampler(rand);
  const x = new Float64Array(N_MAX), z = new Float64Array(N_MAX);
  for (let i = 0; i < N_MAX; i++) { x[i] = rand(); z[i] = norm(); }
  return { x, z };
}

// ---------- basis functions ----------
function legendre(u, d, out, offset) {
  out[offset] = 1;
  if (d >= 1) out[offset + 1] = u;
  for (let j = 1; j < d; j++) {
    out[offset + j + 1] = ((2 * j + 1) * u * out[offset + j] - j * out[offset + j - 1]) / (j + 1);
  }
}

// Cox–de Boor: all order-k (degree k − 1) B-splines on knot vector t at x. x = t[last] is
// moved just inside so the right end belongs to the last interval.
function bsplineVals(x, t, k) {
  const m = t.length;
  const xr = x >= t[m - 1] ? t[m - 1] - 1e-10 : Math.max(x, t[0]);
  let B = new Float64Array(m - 1);
  for (let i = 0; i < m - 1; i++) B[i] = t[i] <= xr && xr < t[i + 1] ? 1 : 0;
  for (let ord = 2; ord <= k; ord++) {
    const nb = m - ord, Bn = new Float64Array(nb);
    for (let i = 0; i < nb; i++) {
      let v = 0;
      const d1 = t[i + ord - 1] - t[i];
      if (d1 > 0) v += (xr - t[i]) / d1 * B[i];
      const d2 = t[i + ord] - t[i + 1];
      if (d2 > 0) v += (t[i + ord] - xr) / d2 * B[i + 1];
      Bn[i] = v;
    }
    B = Bn;
  }
  return B;
}
// d-th derivative: B'_{i,k} = (k − 1)[B_{i,k−1}/(t_{i+k−1} − t_i) − B_{i+1,k−1}/(t_{i+k} − t_{i+1})].
function bsplineDeriv(x, t, k, d) {
  if (d === 0) return bsplineVals(x, t, k);
  const B = bsplineDeriv(x, t, k - 1, d - 1);
  const nb = t.length - k, out = new Float64Array(nb);
  for (let i = 0; i < nb; i++) {
    let v = 0;
    const d1 = t[i + k - 1] - t[i];
    if (d1 > 0) v += B[i] / d1;
    const d2 = t[i + k] - t[i + 1];
    if (d2 > 0) v -= B[i + 1] / d2;
    out[i] = (k - 1) * v;
  }
  return out;
}
const cube = v => (v > 0 ? v * v * v : 0);

// Returns { M, row(x) -> Float64Array(M), edges, discontinuous, describe(j) }.
function makeBasis(method, p, knots) {
  const edges = method === "poly" ? [0, 1] : [0, ...knots, 1];
  const region = x => Math.min(edges.length - 2, d3.bisectRight(knots, x));
  const fmt = v => v.toFixed(2);
  if (method === "poly") {
    return { M: p + 1, edges, discontinuous: false,
      row(x) { const o = new Float64Array(p + 1); legendre(2 * x - 1, p, o, 0); return o; },
      describe: j => `orthogonal polynomial of degree ${j}` };
  }
  if (method === "pwcubic") {
    return { M: 4 * (p + 1), edges, discontinuous: true,
      row(x) {
        const o = new Float64Array(4 * (p + 1)), r = region(x);
        legendre(2 * (x - edges[r]) / (edges[r + 1] - edges[r]) - 1, 3, o, 4 * r);
        return o;
      },
      describe: j => `local polynomial of degree ${j % 4} on the region from x = ` +
        `${fmt(edges[Math.floor(j / 4)])} to ${fmt(edges[Math.floor(j / 4) + 1])}` };
  }
  const t = [0, 0, 0, 0, ...knots, 1, 1, 1, 1];
  const nb = p + 4;
  if (method === "cubic") {
    return { M: nb, edges, discontinuous: false, row: x => bsplineVals(x, t, 4),
      describe: j => `cubic B-spline ${j + 1}` };
  }
  // natural cubic spline
  if (p <= 1) {
    const xi = edges, Kp = xi.length;   // ESL 5.4–5.5 with all K + 2 knots
    const dk = (x, k) => (cube(x - xi[k]) - cube(x - xi[Kp - 1])) / (xi[Kp - 1] - xi[k]);
    return { M: p + 2, edges, discontinuous: false,
      row(x) {
        const o = new Float64Array(p + 2);
        o[0] = 1; o[1] = x;
        for (let k = 0; k < Kp - 2; k++) o[k + 2] = dk(x, k) - dk(x, Kp - 2);
        return o;
      },
      describe: j => ["constant (the intercept)", "linear term b₁(x) = x", "natural-spline term b₂(x) = d₀(x) − d₁(x) (lecture Result 5.1)"][j] };
  }
  // Only B_0..B_2 have nonzero f'' at 0 and only B_{nb−3}..B_{nb−1} at 1, and the three values
  // at each end sum to 0 (the B-splines sum to 1). So B_0 + B_1 + B_2 and c_2B_1 − c_1B_2 are
  // natural at 0; likewise at 1.
  const cL = bsplineDeriv(0, t, 4, 2), cR = bsplineDeriv(1, t, 4, 2);
  const wl = Math.abs(cL[2]) + Math.abs(cL[1]);
  const wr = Math.abs(cR[nb - 3]) + Math.abs(cR[nb - 2]);
  return { M: nb - 2, edges, discontinuous: false,
    row(x) {
      const B = bsplineVals(x, t, 4), o = new Float64Array(nb - 2);
      o[0] = B[0] + B[1] + B[2];
      o[1] = (cL[2] * B[1] - cL[1] * B[2]) / wl;
      for (let i = 3; i <= nb - 4; i++) o[i - 1] = B[i];
      o[nb - 4] = (cR[nb - 3] * B[nb - 2] - cR[nb - 2] * B[nb - 3]) / wr;
      o[nb - 3] = B[nb - 3] + B[nb - 2] + B[nb - 1];
      return o;
    },
    describe: j => j === 0 || j === nb - 3 ? "boundary natural-spline function (sum of three end B-splines)"
      : j === 1 || j === nb - 4 ? "boundary natural-spline function (zero second derivative at the boundary)"
      : `interior cubic B-spline ${j + 1}` };
}

// ---------- least squares by QR ----------
// Columns whose residual norm after projection is below 1e-9 of their norm are dropped
// (not identified), so the fit is the projection onto the column space either way.
function fitOLS(basis, xs, ys) {
  const n = xs.length, M = basis.M;
  const rows = Array.from(xs, x => basis.row(x));
  const Q = [], R = [], kept = [];
  for (let j = 0; j < M; j++) {
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = rows[i][j];
    let norm0 = 0;
    for (let i = 0; i < n; i++) norm0 += v[i] * v[i];
    norm0 = Math.sqrt(norm0);
    const rcol = new Float64Array(M);
    for (let pass = 0; pass < 2; pass++) {
      for (let a = 0; a < Q.length; a++) {
        const q = Q[a];
        let r = 0;
        for (let i = 0; i < n; i++) r += q[i] * v[i];
        for (let i = 0; i < n; i++) v[i] -= r * q[i];
        rcol[a] += r;
      }
    }
    let nrm = 0;
    for (let i = 0; i < n; i++) nrm += v[i] * v[i];
    nrm = Math.sqrt(nrm);
    if (norm0 === 0 || nrm < 1e-9 * norm0) continue;
    for (let i = 0; i < n; i++) v[i] /= nrm;
    rcol[Q.length] = nrm;
    R.push(rcol);           // R[a] is column a of the upper-triangular R
    Q.push(v);
    kept.push(j);
  }
  const r = Q.length;
  const qty = Q.map(q => { let s = 0; for (let i = 0; i < n; i++) s += q[i] * ys[i]; return s; });
  const bk = new Float64Array(r);           // back-solve R b = Qᵀy
  for (let a = r - 1; a >= 0; a--) {
    let s = qty[a];
    for (let c = a + 1; c < r; c++) s -= R[c][a] * bk[c];
    bk[a] = s / R[a][a];
  }
  const beta = new Float64Array(M);
  kept.forEach((j, a) => { beta[j] = bk[a]; });
  let rss = 0, loo = 0, looOk = true;
  for (let i = 0; i < n; i++) {
    let fit = 0, h = 0;
    for (let a = 0; a < r; a++) { fit += Q[a][i] * qty[a]; h += Q[a][i] * Q[a][i]; }
    const e = ys[i] - fit;
    rss += e * e;
    if (h > 1 - 1e-8) looOk = false; else loo += (e / (1 - h)) ** 2;
  }
  const dropped = [];
  for (let j = 0; j < M; j++) if (!kept.includes(j)) dropped.push(j);
  const sigma2 = n > r ? rss / (n - r) : NaN;
  return {
    beta, kept, dropped, rank: r, sigma2,
    trainMSE: rss / n, loocv: looOk ? loo / n : Infinity,
    // Prediction and SE at x0; NaN where a dropped column is nonzero (f̂ not identified there).
    predict(x0) {
      const h = basis.row(x0);
      for (const j of dropped) if (Math.abs(h[j]) > 1e-12) return [NaN, NaN];
      let f = 0;
      for (let j = 0; j < M; j++) f += beta[j] * h[j];
      const aVec = new Float64Array(r);     // forward-solve Rᵀa = h_kept
      let ss = 0;
      for (let a = 0; a < r; a++) {
        let s = h[kept[a]];
        for (let c = 0; c < a; c++) s -= R[a][c] * aVec[c];
        aVec[a] = s / R[a][a];
        ss += aVec[a] * aVec[a];
      }
      return [f, Math.sqrt(sigma2 * ss)];
    },
  };
}

// ---------- state ----------
const S = {
  method: "poly",
  params: Object.fromEntries(Object.entries(METHODS).map(([k, m]) => [k, m.def])),
  preset: "esl", n: 100, seed: 1,
  knots: [], knotsCustom: false,
  bview: "raw", sel: null,
};
let D = draws(S.seed);
let data = null;            // { x, y } for the current n and preset
let current = null;         // { basis, fit, grid }
let sweep = [];

function makeData() {
  const P = PRESETS[S.preset];
  const x = D.x.slice(0, S.n);
  const y = x.map((xi, i) => P.f(xi) + P.sigma * D.z[i]);
  data = { x, y };
}
function quantileKnots(K) {
  const sorted = Float64Array.from(data.x).sort();
  return d3.range(1, K + 1).map(j => d3.quantileSorted(sorted, j / (K + 1)));
}
const K = () => (S.method === "poly" ? 0 : S.params[S.method]);
function maxParam(method) {
  const m = METHODS[method];
  let p = m.max;
  while (p > m.min && m.df(p) > S.n - 3) p--;
  return p;
}
function setKnotsToQuantiles() {
  S.knots = quantileKnots(K());
  S.knotsCustom = false;
}

function testError(fit) {
  const P = PRESETS[S.preset];
  let s = 0;
  for (let g = 0; g < GRID_N; g++) {
    const x0 = g / (GRID_N - 1);
    const fh = fit.predict(x0)[0];
    if (Number.isNaN(fh)) return NaN;
    s += (fh - P.f(x0)) ** 2;
  }
  return P.sigma ** 2 + s / GRID_N;
}

// Plotting x-values: each region gets points in proportion to its width. For a discontinuous
// basis the regions are separated by a NaN so the line breaks at the knot.
function plotXs(edges, discontinuous) {
  const xs = [];
  for (let r = 0; r < edges.length - 1; r++) {
    const lo = edges[r], hi = edges[r + 1];
    const m = Math.max(3, Math.ceil(PLOT_N * (hi - lo)));
    const eps = discontinuous ? 1e-9 : 0;
    for (let i = 0; i < m; i++) xs.push(lo + eps + (hi - lo - 2 * eps) * i / (m - 1));
    if (discontinuous && r < edges.length - 2) xs.push(NaN);
  }
  return xs;
}

function refit() {
  const p = S.params[S.method];
  const basis = makeBasis(S.method, p, S.knots);
  const fit = fitOLS(basis, data.x, data.y);
  const xs = plotXs(basis.edges, basis.discontinuous);
  const grid = xs.map(x => {
    if (Number.isNaN(x)) return { x, f: NaN, se: NaN, h: null };
    const [f, se] = fit.predict(x);
    return { x, f, se, h: basis.row(x) };
  });
  current = { basis, fit, grid, test: testError(fit), p };
}

function recomputeSweep() {
  const m = METHODS[S.method];
  sweep = [];
  for (let p = m.min; p <= maxParam(S.method); p++) {
    const knots = S.method === "poly" ? [] : quantileKnots(p);
    const fit = fitOLS(makeBasis(S.method, p, knots), data.x, data.y);
    sweep.push({ p, df: fit.rank, train: fit.trainMSE, loocv: fit.loocv, test: testError(fit) });
  }
}

// ---------- drawing ----------
const W = 740;
const MG = { top: 14, right: 16, bottom: 40, left: 50 };

function makeChart(id, height, label) {
  const svg = d3.select(`#${id}`).append("svg")
    .attr("viewBox", `0 0 ${W} ${height}`).attr("role", "img").attr("aria-label", label);
  const clipId = `${id}-clip`;
  svg.append("defs").append("clipPath").attr("id", clipId).append("rect")
    .attr("x", MG.left).attr("y", MG.top)
    .attr("width", W - MG.left - MG.right).attr("height", height - MG.top - MG.bottom);
  const gx = svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - MG.bottom})`);
  const gy = svg.append("g").attr("class", "axis").attr("transform", `translate(${MG.left},0)`);
  const plot = svg.append("g").attr("clip-path", `url(#${clipId})`);
  return { svg, gx, gy, plot, height };
}
function axisLabels(c, xl, yl) {
  c.svg.append("text").attr("class", "axis-label").attr("x", (W + MG.left - MG.right) / 2)
    .attr("y", c.height - 6).attr("text-anchor", "middle").text(xl);
  c.svg.append("text").attr("class", "axis-label")
    .attr("transform", `translate(14,${(c.height - MG.bottom + MG.top) / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").text(yl);
}

const fitC = makeChart("fit", 360, "");
const basisC = makeChart("basis", 250, "");
const errC = makeChart("errors", 260, "");
axisLabels(fitC, "x", "y");
axisLabels(basisC, "x", "");
axisLabels(errC, "df (number of coefficients, including the intercept)", "mean squared error");

const xS = d3.scaleLinear().domain([0, 1]).range([MG.left, W - MG.right]);
const fitY = d3.scaleLinear().range([fitC.height - MG.bottom, MG.top]);
const basisY = d3.scaleLinear().range([basisC.height - MG.bottom, MG.top]);
const errX = d3.scaleLinear().range([MG.left, W - MG.right]);
const errY = d3.scaleLinear().range([errC.height - MG.bottom, MG.top]);
fitC.gx.call(d3.axisBottom(xS).ticks(10));
basisC.gx.call(d3.axisBottom(xS).ticks(10));

const knotLines = fitC.plot.append("g");
const bandPath = fitC.plot.append("path").attr("fill", "#1a5fb4").attr("fill-opacity", 0.15);
const bandUp = fitC.plot.append("path").attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-dasharray", "2 2");
const bandLo = fitC.plot.append("path").attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-dasharray", "2 2");
const pointsG = fitC.plot.append("g");
const truePath = fitC.plot.append("path").attr("fill", "none").attr("stroke", "#111")
  .attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
const fitPath = fitC.plot.append("path").attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 3);
const handlesG = fitC.svg.append("g");

const basisLines = basisC.plot.append("g");
const basisSum = basisC.plot.append("path").attr("fill", "none").attr("stroke", "#1a5fb4")
  .attr("stroke-width", 3).attr("stroke-opacity", 0.5);
const basisZero = basisC.plot.append("line").attr("stroke", "#bbb");
basisC.svg.attr("tabindex", 0).attr("aria-describedby", "basis-caption");

const errG = errC.plot.append("g");
const errSigma = errC.plot.append("line").attr("stroke", "#999").attr("stroke-width", 1.5);
const errCurrent = errC.plot.append("path").attr("fill", "#c64600");

const BASIS_COLOURS = ["#1a5fb4", "#c64600", "#26a269", "#813d9c", "#986a44", "#555"];
const BASIS_DASHES = [null, "6 3", "2 3", "8 3 2 3"];

const fmt3 = v => (Number.isFinite(v) ? v.toFixed(3) : v === Infinity ? "∞" : "–");

function lineFor(yScale, acc) {
  return d3.line().defined(d => !Number.isNaN(d.x) && Number.isFinite(acc(d)))
    .x(d => xS(d.x)).y(d => yScale(Math.max(-1e4, Math.min(1e4, acc(d)))));
}

function drawFit() {
  const P = PRESETS[S.preset];
  fitY.domain(P.y);
  fitC.gy.call(d3.axisLeft(fitY).ticks(6));
  const g = current.grid;
  const clampY = v => Math.max(-1e4, Math.min(1e4, v));
  bandPath.attr("d", d3.area()
    .defined(d => Number.isFinite(d.f) && Number.isFinite(d.se))
    .x(d => xS(d.x)).y0(d => fitY(clampY(d.f - 2 * d.se))).y1(d => fitY(clampY(d.f + 2 * d.se)))(g));
  bandUp.attr("d", lineFor(fitY, d => (Number.isFinite(d.se) ? d.f + 2 * d.se : NaN))(g));
  bandLo.attr("d", lineFor(fitY, d => (Number.isFinite(d.se) ? d.f - 2 * d.se : NaN))(g));
  fitPath.attr("d", lineFor(fitY, d => d.f)(g));
  const tx = d3.range(0, 501).map(i => ({ x: i / 500 }));
  truePath.attr("d", lineFor(fitY, d => P.f(d.x))(tx));

  pointsG.selectAll("circle").data(d3.range(S.n)).join("circle")
    .attr("r", 2.6).attr("fill", "none").attr("stroke", "#666").attr("stroke-opacity", 0.8)
    .attr("cx", i => xS(data.x[i])).attr("cy", i => fitY(data.y[i]));

  const knots = S.method === "poly" ? [] : S.knots;
  knotLines.selectAll("line").data(knots).join("line")
    .attr("x1", d => xS(d)).attr("x2", d => xS(d))
    .attr("y1", MG.top).attr("y2", fitC.height - MG.bottom)
    .attr("stroke", "#c64600").attr("stroke-dasharray", "3 3");
  drawHandles(knots);

  const knotText = knots.length ? `, knots at x = ${knots.map(k => k.toFixed(2)).join(", ")}` : "";
  fitC.svg.attr("aria-label",
    `Scatterplot of ${S.n} points with the true function (dashed), the ${METHODS[S.method].label.toLowerCase()} fit ` +
    `with ${current.fit.rank} coefficients (solid) and its ±2 standard-error band${knotText}. ` +
    `Largest standard error ${fmt3(d3.max(g, d => d.se))}, at x = ${maxSeX(g)}.`);
}
function maxSeX(g) {
  let best = null;
  for (const d of g) if (Number.isFinite(d.se) && (!best || d.se > best.se)) best = d;
  return best ? best.x.toFixed(2) : "–";
}

const H_Y = fitC.height - MG.bottom;   // handles sit on the x-axis
function drawHandles(knots) {
  const hs = handlesG.selectAll("g.knot-handle").data(knots, (d, i) => i).join(enter => {
    const g = enter.append("g").attr("class", "knot-handle").attr("tabindex", 0).attr("role", "slider")
      .attr("aria-valuemin", KNOT_LO).attr("aria-valuemax", KNOT_HI);
    g.append("rect").attr("x", -11).attr("y", -16).attr("width", 22).attr("height", 22).attr("fill", "transparent");
    g.append("path").attr("d", "M0,-14 l-7,11 h14 z").attr("fill", "#c64600").attr("stroke", "#fff").attr("stroke-width", 1);
    g.call(d3.drag()
      .on("drag", function (event) {
        const i = +this.dataset.idx;
        moveKnot(i, xS.invert(d3.pointer(event, fitC.svg.node())[0]), false);
      })
      .on("end", () => { refreshSweepMarker(); announce(); }));
    g.on("keydown", function (event) {
      const step = event.shiftKey ? 0.05 : 0.01;
      const d = { ArrowLeft: -step, ArrowDown: -step, ArrowRight: step, ArrowUp: step }[event.key];
      if (d === undefined) return;
      event.preventDefault();
      const i = +this.dataset.idx;
      moveKnot(i, +(S.knots[i] + d).toFixed(3), true);
    });
    return g;
  });
  hs.each(function (d, i) { this.dataset.idx = i; })
    .attr("transform", d => `translate(${xS(d)},${H_Y})`)
    .attr("aria-label", (d, i) => `Knot ${i + 1} of ${knots.length}`)
    .attr("aria-valuenow", d => d.toFixed(2))
    .attr("aria-valuetext", d => `x = ${d.toFixed(2)}`);
}
function moveKnot(i, v, announceNow) {
  const lo = i === 0 ? KNOT_LO : S.knots[i - 1] + KNOT_MIN_GAP;
  const hi = i === S.knots.length - 1 ? KNOT_HI : S.knots[i + 1] - KNOT_MIN_GAP;
  S.knots[i] = Math.max(lo, Math.min(hi, v));
  S.knotsCustom = true;
  refit();
  drawFit(); drawBasis(); drawStats(announceNow);
  if (announceNow) refreshSweepMarker();
}

function drawBasis() {
  const g = current.grid, M = current.basis.M, beta = current.fit.beta;
  const scaled = S.bview === "scaled";
  const val = (d, j) => (d.h ? (scaled ? beta[j] * d.h[j] : d.h[j]) : NaN);
  let lo = 0, hi = 0;
  for (const d of g) {
    if (!d.h) continue;
    for (let j = 0; j < M; j++) { const v = val(d, j); if (v < lo) lo = v; if (v > hi) hi = v; }
    if (scaled && Number.isFinite(d.f)) { lo = Math.min(lo, d.f); hi = Math.max(hi, d.f); }
  }
  const span = PRESETS[S.preset].y[1] - PRESETS[S.preset].y[0];
  if (scaled) { lo = Math.max(lo, -3 * span); hi = Math.min(hi, 3 * span); }
  const pad = 0.05 * (hi - lo || 1);
  basisY.domain([lo - pad, hi + pad]).nice();
  basisC.gy.call(d3.axisLeft(basisY).ticks(5));
  basisZero.attr("x1", MG.left).attr("x2", W - MG.right).attr("y1", basisY(0)).attr("y2", basisY(0));

  if (S.sel !== null && S.sel >= M) S.sel = null;
  const line = j => d3.line().defined(d => d.h && Number.isFinite(val(d, j)))
    .x(d => xS(d.x)).y(d => basisY(Math.max(-1e4, Math.min(1e4, val(d, j)))))(g);
  basisLines.selectAll("path").data(d3.range(M)).join("path")
    .attr("fill", "none")
    .attr("d", line)
    .attr("stroke", j => (S.sel === j ? "#111" : BASIS_COLOURS[j % BASIS_COLOURS.length]))
    .attr("stroke-width", j => (S.sel === j ? 3.5 : 1.6))
    .attr("stroke-opacity", j => (S.sel === null || S.sel === j ? 1 : 0.25))
    .attr("stroke-dasharray", j => BASIS_DASHES[j % BASIS_DASHES.length])
    .on("mouseenter", (e, j) => selectBasis(j))
    .on("mouseleave", () => selectBasis(null));
  basisLines.selectAll("path").filter(j => j === S.sel).raise();
  basisSum.attr("d", scaled ? lineFor(basisY, d => d.f)(g) : null);

  const what = scaled ? "each multiplied by its coefficient, with their sum (the fit) as a thick translucent blue line" : "";
  basisC.svg.attr("aria-label", `The ${M} columns of the basis matrix${what ? ", " + what : ""}, for the ${METHODS[S.method].label.toLowerCase()}.`);
  drawReadout();
}
function drawReadout() {
  const el = document.getElementById("basis-readout");
  if (S.sel === null) { el.textContent = ""; return; }
  const j = S.sel, b = current.fit.beta[j];
  let peak = null;
  for (const d of current.grid) if (d.h && (!peak || Math.abs(d.h[j]) > Math.abs(peak.h[j]))) peak = d;
  const coef = current.fit.dropped.includes(j) ? "not identified (no data where it is nonzero)" : `β̂ = ${b.toFixed(3)}`;
  el.textContent = `Column ${j + 1} of ${current.basis.M}: ${current.basis.describe(j)}; ` +
    `largest in size at x = ${peak ? peak.x.toFixed(2) : "–"}; ${coef}.`;
}
function selectBasis(j) {
  if (S.sel === j) return;
  S.sel = j;
  drawBasis();
}
basisC.svg.on("keydown", event => {
  const M = current.basis.M;
  const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
  if (event.key === "Escape") { selectBasis(null); return; }
  if (d === undefined) return;
  event.preventDefault();
  selectBasis(S.sel === null ? (d > 0 ? 0 : M - 1) : (S.sel + d + M) % M);
});

function drawErrors() {
  const P = PRESETS[S.preset], s2 = P.sigma ** 2;
  const dfs = sweep.map(d => d.df);
  errX.domain([Math.min(1, d3.min(dfs)), Math.max(d3.max(dfs), current.fit.rank, 2)]);
  const finite = sweep.flatMap(d => [d.test, d.loocv]).filter(Number.isFinite);
  errY.domain([0, Math.min(d3.max(finite.concat(current.test)) * 1.05, 4 * s2)]).nice();
  errC.gx.call(d3.axisBottom(errX).ticks(Math.min(15, errX.domain()[1])).tickFormat(d3.format("d")));
  errC.gy.call(d3.axisLeft(errY).ticks(5));
  errSigma.attr("x1", MG.left).attr("x2", W - MG.right).attr("y1", errY(s2)).attr("y2", errY(s2));
  const clip = v => errY(Math.min(v, errY.domain()[1] * 1.5));
  const ln = key => d3.line().defined(d => Number.isFinite(d[key])).x(d => errX(d.df)).y(d => clip(d[key]))(sweep);
  const series = [
    { key: "test", stroke: "#c01c28", width: 2.5, dash: null },
    { key: "loocv", stroke: "#1a5fb4", width: 2, dash: "6 4" },
    { key: "train", stroke: "#555", width: 2, dash: "2 3" },
  ];
  errG.selectAll("path").data(series).join("path").attr("fill", "none")
    .attr("stroke", s => s.stroke).attr("stroke-width", s => s.width).attr("stroke-dasharray", s => s.dash)
    .attr("d", s => ln(s.key));
  errG.selectAll("circle").data(sweep.filter(d => Number.isFinite(d.loocv))).join("circle")
    .attr("r", 3).attr("fill", "#fff").attr("stroke", "#1a5fb4")
    .attr("cx", d => errX(d.df)).attr("cy", d => clip(d.loocv));
  refreshSweepMarker();

  const best = arr => arr.reduce((a, b) => (b.v < a.v ? b : a), { v: Infinity });
  const bt = best(sweep.map(d => ({ v: d.test, df: d.df }))), bl = best(sweep.map(d => ({ v: d.loocv, df: d.df })));
  errC.svg.attr("aria-label",
    `Test error, LOOCV and training MSE against df for the ${METHODS[S.method].label.toLowerCase()} with knots at ` +
    `quantiles, df from ${d3.min(dfs)} to ${d3.max(dfs)}. Test error is smallest at df = ${bt.df} ` +
    `(${fmt3(bt.v)}); LOOCV is smallest at df = ${bl.df} (${fmt3(bl.v)}). σ² = ${s2.toFixed(3)}.`);
}
function refreshSweepMarker() {
  if (!Number.isFinite(current.test)) { errCurrent.attr("d", null); return; }
  const x = errX(current.fit.rank), y = errY(Math.min(current.test, errY.domain()[1] * 1.5));
  errCurrent.attr("d", `M${x},${y - 7} l7,7 l-7,7 l-7,-7 z`).attr("stroke", "#fff");
}

let lastStats = "";
function drawStats(announceNow = true) {
  const m = METHODS[S.method], p = S.params[S.method], f = current.fit;
  const items = [
    [m.label, ""],
    ["df", m.formula(p) + (f.dropped.length ? ` (only ${f.rank} identified)` : "")],
    ["training MSE", fmt3(f.trainMSE)],
    ["LOOCV", fmt3(f.loocv)],
    ["test error", fmt3(current.test)],
    ["σ²", (PRESETS[S.preset].sigma ** 2).toFixed(3)],
  ];
  const html = items.map(([k, v]) => (v === "" ? `<span class="stat"><span class="stat-value">${k}</span></span>`
    : `<span class="stat">${k} = <span class="stat-value">${v}</span></span>`)).join("");
  const el = document.getElementById("stats");
  // While a knot is being dragged, update the numbers without flooding the live region.
  if (!announceNow) el.setAttribute("aria-live", "off");
  if (html !== lastStats) { el.innerHTML = html; lastStats = html; }
  const warn = document.getElementById("warning");
  warn.textContent = f.dropped.length
    ? `${f.dropped.length} coefficient${f.dropped.length > 1 ? "s are" : " is"} not identified: some region has ` +
      `too few points for its basis functions, so the fit is not drawn there. Move the knots apart or increase n.`
    : f.loocv === Infinity ? "Some points have leverage 1 (a region with as many points as coefficients), so LOOCV is infinite." : "";
}
function announce() {
  const el = document.getElementById("stats");
  el.setAttribute("aria-live", "polite");
  lastStats = "";
  drawStats(true);
}

// ---------- controls ----------
const paramSlider = document.getElementById("param-slider");
const paramOut = document.getElementById("param-out");
const paramLabel = document.getElementById("param-label");
const nSlider = document.getElementById("n-slider");
const nOut = document.getElementById("n-out");
const keepDf = document.getElementById("keep-df");
const presetSel = document.getElementById("preset");

function syncControls() {
  const m = METHODS[S.method];
  paramLabel.textContent = m.param;
  paramSlider.min = m.min;
  paramSlider.max = maxParam(S.method);
  S.params[S.method] = Math.min(S.params[S.method], maxParam(S.method));
  paramSlider.value = S.params[S.method];
  paramOut.textContent = `${m.sym} = ${S.params[S.method]}`;
  nSlider.value = S.n; nOut.textContent = `n = ${S.n}`;
  presetSel.value = S.preset;
  document.querySelector(`input[name=method][value=${S.method}]`).checked = true;
  document.querySelector(`input[name=bview][value=${S.bview}]`).checked = true;
  document.getElementById("reknot").disabled = S.method === "poly";
}

// full: data, basis or sweep changed. Otherwise only the fit.
function update(full = true) {
  if (full) recomputeSweep();
  refit();
  drawFit(); drawBasis(); drawErrors();
  announce();
}

document.querySelectorAll("input[name=method]").forEach(r => r.addEventListener("change", () => {
  const oldK = K(), oldDf = current.fit.rank;
  S.method = r.value;
  if (keepDf.checked) {
    const m = METHODS[S.method];
    let bestP = m.min;
    for (let p = m.min; p <= maxParam(S.method); p++) {
      if (Math.abs(m.df(p) - oldDf) < Math.abs(m.df(bestP) - oldDf)) bestP = p;
    }
    S.params[S.method] = bestP;
  }
  S.params[S.method] = Math.min(S.params[S.method], maxParam(S.method));
  if (K() !== oldK || !S.knotsCustom) setKnotsToQuantiles();
  S.sel = null;
  syncControls();
  update();
}));
paramSlider.addEventListener("input", () => {
  S.params[S.method] = +paramSlider.value;
  paramOut.textContent = `${METHODS[S.method].sym} = ${S.params[S.method]}`;
  setKnotsToQuantiles();
  S.sel = null;
  refit(); drawFit(); drawBasis(); refreshSweepMarker(); announce();
});
nSlider.addEventListener("input", () => {
  S.n = +nSlider.value;
  makeData();
  for (const k of Object.keys(METHODS)) S.params[k] = Math.min(S.params[k], maxParam(k));
  if (!S.knotsCustom || S.knots.length !== K()) setKnotsToQuantiles();
  syncControls();
  update();
});
presetSel.addEventListener("change", () => { S.preset = presetSel.value; makeData(); update(); });
document.querySelectorAll("input[name=bview]").forEach(r => r.addEventListener("change", () => {
  S.bview = r.value; drawBasis();
}));
document.getElementById("resample").addEventListener("click", () => {
  S.seed += 1; D = draws(S.seed); makeData();
  if (!S.knotsCustom) setKnotsToQuantiles();
  update();
});
document.getElementById("reknot").addEventListener("click", () => { setKnotsToQuantiles(); update(false); });
document.getElementById("reset").addEventListener("click", () => {
  S.method = "poly";
  S.params = Object.fromEntries(Object.entries(METHODS).map(([k, m]) => [k, m.def]));
  S.preset = "esl"; S.n = 100; S.seed = 1; S.bview = "raw"; S.sel = null;
  keepDf.checked = false;
  D = draws(S.seed); makeData(); setKnotsToQuantiles();
  syncControls();
  update();
});
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

// Exposed for checking the fits against numpy/sklearn.
window.__basis = { S, makeBasis, fitOLS, data: () => data, current: () => current, sweep: () => sweep };

makeData();
setKnotsToQuantiles();
syncControls();
update();
