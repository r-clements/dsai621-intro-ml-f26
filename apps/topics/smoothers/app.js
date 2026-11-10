// Smoothing splines (ISL 7.5, ESL 5.4–5.5). A linear smoother, f̂ = S_λ y.
//
// Minimizes Σ(y_i − g(x_i))² + λ∫g″² (SciPy's make_smoothing_spline convention).
// Reinsch form (ESL 5.17): S_λ = (I + λK), K = Q R⁻¹ Qᵀ with Q (n × n−2) and R (n−2 × n−2)
// tridiagonal (Green and Silverman 1994, §2.1). K is very badly conditioned for random x (close
// points give huge eigenvalues), so instead of eigendecomposing K we eigendecompose the
// well-conditioned S_λ0 = I − λ0 Q (R + λ0 QᵀQ)⁻¹ Qᵀ (Woodbury) for one small λ0, once per
// sample, and recover d_k = (1/ρ_k(λ0) − 1)/λ0. Then for any λ, ρ_k(λ) = 1/(1 + λ d_k),
// df(λ) = Σρ_k, and f̂ = Σ ρ_k (u_kᵀy) u_k. The fitted curve between the x_i is the natural cubic
// spline through f̂ with second derivatives γ = R⁻¹Qᵀf̂, linear outside [x_1, x_n].
//
// LOOCV uses the shortcut mean((y_i − f̂_i)/(1 − S_ii))², exact for the smoothing spline (notes
// §4.1); GCV replaces S_ii by df/n.

const N_MAX = 300;
const GRID_N = 401;           // grid on [0, 1] for curves and the ISE integral
// Reference λ for the eigendecomposition. Error in d_k is about ε/λ0 for small d_k and relative
// ε(1 + λ0 d_k) for large ones; 1e-6 keeps both below ~1e-8 for the λ the df slider reaches.
const LAMBDA0 = 1e-6;

const PRESETS = {
  esl:  { f: x => Math.sin(12 * (x + 0.2)) / (x + 0.2), sigma: 1, y: [-5.5, 6.5], name: "ESL wave" },
  sine: { f: x => Math.sin(2 * Math.PI * x), sigma: 0.3, y: [-2, 2], name: "sine" },
  line: { f: x => 2 * x, sigma: 0.3, y: [-1, 3], name: "straight line" },
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
// x ~ U(0, 1) and z ~ N(0, 1), drawn once per seed; the first n are used, sorted by x.
function draws(seed) {
  const rand = mulberry32(1000 * seed + 621);
  const norm = normalSampler(rand);
  const x = new Float64Array(N_MAX), z = new Float64Array(N_MAX);
  for (let i = 0; i < N_MAX; i++) { x[i] = rand(); z[i] = norm(); }
  return { x, z };
}

// ---------- linear algebra ----------
// Symmetric eigendecomposition: Householder tridiagonalization + implicit QL (JAMA's tred2/tql2).
// A is an array of Float64Array rows (overwritten). Returns eigenvalues ascending and V with
// eigenvector j in column j.
function symEig(A) {
  const n = A.length, V = A, d = new Float64Array(n), e = new Float64Array(n);
  for (let j = 0; j < n; j++) d[j] = V[n - 1][j];
  for (let i = n - 1; i > 0; i--) {
    let scale = 0, h = 0;
    for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
    if (scale === 0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) { d[j] = V[i - 1][j]; V[i][j] = 0; V[j][i] = 0; }
    } else {
      for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
      let f = d[i - 1], g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
      for (let j = 0; j < i; j++) e[j] = 0;
      for (let j = 0; j < i; j++) {
        f = d[j]; V[j][i] = f; g = e[j] + V[j][j] * f;
        for (let k = j + 1; k <= i - 1; k++) { g += V[k][j] * d[k]; e[k] += V[k][j] * f; }
        e[j] = g;
      }
      f = 0;
      for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) e[j] -= hh * d[j];
      for (let j = 0; j < i; j++) {
        f = d[j]; g = e[j];
        for (let k = j; k <= i - 1; k++) V[k][j] -= (f * e[k] + g * d[k]);
        d[j] = V[i - 1][j]; V[i][j] = 0;
      }
    }
    d[i] = h;
  }
  for (let i = 0; i < n - 1; i++) {
    V[n - 1][i] = V[i][i]; V[i][i] = 1;
    const h = d[i + 1];
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
      for (let j = 0; j <= i; j++) {
        let g = 0;
        for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
        for (let k = 0; k <= i; k++) V[k][j] -= g * d[k];
      }
    }
    for (let k = 0; k <= i; k++) V[k][i + 1] = 0;
  }
  for (let j = 0; j < n; j++) { d[j] = V[n - 1][j]; V[n - 1][j] = 0; }
  V[n - 1][n - 1] = 1; e[0] = 0;

  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  let f = 0, tst1 = 0;
  const eps = 2 ** -52;
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m = l;
    while (m < n) { if (Math.abs(e[m]) <= eps * tst1) break; m++; }
    if (m > l) {
      do {
        let g = d[l];
        let p = (d[l + 1] - g) / (2 * e[l]);
        let r = Math.hypot(p, 1);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m];
        let c = 1, c2 = c, c3 = c, s = 0, s2 = 0;
        const el1 = e[l + 1];
        for (let i = m - 1; i >= l; i--) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i]; h = c * p;
          r = Math.hypot(p, e[i]);
          e[i + 1] = s * r; s = e[i] / r; c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for (let k = 0; k < n; k++) {
            h = V[k][i + 1];
            V[k][i + 1] = s * V[k][i] + c * h;
            V[k][i] = c * V[k][i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p; d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] += f; e[l] = 0;
  }
  return { values: d, V };
}

// Thomas algorithm for a symmetric tridiagonal system: diag a (m), off-diagonal b (m − 1).
function solveTridiag(a, b, rhs) {
  const m = a.length, c = new Float64Array(m), x = new Float64Array(m);
  let w = a[0];
  x[0] = rhs[0] / w;
  for (let i = 1; i < m; i++) {
    c[i - 1] = b[i - 1] / w;
    w = a[i] - b[i - 1] * c[i - 1];
    x[i] = (rhs[i] - b[i - 1] * x[i - 1]) / w;
  }
  for (let i = m - 2; i >= 0; i--) x[i] -= c[i] * x[i + 1];
  return x;
}

// ---------- smoothing spline ----------
// Everything here depends on x only, not y.
function splinePrecompute(x) {
  const n = x.length, m = n - 2;
  const h = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = x[i + 1] - x[i];
  // R: diag (h_c + h_{c+1})/3, off-diagonal h_{c+1}/6. Q column c has 1/h_c, −1/h_c − 1/h_{c+1},
  // 1/h_{c+1} in rows c, c+1, c+2.
  const rd = new Float64Array(m), ro = new Float64Array(Math.max(m - 1, 0));
  for (let c = 0; c < m; c++) rd[c] = (h[c] + h[c + 1]) / 3;
  for (let c = 0; c < m - 1; c++) ro[c] = h[c + 1] / 6;
  const q0 = new Float64Array(m), q1 = new Float64Array(m), q2 = new Float64Array(m);
  for (let c = 0; c < m; c++) { q0[c] = 1 / h[c]; q1[c] = -1 / h[c] - 1 / h[c + 1]; q2[c] = 1 / h[c + 1]; }
  const qcol = (c, r) => (r === c ? q0[c] : r === c + 1 ? q1[c] : r === c + 2 ? q2[c] : 0);

  // M = R + λ0 QᵀQ, pentadiagonal; band Cholesky with half-bandwidth 2. L[i][j] stored at L[i][i − j].
  const L = Array.from({ length: m }, () => new Float64Array(3));
  const Mband = (i, j) => {          // i ≥ j, i − j ≤ 2
    let v = 0;
    if (i === j) v += rd[i];
    else if (i - j === 1) v += ro[j];
    for (let r = i; r <= j + 2; r++) v += LAMBDA0 * qcol(i, r) * qcol(j, r);
    return v;
  };
  for (let i = 0; i < m; i++) {
    for (let j = Math.max(0, i - 2); j <= i; j++) {
      let s = Mband(i, j);
      for (let k = Math.max(0, i - 2); k < j; k++) s -= L[i][i - k] * L[j][j - k];
      if (i === j) L[i][0] = Math.sqrt(s);
      else L[i][i - j] = s / L[j][0];
    }
  }
  const cholSolve = rhs => {
    const z = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = rhs[i];
      for (let k = Math.max(0, i - 2); k < i; k++) s -= L[i][i - k] * z[k];
      z[i] = s / L[i][0];
    }
    for (let i = m - 1; i >= 0; i--) {
      let s = z[i];
      for (let k = i + 1; k <= Math.min(m - 1, i + 2); k++) s -= L[k][k - i] * z[k];
      z[i] = s / L[i][0];
    }
    return z;
  };

  // S_λ0, column by column: S e_i = e_i − λ0 Q M⁻¹ (Qᵀ e_i).
  const S0 = Array.from({ length: n }, () => new Float64Array(n));
  const rhs = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    rhs.fill(0);
    for (let c = Math.max(0, i - 2); c <= Math.min(m - 1, i); c++) rhs[c] = qcol(c, i);
    const g = cholSolve(rhs);
    for (let r = 0; r < n; r++) {
      let v = 0;
      for (let c = Math.max(0, r - 2); c <= Math.min(m - 1, r); c++) v += qcol(c, r) * g[c];
      S0[r][i] = (r === i ? 1 : 0) - LAMBDA0 * v;
    }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) {
    const v = 0.5 * (S0[i][j] + S0[j][i]); S0[i][j] = v; S0[j][i] = v;
  }
  const { values, V } = symEig(S0);
  // Largest ρ first (smallest d first).
  const order = Array.from(values.keys()).sort((a, b) => values[b] - values[a]);
  const d = new Float64Array(n), U = [];
  for (let k = 0; k < n; k++) {
    const j = order[k], u = new Float64Array(n);
    for (let r = 0; r < n; r++) u[r] = V[r][j];
    d[k] = Math.max(0, (1 / values[j] - 1) / LAMBDA0);
    U.push(u);
  }
  // Null space of K: replace the first two by an orthonormal basis of {1, x} and set d = 0.
  const xbar = d3.mean(x);
  const u1 = new Float64Array(n).fill(1 / Math.sqrt(n));
  const u2 = Float64Array.from(x, v => v - xbar);
  const nrm = Math.hypot(...u2);
  for (let r = 0; r < n; r++) u2[r] /= nrm;
  U[0] = u1; U[1] = u2; d[0] = 0; d[1] = 0;
  // Sign convention: each eigenvector starts positive at the left end.
  for (let k = 2; k < n; k++) if (U[k][0] < 0) for (let r = 0; r < n; r++) U[k][r] = -U[k][r];
  return { n, x, h, rd, ro, q0, q1, q2, d, U };
}

function splineDf(pre, lam) {
  let s = 0;
  for (let k = 0; k < pre.n; k++) s += 1 / (1 + lam * pre.d[k]);
  return s;
}
function lambdaForDf(pre, df) {
  let lo = -16, hi = 6;            // log10 λ
  for (let it = 0; it < 80; it++) {
    const mid = 0.5 * (lo + hi);
    if (splineDf(pre, 10 ** mid) > df) lo = mid; else hi = mid;
  }
  return 10 ** (0.5 * (lo + hi));
}

function splineFit(pre, y, lam) {
  const n = pre.n, rho = new Float64Array(n), coef = new Float64Array(n), fhat = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    rho[k] = 1 / (1 + lam * pre.d[k]);
    let c = 0;
    const u = pre.U[k];
    for (let r = 0; r < n; r++) c += u[r] * y[r];
    coef[k] = c;
    const w = rho[k] * c;
    if (w !== 0) for (let r = 0; r < n; r++) fhat[r] += w * u[r];
  }
  const Sii = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    if (rho[k] < 1e-14) continue;
    const u = pre.U[k];
    for (let r = 0; r < n; r++) Sii[r] += rho[k] * u[r] * u[r];
  }
  // Second derivatives at the knots: γ = R⁻¹ Qᵀ f̂, zero at both ends.
  const m = n - 2, qtf = new Float64Array(m);
  for (let c = 0; c < m; c++) qtf[c] = pre.q0[c] * fhat[c] + pre.q1[c] * fhat[c + 1] + pre.q2[c] * fhat[c + 2];
  const g = solveTridiag(pre.rd, pre.ro, qtf);
  const gam = new Float64Array(n);
  for (let c = 0; c < m; c++) gam[c + 1] = g[c];
  return { lam, rho, coef, fhat, Sii, gam, df: d3.sum(rho) };
}
function splineRow(pre, rho, i) {   // row i of S_λ = weights ℓ(x_i)
  const n = pre.n, row = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    if (rho[k] < 1e-14) continue;
    const w = rho[k] * pre.U[k][i], u = pre.U[k];
    for (let r = 0; r < n; r++) row[r] += w * u[r];
  }
  return row;
}
function splineEval(pre, fhat, gam, t) {
  const x = pre.x, h = pre.h, n = pre.n;
  if (t <= x[0]) {
    const slope = (fhat[1] - fhat[0]) / h[0] - h[0] * gam[1] / 6;
    return fhat[0] + slope * (t - x[0]);
  }
  if (t >= x[n - 1]) {
    const slope = (fhat[n - 1] - fhat[n - 2]) / h[n - 2] + h[n - 2] * gam[n - 2] / 6;
    return fhat[n - 1] + slope * (t - x[n - 1]);
  }
  const i = Math.min(n - 2, d3.bisectRight(x, t) - 1);
  const hi = h[i], a = t - x[i], b = x[i + 1] - t;
  return (a * fhat[i + 1] + b * fhat[i]) / hi - a * b / 6 * ((1 + a / hi) * gam[i + 1] + (1 + b / hi) * gam[i]);
}

// ---------- criteria ----------
function cvStats(y, fhat, Sii, df) {
  const n = y.length;
  let loo = 0, rss = 0;
  for (let i = 0; i < n; i++) {
    const e = y[i] - fhat[i];
    rss += e * e;
    loo += (e / (1 - Sii[i])) ** 2;
  }
  return { loocv: loo / n, gcv: rss / n / (1 - df / n) ** 2, trainMSE: rss / n };
}
function ise(fGrid, curve) {   // trapezoid rule for ∫_0^1 (f̂ − f)²
  let s = 0;
  for (let g = 0; g < GRID_N; g++) {
    const v = (curve[g] - fGrid[g]) ** 2;
    s += (g === 0 || g === GRID_N - 1 ? 0.5 : 1) * v;
  }
  return s / (GRID_N - 1);
}

// ---------- state ----------
const DEFAULTS = { df: 10, preset: "esl", n: 100, seed: 1, i0: 0 };
const S = { ...DEFAULTS };
let D = draws(S.seed);
let data, pre, sp, grid, fGrid, curves;

function makeData() {
  const idx = Array.from({ length: S.n }, (_, i) => i).sort((a, b) => D.x[a] - D.x[b]);
  const P = PRESETS[S.preset];
  const x = Float64Array.from(idx, i => D.x[i]);
  const y = Float64Array.from(idx, i => P.f(D.x[i]) + P.sigma * D.z[i]);
  data = { x, y, fx: Float64Array.from(x, P.f) };
}
function precomputeX() {         // after x changes (new seed or n)
  pre = splinePrecompute(data.x);
}
function refit() {
  const { y } = data;
  const lam = lambdaForDf(pre, S.df);
  sp = splineFit(pre, y, lam);
  Object.assign(sp, cvStats(y, sp.fhat, sp.Sii, sp.df));
  const P = PRESETS[S.preset];
  grid = d3.range(GRID_N).map(g => g / (GRID_N - 1));
  fGrid = grid.map(P.f);
  curves = {
    spline: grid.map(t => splineEval(pre, sp.fhat, sp.gam, t)),
  };
  sp.ise = ise(fGrid, curves.spline);
}

// ---------- drawing ----------
const W = 740;
const MG = { top: 14, right: 16, bottom: 40, left: 54 };
const BLUE = "#1a5fb4";

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
  if (yl) c.svg.append("text").attr("class", "axis-label")
    .attr("transform", `translate(14,${(c.height - MG.bottom + MG.top) / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").text(yl);
}
const fmt3 = v => (Number.isFinite(v) ? v.toFixed(3) : v === Infinity ? "∞" : "–");
const fmtLam = v => {
  const e = Math.floor(Math.log10(v)), m = v / 10 ** e;
  return `${m.toFixed(2)} × 10${supNum(e)}`;
};
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const supNum = s => [...String(s)].map(ch => SUP[ch]).join("");
const xS = d3.scaleLinear().domain([0, 1]).range([MG.left, W - MG.right]);

// Fit chart
const fitC = makeChart("fit", 350, "");
axisLabels(fitC, "x", "y");
fitC.gx.call(d3.axisBottom(xS).ticks(10));
const fitY = d3.scaleLinear().range([fitC.height - MG.bottom, MG.top]);
const x0Line = fitC.plot.append("line").attr("stroke", "#555").attr("stroke-dasharray", "1 3");
const pointsG = fitC.plot.append("g");
const truePath = fitC.plot.append("path").attr("fill", "none").attr("stroke", "#111")
  .attr("stroke-width", 2).attr("stroke-dasharray", "2 3");
const splinePath = fitC.plot.append("path").attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 3);
const selPoint = fitC.plot.append("circle").attr("r", 6).attr("fill", "none").attr("stroke", "#111").attr("stroke-width", 2);
fitC.svg.attr("tabindex", 0).attr("aria-describedby", "x0-text");

// Kernel chart: the weights in row i0 of S_λ.
const kerC = makeChart("kernel", 220, "");
axisLabels(kerC, "x", "");
kerC.gx.call(d3.axisBottom(xS).ticks(10));
const stripH = kerC.height - MG.top - MG.bottom;
const strips = [0].map(() => {
  const top = MG.top;
  const yS = d3.scaleLinear().range([top + stripH, top + 16]);
  const g = kerC.svg.append("g");
  const gy = kerC.svg.append("g").attr("class", "axis").attr("transform", `translate(${MG.left},0)`);
  const title = kerC.svg.append("text").attr("class", "sub-title").attr("x", MG.left + 6).attr("y", top + 10);
  const zero = g.append("line").attr("stroke", "#888").attr("x1", MG.left).attr("x2", W - MG.right);
  const x0 = g.append("line").attr("stroke", "#555").attr("stroke-dasharray", "1 3");
  const stems = g.append("g");
  return { top, yS, g, gy, title, zero, x0, stems };
});
kerC.svg.append("text").attr("class", "axis-label")
  .attr("transform", `translate(14,${(kerC.height - MG.bottom + MG.top) / 2}) rotate(-90)`)
  .attr("text-anchor", "middle").text("weight ℓᵢ(x₀)");

function drawFit() {
  const { x, y } = data, P = PRESETS[S.preset];
  const ext = d3.extent([...y, ...curves.spline]);
  fitY.domain([Math.min(P.y[0], ext[0]), Math.max(P.y[1], ext[1])]).nice();
  fitC.gy.call(d3.axisLeft(fitY).ticks(6));
  const line = arr => d3.line().x((_, g) => xS(grid[g])).y(v => fitY(v))(arr);
  truePath.attr("d", line(fGrid));
  splinePath.attr("d", line(curves.spline));
  pointsG.selectAll("circle").data(d3.range(S.n)).join("circle")
    .attr("r", 3).attr("fill", "none").attr("stroke", "#777")
    .attr("cx", i => xS(x[i])).attr("cy", i => fitY(y[i]));
  drawX0OnFit();
  fitC.svg.attr("aria-label",
    `Scatterplot of ${S.n} points with the true function (dotted) and the smoothing spline (solid) with ` +
    `${S.df} degrees of freedom. The selected target point is circled. Use the arrow keys to move the target point.`);
}
function drawX0OnFit() {
  const { x, y } = data, i = S.i0, t = x[i];
  const [y0, y1] = fitY.range();
  x0Line.attr("x1", xS(t)).attr("x2", xS(t)).attr("y1", y1).attr("y2", y0);
  selPoint.attr("cx", xS(t)).attr("cy", fitY(y[i]));
}

function kernelRows() {
  return { sRow: splineRow(pre, sp.rho, S.i0) };
}
function drawKernel(rows) {
  const { x } = data, t = x[S.i0];
  const series = [
    { row: rows.sRow, colour: BLUE, title: `Row ${S.i0 + 1} of S(λ), λ = ${fmtLam(sp.lam)}` },
  ];
  let mx = 0, mn = 0;
  for (const s of series) for (const v of s.row) { mx = Math.max(mx, v); mn = Math.min(mn, v); }
  const lo_ = Math.min(mn * 1.15, -0.1 * mx);
  series.forEach((s, k) => {
    const st = strips[k];
    st.yS.domain([lo_, mx * 1.1]);
    st.gy.call(d3.axisLeft(st.yS).ticks(4));
    st.title.text(s.title);
    st.zero.attr("y1", st.yS(0)).attr("y2", st.yS(0));
    st.x0.attr("x1", xS(t)).attr("x2", xS(t)).attr("y1", st.top + 14).attr("y2", st.top + stripH);
    const idx = d3.range(S.n).filter(j => Math.abs(s.row[j]) > 1e-4 * mx);
    const g = st.stems.selectAll("g.stem").data(idx, j => j).join(enter => {
      const e = enter.append("g").attr("class", "stem");
      e.append("line"); e.append("circle"); e.append("rect");
      return e;
    });
    g.select("line").attr("x1", j => xS(x[j])).attr("x2", j => xS(x[j]))
      .attr("y1", st.yS(0)).attr("y2", j => st.yS(s.row[j]))
      .attr("stroke", s.colour).attr("stroke-width", 1.5)
      .attr("stroke-dasharray", j => (s.row[j] < 0 ? "2 2" : null));
    g.select("circle").attr("cx", j => xS(x[j])).attr("cy", j => st.yS(s.row[j])).attr("r", 2.6)
      .attr("fill", s.colour).attr("display", j => (s.row[j] < 0 ? "none" : null));
    g.select("rect").attr("x", j => xS(x[j]) - 3).attr("y", j => st.yS(s.row[j]) - 3)
      .attr("width", 6).attr("height", 6).attr("fill", "#fff").attr("stroke", s.colour).attr("stroke-width", 1.5)
      .attr("display", j => (s.row[j] < 0 ? null : "none"));
  });
}
const NEG_TOL = 1e-3;   // the spline's far-away weights oscillate around zero at the 1e-6 level
function kernelSummary(row) {
  const { x, fx } = data, t = x[S.i0], P = PRESETS[S.preset];
  let fh = 0, ef = 0, mom = 0, ss = 0, mn = Infinity;
  for (let j = 0; j < S.n; j++) {
    fh += row[j] * data.y[j]; ef += row[j] * fx[j]; mom += row[j] * (x[j] - t); ss += row[j] ** 2;
    if (row[j] !== 0) mn = Math.min(mn, row[j]);
  }
  return { fh, sii: row[S.i0], mn, mom, bias: ef - P.f(t), sd: P.sigma * Math.sqrt(ss), neg: row.filter(v => v < -NEG_TOL).length };
}
function drawX0Readout(rows) {
  const { x } = data, t = x[S.i0], P = PRESETS[S.preset];
  const a = kernelSummary(rows.sRow);
  document.getElementById("x0-text").textContent =
    `x₀ = ${t.toFixed(3)}, point ${S.i0 + 1} of ${S.n} (sorted by x); f(x₀) = ${P.f(t).toFixed(3)}. ` +
    `Weights below −0.001: ${a.neg}.`;
  const r = v => (Math.abs(v) < 5e-13 ? "0.000" : fmt3(v));
  const cell = (s) => `<td>${r(s.fh)}</td><td>${r(s.sii)}</td><td>${r(s.mn)}</td><td>${r(s.mom)}</td><td>${r(s.bias)}</td><td>${r(s.sd)}</td>`;
  document.getElementById("x0-table").innerHTML =
    `<table class="stats"><caption>Weights at x₀</caption><thead><tr><th scope="col">Smoother</th>` +
    `<th scope="col">f̂(x₀)</th><th scope="col">own weight Sᵢᵢ</th><th scope="col">smallest weight</th>` +
    `<th scope="col">Σℓᵢ(xᵢ − x₀)</th><th scope="col">bias</th><th scope="col">SD</th></tr></thead><tbody>` +
    `<tr><th scope="row">Smoothing spline</th>${cell(a)}</tr></tbody></table>`;
  document.getElementById("x0-out").textContent = `x₀ = ${t.toFixed(3)} (point ${S.i0 + 1})`;
  const sl = document.getElementById("x0-slider");
  sl.setAttribute("aria-valuetext", `x0 = ${t.toFixed(3)}, point ${S.i0 + 1} of ${S.n}`);
  kerC.svg.attr("aria-label",
    `Equivalent kernel of the smoothing spline at x0 = ${t.toFixed(3)}: ${a.neg} weights below −0.001, smallest ${r(a.mn)}. ` +
    `Positive weights are filled dots above the zero line; negative weights are open squares on dashed stems below it.`);
}
function drawX0() {
  const rows = kernelRows();
  drawX0OnFit(); drawKernel(rows); drawX0Readout(rows);
}

let lastStats = "";
function drawStats() {
  const r = (o, tuning) => `<td>${tuning}</td><td>${o.df.toFixed(2)}</td><td>${fmt3(o.trainMSE)}</td>` +
    `<td>${fmt3(o.loocv)}</td><td>${fmt3(o.gcv)}</td><td>${fmt3(o.ise)}</td>`;
  const html = `<table class="stats"><caption>Fit statistics (target df = ${S.df.toFixed(S.df % 1 ? 2 : 0)})</caption>` +
    `<thead><tr><th scope="col">Smoother</th><th scope="col">tuning</th><th scope="col">df</th>` +
    `<th scope="col">training MSE</th><th scope="col">LOOCV</th><th scope="col">GCV</th>` +
    `<th scope="col">ISE ∫(f̂ − f)²</th></tr></thead><tbody>` +
    `<tr><th scope="row">Smoothing spline</th>${r(sp, `λ = ${fmtLam(sp.lam)}`)}</tr>` +
    `</tbody></table>`;
  const el = document.getElementById("stats");
  if (html !== lastStats) { el.innerHTML = html; lastStats = html; }
}

function drawAll() {
  drawFit(); drawX0(); drawStats();
}

// ---------- controls ----------
const dfSlider = document.getElementById("df-slider"), dfOut = document.getElementById("df-out");
const nSlider = document.getElementById("n-slider"), nOut = document.getElementById("n-out");
const x0Slider = document.getElementById("x0-slider");
const presetSel = document.getElementById("preset");

function syncControls() {
  dfSlider.max = Math.min(30, S.n / 2);
  dfSlider.value = S.df;
  dfOut.textContent = `df = ${Number.isInteger(S.df * 2) ? S.df : S.df.toFixed(2)}`;
  dfSlider.setAttribute("aria-valuetext", dfOut.textContent);
  nSlider.value = S.n; nOut.textContent = `n = ${S.n}`;
  x0Slider.max = S.n; x0Slider.value = S.i0 + 1;
  presetSel.value = S.preset;
}
function update(newX) {
  makeData();
  if (newX) precomputeX();
  S.i0 = Math.min(S.i0, S.n - 1);
  refit();
  syncControls();
  drawAll();
}
function setX0(i) {
  S.i0 = Math.max(0, Math.min(S.n - 1, i));
  x0Slider.value = S.i0 + 1;
  drawX0();
}

dfSlider.addEventListener("input", () => {
  S.df = +dfSlider.value;
  refit(); syncControls(); drawAll();
});
presetSel.addEventListener("change", () => { S.preset = presetSel.value; update(false); });
nSlider.addEventListener("input", () => {
  S.n = +nSlider.value; nOut.textContent = `n = ${S.n}`;
  S.df = Math.min(S.df, S.n / 2);
  update(true);
});
x0Slider.addEventListener("input", () => setX0(+x0Slider.value - 1));
document.getElementById("resample").addEventListener("click", () => {
  S.seed += 1; D = draws(S.seed); update(true);
});
document.getElementById("reset").addEventListener("click", () => {
  Object.assign(S, DEFAULTS);
  D = draws(S.seed); update(true);
});
document.getElementById("gcv").addEventListener("click", () => {
  // Minimize the spline's GCV over λ, using ‖y − f̂‖² = Σ(1 − ρ_k)² (u_kᵀy)².
  const c2 = sp.coef.map(c => c * c), n = S.n;
  let best = null;
  for (let e = -12; e <= 2; e += 0.01) {
    const lam = 10 ** e;
    let rss = 0, df = 0;
    for (let k = 0; k < n; k++) { const r = 1 / (1 + lam * pre.d[k]); df += r; rss += (1 - r) ** 2 * c2[k]; }
    const g = rss / n / (1 - df / n) ** 2;
    if (!best || g < best.g) best = { g, df };
  }
  S.df = Math.max(+dfSlider.min, Math.min(+dfSlider.max, best.df));
  refit(); syncControls(); drawAll();
});

// Click the fit chart: target = the data point nearest in x.
fitC.svg.on("click", event => {
  const [px] = d3.pointer(event, fitC.svg.node());
  const t = xS.invert(px);
  const r = d3.bisectLeft(data.x, t);
  let i = Math.min(S.n - 1, r);
  if (r > 0 && (r >= S.n || t - data.x[r - 1] < data.x[r] - t)) i = r - 1;
  setX0(i);
});
fitC.svg.on("keydown", event => {
  const step = event.shiftKey ? 10 : 1;
  const moves = { ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step };
  if (event.key in moves) setX0(S.i0 + moves[event.key]);
  else if (event.key === "Home") setX0(0);
  else if (event.key === "End") setX0(S.n - 1);
  else return;
  event.preventDefault();
  // Keyboard users get the x0 summary announced through its live region.
});
// Shift+arrow moves a slider by 5 steps.
document.querySelectorAll("input[type=range]").forEach(sl => sl.addEventListener("keydown", event => {
  if (!event.shiftKey) return;
  const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[event.key];
  if (!dir) return;
  event.preventDefault();
  const step = +sl.step, lo_ = +sl.min, hi = +sl.max;
  sl.value = Math.min(hi, Math.max(lo_, +sl.value + 5 * dir * step));
  sl.dispatchEvent(new Event("input"));
}));

// Exposed for checking against SciPy.
window.__smoothers = {
  S, data: () => data, pre: () => pre, sp: () => sp, curves: () => curves, grid: () => grid,
  splineFit, lambdaForDf, splineRow, splineEval,
};

update(true);
