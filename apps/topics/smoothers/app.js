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
// §4.1). SE band: Var f̂(t) = σ² Σ_k ρ_k² g_k(t)², where g_k is the natural spline interpolating u_k
// (so f̂(t) = Σ ρ_k (u_kᵀy) g_k(t)), with σ² estimated by RSS/(n − df).
// Bias–variance panel (fixed x, at the training x_i): bias² = (1/n)‖(S − I)f‖² = (1/n)Σ(1 − ρ_k)²(u_kᵀf)²,
// variance = (σ²/n) tr(S²) = (σ²/n)Σρ_k², expected test error = σ² + bias² + variance.
// Natural cubic spline overlay: OLS on ESL's basis (5.4–5.5) with knots 0, the K = df − 2 interior
// quantiles of x, and 1, solved by QR.

const N_MAX = 300;
const GRID_N = 401;           // grid on [0, 1] for curves and the SE band
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

// Second derivatives of the natural spline through values f at the x_i: γ = R⁻¹Qᵀf, zero at the ends.
function splineGamma(pre, f) {
  const n = pre.n, m = n - 2, qtf = new Float64Array(m);
  for (let c = 0; c < m; c++) qtf[c] = pre.q0[c] * f[c] + pre.q1[c] * f[c + 1] + pre.q2[c] * f[c + 2];
  const g = solveTridiag(pre.rd, pre.ro, qtf), gam = new Float64Array(n);
  for (let c = 0; c < m; c++) gam[c + 1] = g[c];
  return gam;
}
// g_k(t) on the grid for every eigenvector u_k; depends on x only.
function eigenCurves(pre, grid) {
  return pre.U.map(u => {
    const gam = splineGamma(pre, u);
    return Float64Array.from(grid, t => splineEval(pre, u, gam, t));
  });
}

// ---------- natural cubic spline (regression spline) ----------
function nsKnots(x, M) {          // M knots: 0, M − 2 interior quantiles of x, 1
  return [0, ...d3.range(1, M - 1).map(j => d3.quantileSorted(x, j / (M - 1))), 1];
}
function nsRow(xi, t) {           // ESL 5.4–5.5: 1, t, d_k(t) − d_{M−2}(t), k = 0..M−3
  const M = xi.length, o = new Float64Array(M), last = xi[M - 1];
  const cube = v => (v > 0 ? v * v * v : 0);
  const dk = k => (cube(t - xi[k]) - cube(t - last)) / (last - xi[k]);
  o[0] = 1;
  if (M > 1) o[1] = t;
  if (M > 2) { const dEnd = dk(M - 2); for (let k = 0; k < M - 2; k++) o[k + 2] = dk(k) - dEnd; }
  return o;
}
// Least squares by modified Gram–Schmidt with one reorthogonalization pass.
function lsQR(rows) {
  const n = rows.length, M = rows[0].length;
  const Q = Array.from({ length: M }, (_, j) => Float64Array.from(rows, r => r[j]));
  const R = Array.from({ length: M }, () => new Float64Array(M));
  for (let j = 0; j < M; j++) {
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < j; i++) {
        let d = 0;
        for (let r = 0; r < n; r++) d += Q[i][r] * Q[j][r];
        R[i][j] += d;
        for (let r = 0; r < n; r++) Q[j][r] -= d * Q[i][r];
      }
    }
    let nrm = 0;
    for (let r = 0; r < n; r++) nrm += Q[j][r] ** 2;
    nrm = Math.sqrt(nrm);
    R[j][j] = nrm;
    for (let r = 0; r < n; r++) Q[j][r] /= nrm;
  }
  return { Q, R, n, M };
}
function projectQ(Q, v) {        // Q Qᵀ v
  const n = v.length, out = new Float64Array(n);
  for (const q of Q) {
    let d = 0;
    for (let r = 0; r < n; r++) d += q[r] * v[r];
    for (let r = 0; r < n; r++) out[r] += d * q[r];
  }
  return out;
}
function nsFit(x, y, fx, M, sigma) {
  const knots = nsKnots(x, M), rows = Array.from(x, t => nsRow(knots, t));
  const { Q, R, n } = lsQR(rows);
  const fhat = projectQ(Q, y), hii = new Float64Array(n);
  for (const q of Q) for (let r = 0; r < n; r++) hii[r] += q[r] * q[r];
  // Coefficients: R β = Qᵀy.
  const qty = Q.map(q => q.reduce((s, v, r) => s + v * y[r], 0)), beta = new Float64Array(M);
  for (let j = M - 1; j >= 0; j--) {
    let s = qty[j];
    for (let k = j + 1; k < M; k++) s -= R[j][k] * beta[k];
    beta[j] = s / R[j][j];
  }
  const Hf = projectQ(Q, fx);
  let bias2 = 0;
  for (let r = 0; r < n; r++) bias2 += (Hf[r] - fx[r]) ** 2;
  bias2 /= n;
  const variance = sigma * sigma * M / n;
  return { M, knots, fhat, hii, beta, df: M, bias2, variance, testErr: sigma * sigma + bias2 + variance,
    curve: t => nsRow(knots, t).reduce((s, v, j) => s + v * beta[j], 0) };
}

// ---------- criteria ----------
function cvStats(y, fhat, Sii) {
  const n = y.length;
  let loo = 0, rss = 0;
  for (let i = 0; i < n; i++) {
    const e = y[i] - fhat[i];
    rss += e * e;
    loo += (e / (1 - Sii[i])) ** 2;
  }
  return { loocv: loo / n, trainMSE: rss / n, rss };
}
// Exact bias², variance and expected test error at the training x_i for a smoothing spline with these ρ.
function biasVar(rho, fcoef, sigma, n) {
  let b = 0, v = 0;
  for (let k = 0; k < rho.length; k++) { b += (1 - rho[k]) ** 2 * fcoef[k] ** 2; v += rho[k] ** 2; }
  const bias2 = b / n, variance = sigma * sigma * v / n;
  return { bias2, variance, testErr: sigma * sigma + bias2 + variance };
}

// ---------- state ----------
const DF_MIN = 2.5, DF_STEP = 0.5;
const DEFAULTS = { df: 10, preset: "esl", n: 100, seed: 1, showNS: false };
const S = { ...DEFAULTS };
let D = draws(S.seed);
let data, pre, sp, ns, grid, fGrid, G, fcoef, sweep, curves;
grid = d3.range(GRID_N).map(g => g / (GRID_N - 1));

function makeData() {
  const idx = Array.from({ length: S.n }, (_, i) => i).sort((a, b) => D.x[a] - D.x[b]);
  const P = PRESETS[S.preset];
  const x = Float64Array.from(idx, i => D.x[i]);
  const y = Float64Array.from(idx, i => P.f(D.x[i]) + P.sigma * D.z[i]);
  data = { x, y, fx: Float64Array.from(x, P.f) };
  fGrid = grid.map(P.f);
}
function precomputeX() {         // after x changes (new seed or n)
  pre = splinePrecompute(data.x);
  G = eigenCurves(pre, grid);
}
const dfMax = () => Math.min(30, S.n / 2);
// Everything in the error-vs-df panel; recomputed when x or y changes, not when df moves.
function computeSweep() {
  const { y, fx } = data, n = S.n, sigma = PRESETS[S.preset].sigma;
  fcoef = pre.U.map(u => u.reduce((s, v, r) => s + v * fx[r], 0));
  sweep = d3.range(DF_MIN, dfMax() + 1e-9, DF_STEP / 2).map(df => {
    const fit = splineFit(pre, y, lambdaForDf(pre, df));
    return { df, ...cvStats(y, fit.fhat, fit.Sii), ...biasVar(fit.rho, fcoef, sigma, n) };
  });
}
function refit() {
  const { x, y, fx } = data, n = S.n, P = PRESETS[S.preset];
  sp = splineFit(pre, y, lambdaForDf(pre, S.df));
  Object.assign(sp, cvStats(y, sp.fhat, sp.Sii), biasVar(sp.rho, fcoef, P.sigma, n));
  sp.sigmaHat = Math.sqrt(sp.rss / (n - sp.df));
  const curve = new Float64Array(GRID_N), se = new Float64Array(GRID_N);
  for (let k = 0; k < n; k++) {
    const r = sp.rho[k];
    if (r < 1e-10) continue;
    const gk = G[k], w = r * sp.coef[k], r2 = r * r;
    for (let g = 0; g < GRID_N; g++) { curve[g] += w * gk[g]; se[g] += r2 * gk[g] * gk[g]; }
  }
  for (let g = 0; g < GRID_N; g++) se[g] = sp.sigmaHat * Math.sqrt(se[g]);
  curves = { spline: curve, se };
  ns = null;
  if (S.showNS) {
    const M = Math.max(2, Math.round(S.df));
    ns = nsFit(x, y, fx, M, P.sigma);
    Object.assign(ns, cvStats(y, ns.fhat, ns.hii));
    curves.ns = grid.map(ns.curve);
  }
}

// ---------- drawing ----------
const W = 740;
const MG = { top: 14, right: 16, bottom: 40, left: 54 };
const BLUE = "#1a5fb4", ORANGE = "#c64600", RED = "#c01c28", GREEN = "#26a269", PURPLE = "#813d9c";

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
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const supNum = s => [...String(s)].map(ch => SUP[ch]).join("");
const fmtLam = v => {
  const e = Math.floor(Math.log10(v)), m = v / 10 ** e;
  return `${m.toFixed(2)} × 10${supNum(e)}`;
};
const fmtDf = v => (Number.isInteger(v * 2) ? String(v) : v.toFixed(2));
const xS = d3.scaleLinear().domain([0, 1]).range([MG.left, W - MG.right]);

// Fit chart
const fitC = makeChart("fit", 350, "");
axisLabels(fitC, "x", "y");
fitC.gx.call(d3.axisBottom(xS).ticks(10));
const fitY = d3.scaleLinear().range([fitC.height - MG.bottom, MG.top]);
const bandPath = fitC.plot.append("path").attr("fill", BLUE).attr("fill-opacity", 0.15);
const bandUp = fitC.plot.append("path").attr("fill", "none").attr("stroke", BLUE).attr("stroke-dasharray", "2 2");
const bandLo = fitC.plot.append("path").attr("fill", "none").attr("stroke", BLUE).attr("stroke-dasharray", "2 2");
const pointsG = fitC.plot.append("g");
const knotsG = fitC.plot.append("g");
const truePath = fitC.plot.append("path").attr("fill", "none").attr("stroke", "#111")
  .attr("stroke-width", 2).attr("stroke-dasharray", "2 3");
const nsPath = fitC.plot.append("path").attr("fill", "none").attr("stroke", ORANGE)
  .attr("stroke-width", 2.5).attr("stroke-dasharray", "9 4");
const splinePath = fitC.plot.append("path").attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 3);

// Error-vs-df chart
const errC = makeChart("errors", 300, "");
axisLabels(errC, "df = tr(S)", "mean squared error");
const errX = d3.scaleLinear().range([MG.left, W - MG.right]);
const errY = d3.scaleLinear().range([errC.height - MG.bottom, MG.top]);
const errSigma = errC.plot.append("line").attr("stroke", "#999").attr("stroke-width", 1.5);
const errSigmaText = errC.svg.append("text").attr("class", "axis-label").attr("text-anchor", "end");
const errCurrent = errC.plot.append("line").attr("stroke", "#111").attr("stroke-dasharray", "1 3");
const errG = errC.plot.append("g");
const errMarks = errC.plot.append("g");
errC.svg.style("cursor", "pointer");

const SERIES = [
  { key: "testErr", label: "expected test error", colour: RED, width: 3, dash: null },
  { key: "bias2", label: "bias²", colour: PURPLE, width: 2, dash: "8 4" },
  { key: "variance", label: "variance", colour: GREEN, width: 2, dash: "8 3 2 3" },
  { key: "trainMSE", label: "training MSE", colour: "#555", width: 2, dash: "2 3" },
  { key: "loocv", label: "LOOCV", colour: BLUE, width: 2, dash: "5 3", dots: true },
];

function drawFit() {
  const { x, y } = data, P = PRESETS[S.preset];
  const lo = curves.spline.map((v, g) => v - 2 * curves.se[g]), hi = curves.spline.map((v, g) => v + 2 * curves.se[g]);
  const ext = d3.extent([...y, ...lo, ...hi, ...(curves.ns || [])]);
  fitY.domain([Math.max(Math.min(P.y[0], ext[0]), P.y[0] - 3 * P.sigma),
               Math.min(Math.max(P.y[1], ext[1]), P.y[1] + 3 * P.sigma)]).nice();
  fitC.gy.call(d3.axisLeft(fitY).ticks(6));
  const line = arr => d3.line().x((_, g) => xS(grid[g])).y(v => fitY(v))(arr);
  bandPath.attr("d", d3.area().x((_, g) => xS(grid[g])).y0((_, g) => fitY(lo[g])).y1(v => fitY(v))(hi));
  bandUp.attr("d", line(hi));
  bandLo.attr("d", line(lo));
  truePath.attr("d", line(fGrid));
  splinePath.attr("d", line(curves.spline));
  nsPath.attr("d", curves.ns ? line(curves.ns) : null);
  const [y0, y1] = fitY.range();
  knotsG.selectAll("line").data(ns ? ns.knots.slice(1, -1) : []).join("line")
    .attr("x1", k => xS(k)).attr("x2", k => xS(k)).attr("y1", y0).attr("y2", y0 - 10)
    .attr("stroke", ORANGE).attr("stroke-width", 2);
  pointsG.selectAll("circle").data(d3.range(S.n)).join("circle")
    .attr("r", 3).attr("fill", "none").attr("stroke", "#777")
    .attr("cx", i => xS(x[i])).attr("cy", i => fitY(y[i]));
  const seAt = t => curves.se[Math.round(t * (GRID_N - 1))];
  fitC.svg.attr("aria-label",
    `Scatterplot of ${S.n} points with the true function (dotted) and the smoothing spline (solid) with ` +
    `${fmtDf(S.df)} degrees of freedom and a shaded ±2 standard-error band. The standard error is ` +
    `${seAt(0).toFixed(2)} at x = 0, ${seAt(0.5).toFixed(2)} at x = 0.5 and ${seAt(1).toFixed(2)} at x = 1.` +
    (ns ? ` A natural cubic spline with ${ns.M} degrees of freedom is overlaid (long dashes), with its ${ns.M - 2} interior knots marked by ticks on the x axis.` : ""));
}

function drawErrors() {
  const sigma2 = PRESETS[S.preset].sigma ** 2;
  errX.domain([DF_MIN, dfMax()]);
  // Cap the y axis so the U-shape is visible; very large values at small df are clipped.
  const yMax = Math.max(sigma2 * 2.2, d3.max(sweep, d => Math.min(d.testErr, sigma2 * 3)));
  errY.domain([0, yMax]).nice();
  errC.gx.call(d3.axisBottom(errX).ticks(10));
  errC.gy.call(d3.axisLeft(errY).ticks(6));
  errSigma.attr("x1", MG.left).attr("x2", W - MG.right).attr("y1", errY(sigma2)).attr("y2", errY(sigma2));
  errSigmaText.attr("x", W - MG.right - 4).attr("y", errY(sigma2) + 14).text("σ² (irreducible)");
  const g = errG.selectAll("g.series").data(SERIES).join(enter => {
    const e = enter.append("g").attr("class", "series");
    e.append("path"); e.append("g").attr("class", "dots");
    return e;
  });
  g.select("path").attr("fill", "none").attr("stroke", s => s.colour).attr("stroke-width", s => s.width)
    .attr("stroke-dasharray", s => s.dash)
    .attr("d", s => d3.line().x(d => errX(d.df)).y(d => errY(Math.min(d[s.key], yMax * 1.5)))(sweep));
  g.select("g.dots").selectAll("circle").data(s => (s.dots ? sweep.filter((_, i) => i % 4 === 0).map(d => ({ d, s })) : []))
    .join("circle").attr("r", 3).attr("fill", "#fff").attr("stroke", o => o.s.colour)
    .attr("cx", o => errX(o.d.df)).attr("cy", o => errY(Math.min(o.d[o.s.key], yMax * 1.5)));
  drawErrorMarker();
  const best = d3.least(sweep, d => d.testErr), bestCV = d3.least(sweep, d => d.loocv);
  errC.svg.attr("aria-label",
    `Line chart of error against df from ${DF_MIN} to ${fmtDf(dfMax())}. Bias squared falls and variance rises as df grows; ` +
    `expected test error is smallest at df ≈ ${fmtDf(best.df)} (${best.testErr.toFixed(3)}), and this sample's LOOCV ` +
    `is smallest at df ≈ ${fmtDf(bestCV.df)}. Training MSE keeps falling. The current df, ${fmtDf(S.df)}, is marked.`);
}
function drawErrorMarker() {
  const [y0, y1] = errY.range();
  errCurrent.attr("x1", errX(S.df)).attr("x2", errX(S.df)).attr("y1", y0).attr("y2", y1);
  const yMax = errY.domain()[1];
  errMarks.selectAll("path").data(SERIES.filter(s => s.key !== "trainMSE")).join("path")
    .attr("d", s => {
      const cx = errX(S.df), cy = errY(Math.min(sp[s.key], yMax * 1.5));
      return `M${cx},${cy - 6} l6,6 l-6,6 l-6,-6 z`;
    })
    .attr("fill", s => s.colour).attr("stroke", "#fff");
}

let lastStats = "";
function drawStats() {
  const row = (name, o, tuning, df) => `<tr><th scope="row">${name}</th><td>${tuning}</td><td>${df}</td>` +
    `<td>${fmt3(o.trainMSE)}</td><td>${fmt3(o.loocv)}</td><td>${fmt3(o.bias2)}</td><td>${fmt3(o.variance)}</td>` +
    `<td>${fmt3(o.testErr)}</td></tr>`;
  const html = `<table class="stats"><caption>Fit statistics (σ² = ${(PRESETS[S.preset].sigma ** 2).toFixed(2)})</caption>` +
    `<thead><tr><th scope="col">Fit</th><th scope="col">tuning</th><th scope="col">df</th>` +
    `<th scope="col">training MSE</th><th scope="col">LOOCV</th><th scope="col">bias²</th>` +
    `<th scope="col">variance</th><th scope="col">expected test error</th></tr></thead><tbody>` +
    row("Smoothing spline", sp, `λ = ${fmtLam(sp.lam)}`, sp.df.toFixed(2)) +
    (ns ? row("Natural cubic spline", ns, `${ns.M - 2} interior knots`, ns.M) : "") +
    `</tbody></table>`;
  const el = document.getElementById("stats");
  if (html !== lastStats) { el.innerHTML = html; lastStats = html; }
  document.getElementById("ns-note").textContent = ns && ns.M !== S.df
    ? `The natural spline needs a whole number of df, so it uses df = ${ns.M}.` : "";
}

function drawAll(full) {
  drawFit();
  if (full) drawErrors(); else drawErrorMarker();
  drawStats();
}

// ---------- controls ----------
const dfSlider = document.getElementById("df-slider"), dfOut = document.getElementById("df-out");
const nSlider = document.getElementById("n-slider"), nOut = document.getElementById("n-out");
const presetSel = document.getElementById("preset");
const nsBox = document.getElementById("show-ns");

function syncControls() {
  dfSlider.max = dfMax();
  dfSlider.value = S.df;
  dfOut.textContent = `df = ${fmtDf(S.df)}`;
  dfSlider.setAttribute("aria-valuetext", dfOut.textContent);
  nSlider.value = S.n; nOut.textContent = `n = ${S.n}`;
  presetSel.value = S.preset;
  nsBox.checked = S.showNS;
}
function update(newX) {
  makeData();
  if (newX) precomputeX();
  S.df = Math.min(S.df, dfMax());
  computeSweep();
  refit();
  syncControls();
  drawAll(true);
}
function setDf(df) {
  S.df = Math.max(DF_MIN, Math.min(dfMax(), Math.round(df / DF_STEP) * DF_STEP));
  refit(); syncControls(); drawAll(false);
}

dfSlider.addEventListener("input", () => setDf(+dfSlider.value));
presetSel.addEventListener("change", () => { S.preset = presetSel.value; update(false); });
nSlider.addEventListener("input", () => { S.n = +nSlider.value; nOut.textContent = `n = ${S.n}`; update(true); });
nsBox.addEventListener("change", () => { S.showNS = nsBox.checked; refit(); drawAll(false); });
document.getElementById("resample").addEventListener("click", () => {
  S.seed += 1; D = draws(S.seed); update(true);
});
document.getElementById("reset").addEventListener("click", () => {
  Object.assign(S, DEFAULTS);
  D = draws(S.seed); update(true);
});
// Clicking the error chart sets df (the df slider does the same from the keyboard).
errC.svg.on("click", event => {
  const [px] = d3.pointer(event, errC.svg.node());
  setDf(errX.invert(px));
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

// Exposed for checking against SciPy and numpy.
window.__smoothers = {
  S, data: () => data, pre: () => pre, sp: () => sp, ns: () => ns, curves: () => curves, grid: () => grid,
  sweep: () => sweep, splineFit, lambdaForDf, splineEval, nsFit, setDf,
};

update(true);
