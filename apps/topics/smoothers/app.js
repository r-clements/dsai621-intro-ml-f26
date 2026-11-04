// Smoothers (ISL 7.5–7.6, ESL 5.4–5.5, 6.1–6.2). Two linear smoothers, f̂ = S y:
//
// Smoothing spline. Minimizes Σ(y_i − g(x_i))² + λ∫g″² (SciPy's make_smoothing_spline convention).
// Reinsch form (ESL 5.17): S_λ = (I + λK), K = Q R⁻¹ Qᵀ with Q (n × n−2) and R (n−2 × n−2)
// tridiagonal (Green and Silverman 1994, §2.1). K is very badly conditioned for random x (close
// points give huge eigenvalues), so instead of eigendecomposing K we eigendecompose the
// well-conditioned S_λ0 = I − λ0 Q (R + λ0 QᵀQ)⁻¹ Qᵀ (Woodbury) for one small λ0, once per
// sample, and recover d_k = (1/ρ_k(λ0) − 1)/λ0. Then for any λ, ρ_k(λ) = 1/(1 + λ d_k),
// df(λ) = Σρ_k, and f̂ = Σ ρ_k (u_kᵀy) u_k. The fitted curve between the x_i is the natural cubic
// spline through f̂ with second derivatives γ = R⁻¹Qᵀf̂, linear outside [x_1, x_n].
//
// Loess. At target t: the k = s·n nearest x_i, h = distance to the k-th nearest, tri-cube weights
// (1 − (|x_i − t|/h)³)³, weighted least squares fit of a degree-q polynomial in (x_i − t),
// f̂(t) = intercept (ISL Algorithm 7.1, ESL 6.1). With q = 1 and it=0 this is statsmodels' lowess.
// df = tr(S) = Σ_i ℓ_i(x_i), tabulated for every k so the span can be matched to a target df.
//
// LOOCV uses the shortcut mean((y_i − f̂_i)/(1 − S_ii))², exact for both smoothers (notes §4.1,
// §5.5); GCV replaces S_ii by df/n.

const N_MAX = 300;
const GRID_N = 401;           // grid on [0, 1] for curves and the ISE integral
// Reference λ for the eigendecomposition. Error in d_k is about ε/λ0 for small d_k and relative
// ε(1 + λ0 d_k) for large ones; 1e-6 keeps both below ~1e-8 for the λ the df slider reaches.
const LAMBDA0 = 1e-6;
const N_EIG_SHOW = 30;

const PRESETS = {
  esl:  { f: x => Math.sin(12 * (x + 0.2)) / (x + 0.2), sigma: 1, y: [-5.5, 6.5], name: "ESL wave" },
  sine: { f: x => Math.sin(2 * Math.PI * x), sigma: 0.3, y: [-2, 2], name: "sine" },
  line: { f: x => 2 * x, sigma: 0.3, y: [-1, 3], name: "straight line" },
};
const DEGREE_NAMES = ["local constant", "local linear", "local quadratic"];

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

// Small dense solve (Gaussian elimination with partial pivoting); A is (q+1)×(q+1), copied.
function solveSmall(A, b) {
  const m = b.length, M = A.map(r => r.slice()), x = b.slice();
  for (let c = 0; c < m; c++) {
    let p = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-300) return null;
    [M[c], M[p]] = [M[p], M[c]]; [x[c], x[p]] = [x[p], x[c]];
    for (let r = c + 1; r < m; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k < m; k++) M[r][k] -= f * M[c][k];
      x[r] -= f * x[c];
    }
  }
  for (let c = m - 1; c >= 0; c--) {
    let s = x[c];
    for (let k = c + 1; k < m; k++) s -= M[c][k] * x[k];
    x[c] = s / M[c][c];
  }
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

// ---------- loess ----------
function loessKmin(q) { return q + 3; }
// The k nearest x_i to t (x sorted): a contiguous index range [a, b], and h = the k-th distance.
function knnWindow(x, t, k) {
  const n = x.length;
  let r = d3.bisectLeft(x, t), l = r - 1;
  for (let c = 0; c < k; c++) {
    if (r >= n || (l >= 0 && t - x[l] <= x[r] - t)) l--; else r++;
  }
  const a = l + 1, b = r - 1;
  return { a, b, h: Math.max(t - x[a], x[b] - t) };
}
// Local polynomial of degree q at t with the k nearest points. Returns the window, weights
// ℓ_i(t) for i in [a, b] (zero elsewhere), and the solve for the local coefficients.
function loessAt(x, t, k, q) {
  const { a, b, h } = knnWindow(x, t, k);
  const len = b - a + 1, K = new Float64Array(len), P = q + 1;
  const A = Array.from({ length: P }, () => new Array(P).fill(0));
  for (let i = 0; i < len; i++) {
    const u = (x[a + i] - t) / h, au = Math.abs(u);
    K[i] = au < 1 ? (1 - au * au * au) ** 3 : 0;
    let pw = 1;
    const pows = [];
    for (let j = 0; j <= 2 * q; j++) { pows.push(pw); pw *= u; }
    for (let r = 0; r < P; r++) for (let c = 0; c < P; c++) A[r][c] += K[i] * pows[r + c];
  }
  const e1 = new Array(P).fill(0); e1[0] = 1;
  const cvec = solveSmall(A, e1);
  if (!cvec) return null;
  const ell = new Float64Array(len);
  for (let i = 0; i < len; i++) {
    const u = (x[a + i] - t) / h;
    let s = 0, pw = 1;
    for (let j = 0; j < P; j++) { s += cvec[j] * pw; pw *= u; }
    ell[i] = K[i] * s;
  }
  return { a, b, h, ell, A, K };
}
// df(k) = tr S = Σ_i ℓ_i(x_i). At t = x_i, u_i = 0 and K_i = 1, so ℓ_i(x_i) = (A⁻¹)_00, with A the
// moment matrix m_{r+c}, m_r = Σ_j K_j u_j^r. Closed forms for q ≤ 2; no allocation.
function loessTrace(x, k, q) {
  const n = x.length;
  let tr = 0;
  for (let i = 0; i < n; i++) {
    const t = x[i], { a, b, h } = knnWindow(x, t, k);
    let m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0;
    for (let j = a; j <= b; j++) {
      const u = (x[j] - t) / h, au = Math.abs(u);
      if (au >= 1) continue;
      const w = (1 - au * au * au) ** 3, u2 = u * u;
      m0 += w; m1 += w * u; m2 += w * u2; m3 += w * u2 * u; m4 += w * u2 * u2;
    }
    if (q === 0) tr += 1 / m0;
    else if (q === 1) tr += m2 / (m0 * m2 - m1 * m1);
    else {
      const c00 = m2 * m4 - m3 * m3;
      tr += c00 / (m0 * c00 - m1 * (m1 * m4 - m2 * m3) + m2 * (m1 * m3 - m2 * m2));
    }
  }
  return tr;
}
function loessFit(x, y, k, q) {
  const n = x.length, fhat = new Float64Array(n), Sii = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const L = loessAt(x, x[i], k, q);
    let s = 0;
    for (let j = 0; j < L.ell.length; j++) s += L.ell[j] * y[L.a + j];
    fhat[i] = s;
    Sii[i] = L.ell[i - L.a];
  }
  return { k, q, fhat, Sii, df: d3.sum(Sii) };
}
function loessPredict(x, y, k, q, t) {
  const L = loessAt(x, t, k, q);
  let s = 0;
  for (let j = 0; j < L.ell.length; j++) s += L.ell[j] * y[L.a + j];
  return s;
}
// The local polynomial itself at target t, for drawing over its window.
function loessLocalPoly(x, y, k, q, t) {
  const L = loessAt(x, t, k, q), P = q + 1, rhs = new Array(P).fill(0);
  for (let i = 0; i < L.ell.length; i++) {
    const u = (x[L.a + i] - t) / L.h;
    let pw = 1;
    for (let j = 0; j < P; j++) { rhs[j] += L.K[i] * pw * y[L.a + i]; pw *= u; }
  }
  const beta = solveSmall(L.A, rhs);
  return { L, beta, at: s => { const u = (s - t) / L.h; let v = 0, pw = 1; for (const bj of beta) { v += bj * pw; pw *= u; } return v; } };
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
const DEFAULTS = { df: 10, degree: 1, preset: "esl", n: 100, seed: 1, i0: 0, eigSel: 2 };
const S = { ...DEFAULTS };
let D = draws(S.seed);
let data, pre, dfTabs, sp, lo, grid, fGrid, curves;

function makeData() {
  const idx = Array.from({ length: S.n }, (_, i) => i).sort((a, b) => D.x[a] - D.x[b]);
  const P = PRESETS[S.preset];
  const x = Float64Array.from(idx, i => D.x[i]);
  const y = Float64Array.from(idx, i => P.f(D.x[i]) + P.sigma * D.z[i]);
  data = { x, y, fx: Float64Array.from(x, P.f) };
}
function precomputeX() {         // after x changes (new seed or n)
  pre = splinePrecompute(data.x);
  dfTabs = [new Map(), new Map(), new Map()];
}
function loessDf(q, k) {          // cached tr S for span k/n
  const m = dfTabs[q];
  if (!m.has(k)) m.set(k, loessTrace(data.x, k, q));
  return m.get(k);
}
// df(k) decreases (almost always strictly) as the span grows. Binary-search for the crossing,
// then check a few neighbours in case it is not exactly monotone.
function bestK(q, target) {
  let lo_ = loessKmin(q), hi = S.n;
  while (hi - lo_ > 1) {
    const mid = (lo_ + hi) >> 1;
    if (loessDf(q, mid) >= target) lo_ = mid; else hi = mid;
  }
  let best = lo_;
  for (let k = Math.max(loessKmin(q), lo_ - 3); k <= Math.min(S.n, lo_ + 4); k++) {
    if (Math.abs(loessDf(q, k) - target) < Math.abs(loessDf(q, best) - target)) best = k;
  }
  return best;
}
function refit() {
  const { x, y } = data;
  const lam = lambdaForDf(pre, S.df);
  sp = splineFit(pre, y, lam);
  Object.assign(sp, cvStats(y, sp.fhat, sp.Sii, sp.df));
  const k = bestK(S.degree, S.df);
  lo = loessFit(x, y, k, S.degree);
  Object.assign(lo, cvStats(y, lo.fhat, lo.Sii, lo.df));
  const P = PRESETS[S.preset];
  grid = d3.range(GRID_N).map(g => g / (GRID_N - 1));
  fGrid = grid.map(P.f);
  curves = {
    spline: grid.map(t => splineEval(pre, sp.fhat, sp.gam, t)),
    loess: grid.map(t => loessPredict(x, y, k, S.degree, t)),
  };
  sp.ise = ise(fGrid, curves.spline);
  lo.ise = ise(fGrid, curves.loess);
}

// ---------- drawing ----------
const W = 740;
const MG = { top: 14, right: 16, bottom: 40, left: 54 };
const BLUE = "#1a5fb4", ORANGE = "#c64600";

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
const winRect = fitC.plot.append("rect").attr("fill", ORANGE).attr("fill-opacity", 0.07)
  .attr("stroke", ORANGE).attr("stroke-dasharray", "3 2").attr("stroke-opacity", 0.7);
const x0Line = fitC.plot.append("line").attr("stroke", "#555").attr("stroke-dasharray", "1 3");
const pointsG = fitC.plot.append("g");
const truePath = fitC.plot.append("path").attr("fill", "none").attr("stroke", "#111")
  .attr("stroke-width", 2).attr("stroke-dasharray", "2 3");
const splinePath = fitC.plot.append("path").attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 3);
const loessPath = fitC.plot.append("path").attr("fill", "none").attr("stroke", ORANGE)
  .attr("stroke-width", 3).attr("stroke-dasharray", "9 4");
const localPath = fitC.plot.append("path").attr("fill", "none").attr("stroke", ORANGE).attr("stroke-width", 1.5);
const selPoint = fitC.plot.append("circle").attr("r", 6).attr("fill", "none").attr("stroke", "#111").attr("stroke-width", 2);
fitC.svg.attr("tabindex", 0).attr("aria-describedby", "x0-text");

// Kernel chart: two strips sharing the x axis.
const kerC = makeChart("kernel", 330, "");
axisLabels(kerC, "x", "");
kerC.gx.call(d3.axisBottom(xS).ticks(10));
const STRIP_GAP = 26;
const stripH = (kerC.height - MG.top - MG.bottom - STRIP_GAP) / 2;
const strips = [0, 1].map(s => {
  const top = MG.top + s * (stripH + STRIP_GAP);
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

// Eigenvalue bars and eigenvector
const eigC = makeChart("eig", 220, "");
axisLabels(eigC, "k (eigen-direction, smoothest first)", "ρₖ(λ)");
const eigX = d3.scaleBand().domain(d3.range(1, N_EIG_SHOW + 1)).range([MG.left, W - MG.right]).padding(0.2);
const eigY = d3.scaleLinear().domain([0, 1.05]).range([eigC.height - MG.bottom, MG.top]);
eigC.gx.call(d3.axisBottom(eigX).tickValues([1, 2, 5, 10, 15, 20, 25, 30]));
eigC.gy.call(d3.axisLeft(eigY).ticks(5));
const barsG = eigC.plot.append("g");
const eigDfText = eigC.svg.append("text").attr("class", "sub-title").attr("text-anchor", "end")
  .attr("x", W - MG.right - 6).attr("y", MG.top + 12);
eigC.svg.append("text").attr("class", "axis-label").attr("text-anchor", "middle")
  .attr("x", (eigX(1) + eigX(2) + eigX.bandwidth()) / 2).attr("y", MG.top + 4).attr("dy", "-0.1em")
  .text("1, x: never shrunk");
eigC.svg.attr("tabindex", 0).attr("aria-describedby", "eig-readout");

const evC = makeChart("evec", 190, "");
axisLabels(evC, "xᵢ", "uₖ");
evC.gx.call(d3.axisBottom(xS).ticks(10));
const evY = d3.scaleLinear().range([evC.height - MG.bottom, MG.top]);
const evZero = evC.plot.append("line").attr("stroke", "#bbb").attr("x1", MG.left).attr("x2", W - MG.right);
const evPath = evC.plot.append("path").attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 2);
const evPts = evC.plot.append("g");

function drawFit() {
  const { x, y } = data, P = PRESETS[S.preset];
  const ext = d3.extent([...y, ...curves.spline, ...curves.loess]);
  fitY.domain([Math.min(P.y[0], ext[0]), Math.max(P.y[1], ext[1])]).nice();
  fitC.gy.call(d3.axisLeft(fitY).ticks(6));
  const line = arr => d3.line().x((_, g) => xS(grid[g])).y(v => fitY(v))(arr);
  truePath.attr("d", line(fGrid));
  splinePath.attr("d", line(curves.spline));
  loessPath.attr("d", line(curves.loess));
  pointsG.selectAll("circle").data(d3.range(S.n)).join("circle")
    .attr("r", 3).attr("fill", "none").attr("stroke", "#777")
    .attr("cx", i => xS(x[i])).attr("cy", i => fitY(y[i]));
  drawX0OnFit();
  fitC.svg.attr("aria-label",
    `Scatterplot of ${S.n} points with the true function (dotted), the smoothing spline (solid) and the ` +
    `${DEGREE_NAMES[S.degree]} loess fit (dashed), both with about ${S.df} degrees of freedom. ` +
    `The selected target point is circled and the loess window around it is shaded. Use the arrow keys to move the target point.`);
}
function drawX0OnFit() {
  const { x, y } = data, i = S.i0, t = x[i];
  const lp = loessLocalPoly(x, y, lo.k, S.degree, t);
  const a = Math.max(0, t - lp.L.h), b = Math.min(1, t + lp.L.h);
  const [y0, y1] = fitY.range();
  winRect.attr("x", xS(t - lp.L.h)).attr("width", xS(t + lp.L.h) - xS(t - lp.L.h)).attr("y", y1).attr("height", y0 - y1);
  x0Line.attr("x1", xS(t)).attr("x2", xS(t)).attr("y1", y1).attr("y2", y0);
  const ss = d3.range(61).map(j => a + (b - a) * j / 60);
  localPath.attr("d", d3.line().x(s => xS(s)).y(s => fitY(lp.at(s)))(ss));
  selPoint.attr("cx", xS(t)).attr("cy", fitY(y[i]));
}

function kernelRows() {
  const { x } = data, i = S.i0, n = S.n;
  const sRow = splineRow(pre, sp.rho, i);
  const L = loessAt(x, x[i], lo.k, S.degree);
  const lRow = new Float64Array(n);
  for (let j = 0; j < L.ell.length; j++) lRow[L.a + j] = L.ell[j];
  return { sRow, lRow, L };
}
function drawKernel(rows) {
  const { x } = data, t = x[S.i0];
  const series = [
    { row: rows.sRow, colour: BLUE, title: `Smoothing spline: row ${S.i0 + 1} of S(λ), λ = ${fmtLam(sp.lam)}` },
    { row: rows.lRow, colour: ORANGE, title: `Loess, ${DEGREE_NAMES[S.degree]}, span ${(lo.k / S.n).toFixed(2)} (k = ${lo.k})` },
  ];
  // Shared y scale so the two kernels can be compared.
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
  const a = kernelSummary(rows.sRow), b = kernelSummary(rows.lRow);
  document.getElementById("x0-text").textContent =
    `x₀ = ${t.toFixed(3)}, point ${S.i0 + 1} of ${S.n} (sorted by x); f(x₀) = ${P.f(t).toFixed(3)}. ` +
    `Weights below −0.001: spline ${a.neg}, loess ${b.neg}.`;
  const r = v => (Math.abs(v) < 5e-13 ? "0.000" : fmt3(v));
  const cell = (s) => `<td>${r(s.fh)}</td><td>${r(s.sii)}</td><td>${r(s.mn)}</td><td>${r(s.mom)}</td><td>${r(s.bias)}</td><td>${r(s.sd)}</td>`;
  document.getElementById("x0-table").innerHTML =
    `<table class="stats"><caption>Weights at x₀</caption><thead><tr><th scope="col">Smoother</th>` +
    `<th scope="col">f̂(x₀)</th><th scope="col">own weight Sᵢᵢ</th><th scope="col">smallest weight</th>` +
    `<th scope="col">Σℓᵢ(xᵢ − x₀)</th><th scope="col">bias</th><th scope="col">SD</th></tr></thead><tbody>` +
    `<tr><th scope="row">Smoothing spline</th>${cell(a)}</tr>` +
    `<tr><th scope="row">Loess (${DEGREE_NAMES[S.degree]})</th>${cell(b)}</tr></tbody></table>`;
  document.getElementById("x0-out").textContent = `x₀ = ${t.toFixed(3)} (point ${S.i0 + 1})`;
  const sl = document.getElementById("x0-slider");
  sl.setAttribute("aria-valuetext", `x0 = ${t.toFixed(3)}, point ${S.i0 + 1} of ${S.n}`);
  kerC.svg.attr("aria-label",
    `Equivalent kernels at x0 = ${t.toFixed(3)}. Top strip: smoothing spline weights, ${a.neg} below −0.001, smallest ${r(a.mn)}. ` +
    `Bottom strip: ${DEGREE_NAMES[S.degree]} loess weights on its ${lo.k} nearest points, ${b.neg} below −0.001, smallest ${r(b.mn)}. ` +
    `Positive weights are filled dots above the zero line; negative weights are open squares on dashed stems below it.`);
}
function drawX0() {
  const rows = kernelRows();
  drawX0OnFit(); drawKernel(rows); drawX0Readout(rows);
}

function drawEig() {
  const ks = d3.range(1, Math.min(N_EIG_SHOW, S.n) + 1);
  barsG.selectAll("rect").data(ks).join("rect")
    .attr("x", k => eigX(k)).attr("width", eigX.bandwidth())
    .attr("y", k => eigY(sp.rho[k - 1])).attr("height", k => eigY(0) - eigY(sp.rho[k - 1]))
    .attr("fill", k => (k <= 2 ? "#6b8fc9" : BLUE))
    .attr("stroke", k => (k - 1 === S.eigSel ? "#111" : "none")).attr("stroke-width", 3)
    .style("cursor", "pointer")
    .on("mouseenter click", (_, k) => selectEig(k - 1));
  eigDfText.text(`df = Σρₖ = ${sp.df.toFixed(2)} (over all ${S.n})`);
  const first = sp.rho.slice(0, 8).map(v => v.toFixed(2)).join(", ");
  eigC.svg.attr("aria-label",
    `Bar chart of the first ${ks.length} eigenvalues of the smoother matrix: ${first}, and so on. ` +
    `The first two are exactly 1. Use the Left and Right arrow keys to pick an eigenvector.`);
  drawEvec();
}
function signChanges(u) {
  let c = 0;
  for (let r = 1; r < u.length; r++) if (u[r] * u[r - 1] < 0) c++;
  return c;
}
function drawEvec() {
  const k = S.eigSel, u = pre.U[k], { x } = data;
  const m = d3.max(u, Math.abs) * 1.1;
  evY.domain([-m, m]);
  evC.gy.call(d3.axisLeft(evY).ticks(4));
  evZero.attr("y1", evY(0)).attr("y2", evY(0));
  evPath.attr("d", d3.line().x((_, r) => xS(x[r])).y(v => evY(v))(u));
  evPts.selectAll("circle").data(u).join("circle").attr("r", 2).attr("fill", BLUE)
    .attr("cx", (_, r) => xS(x[r])).attr("cy", v => evY(v));
  barsG.selectAll("rect").attr("stroke", kk => (kk - 1 === k ? "#111" : "none"));
  const what = k === 0 ? "the constant direction" : k === 1 ? "the linear direction (centred x)"
    : `changes sign ${signChanges(u)} times`;
  const txt = `Eigenvector u${supSub(k + 1)}: ${what}; d = ${k < 2 ? "0" : pre.d[k].toPrecision(3)}, ` +
    `ρ(λ) = ${sp.rho[k].toFixed(3)}, so ${(100 * sp.rho[k]).toFixed(1)}% of y's component along it is kept.`;
  document.getElementById("eig-readout").textContent = txt;
  evC.svg.attr("aria-label", `Line plot of eigenvector ${k + 1} against x. It ${k < 2 ? "is " + what : what}.`);
}
const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };
const supSub = v => [...String(v)].map(ch => SUB[ch]).join("");
function selectEig(k) {
  S.eigSel = Math.max(0, Math.min(Math.min(N_EIG_SHOW, S.n) - 1, k));
  drawEvec();
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
    `<tr><th scope="row">Loess (${DEGREE_NAMES[S.degree]})</th>${r(lo, `s = ${(lo.k / S.n).toFixed(2)} (k = ${lo.k})`)}</tr>` +
    `</tbody></table>`;
  const el = document.getElementById("stats");
  if (html !== lastStats) { el.innerHTML = html; lastStats = html; }
  const lo_df = loessDf(S.degree, S.n), hi_df = loessDf(S.degree, loessKmin(S.degree));
  const note = document.getElementById("note");
  note.textContent = Math.abs(lo.df - S.df) > 0.25
    ? `Loess can't hit df = ${S.df} exactly: its span moves in steps of 1/n, and with ${DEGREE_NAMES[S.degree]} fits ` +
      `its df runs from ${lo_df.toFixed(2)} (s = 1) to ${hi_df.toFixed(1)} (k = ${loessKmin(S.degree)}). This is the closest span.`
    : "";
}

function drawAll() {
  drawFit(); drawX0(); drawEig(); drawStats();
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
  document.querySelector(`input[name=degree][value="${S.degree}"]`).checked = true;
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
document.querySelectorAll("input[name=degree]").forEach(r => r.addEventListener("change", () => {
  S.degree = +r.value;
  refit(); drawAll();
}));
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
eigC.svg.on("keydown", event => {
  const moves = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 };
  if (event.key in moves) selectEig(S.eigSel + moves[event.key]);
  else if (event.key === "Home") selectEig(0);
  else if (event.key === "End") selectEig(N_EIG_SHOW - 1);
  else return;
  event.preventDefault();
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

// Exposed for checking against SciPy and statsmodels.
window.__smoothers = {
  S, data: () => data, pre: () => pre, sp: () => sp, lo: () => lo, curves: () => curves, grid: () => grid,
  splineFit, lambdaForDf, splineRow, splineEval, loessFit, loessAt, loessPredict, loessDf, loessTrace,
};

update(true);
