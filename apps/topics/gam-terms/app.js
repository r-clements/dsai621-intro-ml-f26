// Generalized additive models with four kinds of term (ISL 7.7, ESL 9.1; lecture notes §2–§5).
//
//   log price = β0 + f1(age) + β2·size + γ(type) + f4(east, north) + ε
//
// Fitted by backfitting (ESL Algorithm 9.1): β̂0 = ȳ, then cycle over the terms, re-fitting each to its partial
// residual r = y − ȳ − Σ_{k≠j} f̂_k and centering it. The four smoothers:
//   age      cubic P-spline: K_A uniform cubic B-splines, penalty λ‖D₂θ‖² (second differences), so the fit is
//            θ = (BᵀB + λP)⁻¹Bᵀr. This is the penalized-regression-spline approximation to the smoothing spline.
//   size     simple linear regression on centered size (a projection, notes Result 4.2).
//   type     group means (also a projection).
//   location tensor-product cubic P-spline, K_L × K_L basis b_i(east)b_j(north), penalty
//            λ(P_K ⊗ I + I ⊗ P_K): second differences along each axis. Its null space is the bilinear functions.
// Term df = tr(S) − 1 (the constant belongs to β0; notes §5). For a P-spline, tr(S) = tr((A + λP)⁻¹A),
// A = BᵀB, which the Demmler–Reinsch form makes cheap: with A = LLᵀ and L⁻¹PL⁻ᵀ = UΛUᵀ,
// tr(S) = Σ_k 1/(1 + λΛ_k). λ for a target df is found by bisection on that sum.
// Backfitting is run to convergence once per data set and df setting; every update's coefficients are
// stored, and the fitting slider replays them.

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

// ---------- dense Cholesky (lower), in place on a copy ----------
function cholesky(M) {
  const K = M.length, L = M.map(r => Float64Array.from(r));
  for (let j = 0; j < K; j++) {
    let s = L[j][j];
    for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    if (!(s > 0)) throw new Error("not positive definite");
    const d = Math.sqrt(s);
    L[j][j] = d;
    for (let i = j + 1; i < K; i++) {
      let t = L[i][j];
      for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
      L[i][j] = t / d;
    }
    for (let k = j + 1; k < K; k++) L[j][k] = 0;
  }
  return L;
}
function forwardSub(L, b) {
  const K = L.length, x = Float64Array.from(b);
  for (let i = 0; i < K; i++) { let s = x[i]; for (let k = 0; k < i; k++) s -= L[i][k] * x[k]; x[i] = s / L[i][i]; }
  return x;
}
function cholSolve(L, b) {
  const K = L.length, z = forwardSub(L, b);
  for (let i = K - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < K; k++) s -= L[k][i] * z[k]; z[i] = s / L[i][i]; }
  return z;
}

// ---------- uniform cubic B-splines on [a, b] with K basis functions ----------
function bspline(x, a, b, K) {
  const h = (b - a) / (K - 3), t = (x - a) / h;
  const m = Math.min(K - 4, Math.max(0, Math.floor(t)));
  const u = Math.min(1, Math.max(0, t - m)), u2 = u * u, u3 = u2 * u;
  return { m, w: [(1 - u) ** 3 / 6, (3 * u3 - 6 * u2 + 4) / 6, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6, u3 / 6] };
}
function diffPenalty(K) {                   // P = D₂ᵀD₂
  const P = Array.from({ length: K }, () => new Float64Array(K)), d = [1, -2, 1];
  for (let r = 0; r < K - 2; r++) for (let a = 0; a < 3; a++) for (let c = 0; c < 3; c++) P[r + a][r + c] += d[a] * d[c];
  return P;
}

// ---------- P-spline term (1-D or tensor 2-D) ----------
// rows: for each observation, nnz basis indices and values (sparse B). basisAt(…) gives the same for a new point.
function makePSplineTerm(name, n, rowsIdx, rowsVal, nnz, K, P, basisAt) {
  const A = Array.from({ length: K }, () => new Float64Array(K));
  for (let i = 0; i < n; i++) {
    const o = i * nnz;
    for (let a = 0; a < nnz; a++) {
      const ia = rowsIdx[o + a], va = rowsVal[o + a];
      for (let c = 0; c < nnz; c++) A[ia][rowsIdx[o + c]] += va * rowsVal[o + c];
    }
  }
  // Demmler–Reinsch: A + εI = LLᵀ (ε tiny, only so L exists when some basis functions have little data).
  const eps = 1e-9 * d3.mean(A, (r, i) => r[i]);
  const L = cholesky(A.map((r, i) => Float64Array.from(r, (v, j) => v + (i === j ? eps : 0))));
  const X = Array.from({ length: K }, (_, c) => forwardSub(L, Float64Array.from(P, r => r[c])));   // columns of L⁻¹P
  // L⁻¹PL⁻ᵀ = (L⁻¹ (L⁻¹P)ᵀ)ᵀ; X[c] is column c of L⁻¹P, i.e. row c of (L⁻¹P)ᵀ.
  const Y = Array.from({ length: K }, (_, c) => forwardSub(L, Float64Array.from(X, row => row[c])));
  const G = Array.from({ length: K }, (_, i) => Float64Array.from({ length: K }, (_, j) => 0.5 * (Y[i][j] + Y[j][i])));
  const ev = Array.from(symEig(G).values, v => Math.max(0, v));
  const trS = lam => ev.reduce((s, d) => s + 1 / (1 + lam * d), 0);
  const term = {
    name, kind: "pspline", n, K, nnz, rowsIdx, rowsVal, A, P, ev, basisAt,
    trS,
    setDf(df) {
      const target = df + 1;
      let lo = -10, hi = 12;
      for (let it = 0; it < 70; it++) { const mid = (lo + hi) / 2; if (trS(10 ** mid) > target) lo = mid; else hi = mid; }
      this.lam = 10 ** ((lo + hi) / 2);
      this.df = trS(this.lam) - 1;
      this.chol = cholesky(A.map((r, i) => Float64Array.from(r, (v, j) => v + this.lam * P[i][j])));
    },
    raw(theta) {                          // Bθ at the data
      const f = new Float64Array(n);
      for (let i = 0; i < n; i++) { const o = i * nnz; let s = 0; for (let a = 0; a < nnz; a++) s += rowsVal[o + a] * theta[rowsIdx[o + a]]; f[i] = s; }
      return f;
    },
    smooth(r) {
      const btr = new Float64Array(K);
      for (let i = 0; i < n; i++) { const o = i * nnz; for (let a = 0; a < nnz; a++) btr[rowsIdx[o + a]] += rowsVal[o + a] * r[i]; }
      const theta = cholSolve(this.chol, btr);
      return { theta, c: d3.mean(this.raw(theta)) };
    },
    zero() { return { theta: new Float64Array(K), c: 0 }; },
    values(co) { const f = this.raw(co.theta); for (let i = 0; i < n; i++) f[i] -= co.c; return f; },
    at(co, ...pt) { const { idx, val } = basisAt(...pt); let s = 0; for (let a = 0; a < idx.length; a++) s += val[a] * co.theta[idx[a]]; return s - co.c; },
  };
  return term;
}

// ---------- the simulated city ----------
const N_MAX = 2000, SIGMA = 0.15, BETA0 = 12.5, SIZE_BETA = 0.003;
const TYPES = ["Condo", "Townhouse", "Single-family", "Victorian"];
const TYPE_EFF = [-0.2, -0.05, 0.1, 0.15];
const K_A = 20, K_L = 10, AGE_RANGE = [0, 100];
const fAge = a => 0.25 * Math.exp(-a / 8) - 0.004 * a + 0.2 * Math.exp(-(((a - 80) / 10) ** 2));
const bump = (u, v, cu, cv, s) => Math.exp(-((u - cu) ** 2 + (v - cv) ** 2) / (2 * s * s));
const fLoc = (u, v) => 0.45 * bump(u, v, 0.3, 0.7, 0.14) + 0.3 * bump(u, v, 0.75, 0.3, 0.14) - 0.3 * bump(u, v, 0.72, 0.8, 0.1);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

function draws(seed) {
  const rand = mulberry32(1000 * seed + 621), norm = normalSampler(rand);
  const D = { u: [], v: [], za: [], zs: [], t: [], e: [] };
  for (let i = 0; i < N_MAX; i++) {
    D.u.push(rand()); D.v.push(rand()); D.za.push(norm()); D.zs.push(norm());
    D.t.push(Math.min(3, Math.floor(rand() * 4))); D.e.push(norm());
  }
  return D;
}

function makeData(n) {
  const u = Float64Array.from({ length: n }, (_, i) => DR.u[i]), v = Float64Array.from({ length: n }, (_, i) => DR.v[i]);
  // Older houses cluster in the pricey north-west neighbourhood, so age and location are confounded.
  const age = Float64Array.from({ length: n }, (_, i) => clamp(30 + 35 * bump(u[i], v[i], 0.3, 0.7, 0.2) + 20 * DR.za[i], 0, 100));
  const size = Float64Array.from({ length: n }, (_, i) => clamp(150 + 40 * DR.zs[i], 50, 300));
  const type = Int8Array.from({ length: n }, (_, i) => DR.t[i]);
  const parts = [Float64Array.from(age, fAge), Float64Array.from(size, s => SIZE_BETA * s),
    Float64Array.from(type, k => TYPE_EFF[k]), Float64Array.from({ length: n }, (_, i) => fLoc(u[i], v[i]))];
  const means = parts.map(p => d3.mean(p));
  const truth = parts.map((p, j) => Float64Array.from(p, x => x - means[j]));
  const mu = Float64Array.from({ length: n }, (_, i) => BETA0 + parts.reduce((s, p) => s + p[i], 0));
  const y = Float64Array.from(mu, (m, i) => m + SIGMA * DR.e[i]);
  const ybar = d3.mean(y);
  const counts = TYPES.map((_, k) => type.reduce((s, t) => s + (t === k), 0));
  const trueFns = [a => fAge(a) - means[0], s => SIZE_BETA * s - means[1], k => TYPE_EFF[k] - means[2], (a, b) => fLoc(a, b) - means[3]];
  return { n, u, v, age, size, type, truth, trueFns, mu, y, ybar, yc: Float64Array.from(y, x => x - ybar), counts,
    sizeMean: d3.mean(size), uR: d3.extent(u), vR: d3.extent(v) };
}

function makeTerms(dat) {
  const n = dat.n;
  // Age: 1-D P-spline.
  const ageAt = a => { const { m, w } = bspline(clamp(a, ...AGE_RANGE), ...AGE_RANGE, K_A); return { idx: [m, m + 1, m + 2, m + 3], val: w }; };
  const ai = new Int32Array(n * 4), av = new Float64Array(n * 4);
  for (let i = 0; i < n; i++) { const { idx, val } = ageAt(dat.age[i]); for (let a = 0; a < 4; a++) { ai[i * 4 + a] = idx[a]; av[i * 4 + a] = val[a]; } }
  const ageT = makePSplineTerm("age", n, ai, av, 4, K_A, diffPenalty(K_A), ageAt);
  // Location: tensor product over the data's range.
  const locAt = (a, b) => {
    const bu = bspline(clamp(a, ...dat.uR), ...dat.uR, K_L), bv = bspline(clamp(b, ...dat.vR), ...dat.vR, K_L);
    const idx = [], val = [];
    for (let p = 0; p < 4; p++) for (let q = 0; q < 4; q++) { idx.push((bu.m + p) * K_L + bv.m + q); val.push(bu.w[p] * bv.w[q]); }
    return { idx, val };
  };
  const li = new Int32Array(n * 16), lv = new Float64Array(n * 16);
  for (let i = 0; i < n; i++) { const { idx, val } = locAt(dat.u[i], dat.v[i]); for (let a = 0; a < 16; a++) { li[i * 16 + a] = idx[a]; lv[i * 16 + a] = val[a]; } }
  const P1 = diffPenalty(K_L), KK = K_L * K_L, PL = Array.from({ length: KK }, () => new Float64Array(KK));
  for (let i = 0; i < K_L; i++) for (let j = 0; j < K_L; j++) for (let i2 = 0; i2 < K_L; i2++) for (let j2 = 0; j2 < K_L; j2++) {
    let s = 0;
    if (j === j2) s += P1[i][i2];
    if (i === i2) s += P1[j][j2];
    if (s) PL[i * K_L + j][i2 * K_L + j2] = s;
  }
  const locT = makePSplineTerm("location", n, li, lv, 16, KK, PL, locAt);
  // Size: simple regression on centered size.
  const sc = Float64Array.from(dat.size, s => s - dat.sizeMean), sxx = d3.sum(sc, x => x * x);
  const sizeT = {
    name: "size", kind: "linear", df: 1,
    smooth(r) { let s = 0; for (let i = 0; i < n; i++) s += sc[i] * r[i]; return { beta: s / sxx }; },
    zero() { return { beta: 0 }; },
    values(co) { return Float64Array.from(sc, x => co.beta * x); },
    at(co, s) { return co.beta * (s - dat.sizeMean); },
  };
  // Type: group means, centered over the houses.
  const typeT = {
    name: "type", kind: "factor", df: TYPES.length - 1,
    smooth(r) {
      const sum = new Float64Array(4);
      for (let i = 0; i < n; i++) sum[dat.type[i]] += r[i];
      const g = Float64Array.from(sum, (s, k) => (dat.counts[k] ? s / dat.counts[k] : 0));
      const m = d3.sum(g, (x, k) => x * dat.counts[k]) / n;
      return { g: Float64Array.from(g, x => x - m) };
    },
    zero() { return { g: new Float64Array(4) }; },
    values(co) { return Float64Array.from(dat.type, k => co.g[k]); },
    at(co, k) { return co.g[k]; },
  };
  return [ageT, sizeT, typeT, locT];
}

// ---------- backfitting, recording every update ----------
const TOL = 1e-7, MAX_SWEEPS = 300;
function backfit(dat, terms) {
  const n = dat.n, yc = dat.yc;
  let co = terms.map(t => t.zero());
  const F = terms.map(() => new Float64Array(n)), tot = new Float64Array(n);
  const hist = [{ co, j: -1, change: 0, sweep: 0 }];
  let sweeps = 0, converged = false;
  for (let s = 1; s <= MAX_SWEEPS && !converged; s++) {
    let sweepMax = 0;
    for (let j = 0; j < terms.length; j++) {
      const r = new Float64Array(n);
      for (let i = 0; i < n; i++) r[i] = yc[i] - tot[i] + F[j][i];
      const c = terms[j].smooth(r), f = terms[j].values(c);
      let ch = 0;
      for (let i = 0; i < n; i++) { ch = Math.max(ch, Math.abs(f[i] - F[j][i])); tot[i] += f[i] - F[j][i]; }
      F[j] = f;
      co = co.slice(); co[j] = c;
      hist.push({ co, j, change: ch, sweep: s });
      sweepMax = Math.max(sweepMax, ch);
    }
    sweeps = s;
    if (sweepMax < TOL) converged = true;
  }
  return { hist, sweeps, converged };
}

// ---------- state ----------
const DEFAULTS = { n: 500, dfA: 5, dfL: 20, seed: 1 };
const S = { ...DEFAULTS, k: null };     // k = index into the history (null = converged)
let DR = draws(S.seed), dat, terms, run;

function rebuild(newData) {
  if (newData) { dat = makeData(S.n); terms = makeTerms(dat); }
  terms[0].setDf(S.dfA);
  terms[3].setDf(S.dfL);
  run = backfit(dat, terms);
  // Fixed y ranges so replaying the fit doesn't rescale the panels.
  const fin = run.hist[run.hist.length - 1].co;
  const Ffin = terms.map((t, j) => t.values(fin[j]));
  const pr = j => Float64Array.from(dat.yc, (y, i) => y - Ffin.reduce((s, f, k) => s + (k === j ? 0 : f[i]), 0));
  ranges = [0, 1, 2].map(j => {
    const e = d3.extent([...d3.extent(pr(j)), ...d3.extent(dat.truth[j])]);
    return [Math.max(e[0], -1.2), Math.min(e[1], 1.2)];
  });
}
let ranges;

// ---------- drawing ----------
const BLUE = "#1a5fb4", GREY = "#888";
function makeChart(id, W, H, MG, xl, yl) {
  const svg = d3.select(`#${id}`).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img");
  svg.append("defs").append("clipPath").attr("id", `${id}-clip`).append("rect")
    .attr("x", MG.left).attr("y", MG.top).attr("width", W - MG.left - MG.right).attr("height", H - MG.top - MG.bottom);
  const c = {
    svg, W, H, MG,
    gx: svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - MG.bottom})`),
    gy: svg.append("g").attr("class", "axis").attr("transform", `translate(${MG.left},0)`),
    plot: svg.append("g").attr("clip-path", `url(#${id}-clip)`),
    x: d3.scaleLinear().range([MG.left, W - MG.right]),
    y: d3.scaleLinear().range([H - MG.bottom, MG.top]),
    title: svg.append("text").attr("class", "sub-title").attr("x", MG.left).attr("y", 16),
  };
  svg.append("text").attr("class", "axis-label").attr("x", (W + MG.left - MG.right) / 2).attr("y", H - 6).attr("text-anchor", "middle").text(xl);
  svg.append("text").attr("class", "axis-label").attr("text-anchor", "middle")
    .attr("transform", `translate(13,${(H - MG.bottom + MG.top) / 2}) rotate(-90)`).text(yl);
  return c;
}
const MG = { top: 26, right: 12, bottom: 40, left: 46 };
const ageC = makeChart("age-chart", 370, 270, MG, "age (years)", "f₁(age)");
const sizeC = makeChart("size-chart", 370, 270, MG, "size (m²)", "β₂ · size, centered");
const typeC = makeChart("type-chart", 370, 270, MG, "house type", "γ(type)");
const fitC = makeChart("fit-chart", 370, 270, MG, "true mean log price", "fitted log price");
[ageC, sizeC, typeC, fitC].forEach(c => {
  c.pts = c.plot.append("g");
  c.truth = c.plot.append("path").attr("fill", "none").attr("stroke", "#111").attr("stroke-width", 2).attr("stroke-dasharray", "2 3");
  c.est = c.plot.append("path").attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 3);
  c.marks = c.plot.append("g");
});
typeC.x = d3.scaleBand().domain(d3.range(4)).range([MG.left, 370 - MG.right]).padding(0.25);

const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "–");
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const fmtE = v => {
  if (!Number.isFinite(v) || v <= 0) return "–";
  const e = Math.floor(Math.log10(v)), m = v / 10 ** e;
  return `${m.toFixed(2)} × 10${[...String(e)].map(ch => SUP[ch]).join("")}`;
};
const mse = (a, b) => d3.mean(a, (x, i) => (x - b[i]) ** 2);
const TERM_LABEL = ["age curve f₁", "size line β₂·size", "type effects γ", "location map f₄"];
const jitter = Float64Array.from({ length: N_MAX }, (_, i) => mulberry32(i + 7)() - 0.5);

function stateAt(k) {
  const h = run.hist[k], F = terms.map((t, j) => t.values(h.co[j]));
  const tot = new Float64Array(dat.n);
  F.forEach(f => f.forEach((x, i) => { tot[i] += x; }));
  const partial = j => Float64Array.from(dat.yc, (y, i) => y - tot[i] + F[j][i]);
  return { h, F, tot, partial };
}

function drawPanels(st) {
  const { h, F } = st, co = h.co, n = dat.n;
  const ptR = n > 1000 ? 1.3 : n > 400 ? 1.8 : 2.4, ptO = n > 1000 ? 0.25 : 0.4;
  const hl = j => (h.j === j ? " ← this update" : "");
  // Age
  {
    const c = ageC, r = st.partial(0), grid = d3.range(0, 100.01, 0.5);
    c.x.domain(AGE_RANGE); c.y.domain(ranges[0]).nice();
    c.gx.call(d3.axisBottom(c.x).ticks(5)); c.gy.call(d3.axisLeft(c.y).ticks(5));
    c.pts.selectAll("circle").data(d3.range(n)).join("circle").attr("r", ptR).attr("fill", GREY).attr("fill-opacity", ptO)
      .attr("cx", i => c.x(dat.age[i])).attr("cy", i => c.y(r[i]));
    const line = d3.line().x(a => c.x(a[0])).y(a => c.y(a[1]));
    c.truth.attr("d", line(grid.map(a => [a, dat.trueFns[0](a)])));
    c.est.attr("d", line(grid.map(a => [a, terms[0].at(co[0], a)])));
    const m = mse(F[0], dat.truth[0]);
    c.title.text(`Smooth: age (MSE ${fmt(m, 4)})${hl(0)}`).attr("fill", h.j === 0 ? BLUE : "#222");
    const pk = d3.greatest(grid, a => terms[0].at(co[0], a));
    c.svg.attr("aria-label", `Age curve: estimated f₁ (solid) against the true curve (dotted) for ages 0 to 100, over the partial residuals. ` +
      `The estimate is ${fmt(terms[0].at(co[0], 0), 2)} at age 0, ${fmt(terms[0].at(co[0], 50), 2)} at 50 and ${fmt(terms[0].at(co[0], 95), 2)} at 95 ` +
      `(truth ${fmt(dat.trueFns[0](0), 2)}, ${fmt(dat.trueFns[0](50), 2)}, ${fmt(dat.trueFns[0](95), 2)}). Mean squared error ${fmt(m, 4)}.`);
  }
  // Size
  {
    const c = sizeC, r = st.partial(1), ends = [50, 300];
    c.x.domain(ends); c.y.domain(ranges[1]).nice();
    c.gx.call(d3.axisBottom(c.x).ticks(5)); c.gy.call(d3.axisLeft(c.y).ticks(5));
    c.pts.selectAll("circle").data(d3.range(n)).join("circle").attr("r", ptR).attr("fill", GREY).attr("fill-opacity", ptO)
      .attr("cx", i => c.x(dat.size[i])).attr("cy", i => c.y(r[i]));
    const line = d3.line().x(a => c.x(a[0])).y(a => c.y(a[1]));
    c.truth.attr("d", line(ends.map(s => [s, dat.trueFns[1](s)])));
    c.est.attr("d", line(ends.map(s => [s, terms[1].at(co[1], s)])));
    c.title.text(`Linear: size (slope ${fmt(co[1].beta * 100, 3)} per 100 m²)${hl(1)}`).attr("fill", h.j === 1 ? BLUE : "#222");
    c.svg.attr("aria-label", `Size line: estimated slope ${fmt(co[1].beta * 100, 3)} log-price units per 100 square metres (solid) against the true slope ` +
      `${fmt(SIZE_BETA * 100, 3)} (dotted), over the partial residuals.`);
  }
  // Type
  {
    const c = typeC, r = st.partial(2), bw = c.x.bandwidth();
    c.y.domain(ranges[2]).nice();
    c.gx.call(d3.axisBottom(c.x).tickFormat(k => TYPES[k]));
    c.gy.call(d3.axisLeft(c.y).ticks(5));
    c.pts.selectAll("circle").data(d3.range(n)).join("circle").attr("r", ptR).attr("fill", GREY).attr("fill-opacity", ptO)
      .attr("cx", i => c.x(dat.type[i]) + bw / 2 + jitter[i] * bw * 0.8).attr("cy", i => c.y(r[i]));
    c.truth.attr("d", null); c.est.attr("d", null);
    const diam = (x, y) => `M${x},${y - 8} l8,8 l-8,8 l-8,-8 z`;
    c.marks.selectAll("path").data(d3.range(4)).join("path").attr("fill", "#fff").attr("stroke", "#111").attr("stroke-width", 2)
      .attr("d", k => diam(c.x(k) + bw / 2, c.y(dat.trueFns[2](k))));
    c.marks.selectAll("circle").data(d3.range(4)).join("circle").attr("r", 5).attr("fill", BLUE).attr("stroke", "#fff")
      .attr("cx", k => c.x(k) + bw / 2).attr("cy", k => c.y(co[2].g[k]));
    c.marks.selectAll("circle").raise();
    c.title.text(`Categorical: house type${hl(2)}`).attr("fill", h.j === 2 ? BLUE : "#222");
    c.svg.attr("aria-label", "House-type effects: " + TYPES.map((t, k) => `${t} estimated ${fmt(co[2].g[k], 3)}, true ${fmt(dat.trueFns[2](k), 3)}`).join("; ") + ".");
  }
  // Fitted vs true mean
  {
    const c = fitC, yhat = Float64Array.from(st.tot, t => t + dat.ybar);
    const e = d3.extent([...d3.extent(dat.mu), ...d3.extent(dat.y)]);
    c.x.domain(e).nice(); c.y.domain(c.x.domain());
    c.gx.call(d3.axisBottom(c.x).ticks(5)); c.gy.call(d3.axisLeft(c.y).ticks(5));
    c.pts.selectAll("circle").data(d3.range(n)).join("circle").attr("r", ptR).attr("fill", BLUE).attr("fill-opacity", ptO + 0.1)
      .attr("cx", i => c.x(dat.mu[i])).attr("cy", i => c.y(yhat[i]));
    c.truth.attr("d", d3.line()(c.x.domain().map(v => [c.x(v), c.y(v)]))).attr("stroke-dasharray", "6 4").attr("stroke-width", 1.5);
    const rmse = Math.sqrt(mse(yhat, dat.mu));
    c.title.text(`Whole model: RMSE vs truth ${fmt(rmse, 3)}`);
    c.svg.attr("aria-label", `Scatterplot of each house's fitted log price against its true mean log price, with the 45 degree line dashed. ` +
      `Root mean squared error ${fmt(rmse, 3)}; the noise standard deviation is ${SIGMA}.`);
  }
}

// ---------- maps ----------
const G = 40, MAPW = 370, MAPH = 330, MAPM = { top: 26, left: 40, right: 12, bottom: 70 };
const mapSide = Math.min(MAPW - MAPM.left - MAPM.right, MAPH - MAPM.top - MAPM.bottom);
const colour = d3.scaleDiverging(t => d3.interpolateRdBu(1 - t));
const THRESH = [-0.2, -0.1, 0.1, 0.2, 0.3];
function makeMap(id, title) {
  const svg = d3.select(`#${id}`).append("svg").attr("viewBox", `0 0 ${MAPW} ${MAPH}`).attr("role", "img");
  const x0 = MAPM.left + (MAPW - MAPM.left - MAPM.right - mapSide) / 2, y0 = MAPM.top, cs = mapSide / G;
  const m = { svg, x0, y0, cs };
  m.title = svg.append("text").attr("class", "sub-title").attr("x", x0).attr("y", 16).text(title);
  m.cells = svg.append("g");
  m.cont = svg.append("g").attr("fill", "none").attr("stroke", "#111").attr("stroke-width", 1.2);
  m.dots = svg.append("g");
  svg.append("rect").attr("x", x0).attr("y", y0).attr("width", mapSide).attr("height", mapSide).attr("fill", "none").attr("stroke", "#999");
  const sx = d3.scaleLinear().domain([0, 1]).range([x0, x0 + mapSide]), sy = d3.scaleLinear().domain([0, 1]).range([y0 + mapSide, y0]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y0 + mapSide})`).call(d3.axisBottom(sx).ticks(5));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(sy).ticks(5));
  svg.append("text").attr("class", "axis-label").attr("x", x0 + mapSide / 2).attr("y", y0 + mapSide + 30).attr("text-anchor", "middle").text("east");
  svg.append("text").attr("class", "axis-label").attr("text-anchor", "middle").attr("transform", `translate(12,${y0 + mapSide / 2}) rotate(-90)`).text("north");
  m.bar = svg.append("g").attr("transform", `translate(${x0},${y0 + mapSide + 42})`);
  m.sx = sx; m.sy = sy;
  return m;
}
const mapT = makeMap("map-true", "True f₄");
const mapF = makeMap("map-fit", "Estimated f̂₄");
const gridU = d3.range(G).map(c => (c + 0.5) / G);

function drawBar(m) {
  const [lo, , hi] = colour.domain(), w = mapSide, x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
  m.bar.selectAll("rect").data(d3.range(60)).join("rect").attr("x", i => i * w / 60).attr("width", w / 60 + 0.5).attr("height", 8)
    .attr("fill", i => colour(lo + (hi - lo) * (i + 0.5) / 60));
  m.bar.selectAll("g.axis").data([0]).join("g").attr("class", "axis").attr("transform", "translate(0,8)").call(d3.axisBottom(x).ticks(5));
}
function drawMap(m, valueAt, showDots) {
  const vals = new Float64Array(G * G);
  for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) vals[r * G + c] = valueAt(gridU[c], 1 - gridU[r]);
  m.cells.selectAll("rect").data(vals).join("rect")
    .attr("x", (_, i) => m.x0 + (i % G) * m.cs).attr("y", (_, i) => m.y0 + Math.floor(i / G) * m.cs)
    .attr("width", m.cs + 0.4).attr("height", m.cs + 0.4).attr("fill", v => colour(v));
  const path = d3.geoPath(d3.geoIdentity().scale(m.cs).translate([m.x0 + m.cs / 2, m.y0 + m.cs / 2]));
  // Positive levels: regions above t. Negative levels: contour −value above −t, so each line circles a low
  // region rather than everything else.
  const up = d3.contours().size([G, G]).thresholds(THRESH.filter(t => t > 0))(Array.from(vals));
  const dn = d3.contours().size([G, G]).thresholds(THRESH.filter(t => t < 0).map(t => -t))(Array.from(vals, x => -x))
    .map(c => ({ ...c, value: -c.value }));
  const cs = [...up, ...dn];
  m.cont.selectAll("path").data(cs).join("path").attr("d", path).attr("stroke-dasharray", d => (d.value < 0 ? "4 3" : null));
  m.dots.selectAll("circle").data(showDots ? d3.range(dat.n) : []).join("circle").attr("r", dat.n > 1000 ? 0.9 : 1.4)
    .attr("fill", "#111").attr("fill-opacity", 0.55).attr("cx", i => m.sx(dat.u[i])).attr("cy", i => m.sy(dat.v[i]));
  drawBar(m);
  let mx = 0, mn = 0;
  vals.forEach((v, i) => { if (v > vals[mx]) mx = i; if (v < vals[mn]) mn = i; });
  const where = i => `east ${gridU[i % G].toFixed(2)}, north ${(1 - gridU[Math.floor(i / G)]).toFixed(2)}`;
  return { max: vals[mx], maxAt: where(mx), min: vals[mn], minAt: where(mn) };
}
function drawMaps(st) {
  const co = st.h.co[3];
  const hiT = drawMap(mapT, dat.trueFns[3], false);
  const hiF = drawMap(mapF, (a, b) => terms[3].at(co, a, b), true);
  const m = mse(st.F[3], dat.truth[3]);
  mapF.title.text(`Estimated f̂₄ (MSE ${fmt(m, 4)})${st.h.j === 3 ? " ← this update" : ""}`).attr("fill", st.h.j === 3 ? BLUE : "#222");
  mapT.svg.attr("aria-label", `Heat map of the true location effect over the unit square, blue cheaper and red pricier. Highest ${fmt(hiT.max, 2)} at ${hiT.maxAt}; ` +
    `lowest ${fmt(hiT.min, 2)} at ${hiT.minAt}. A second, smaller hot spot lies in the south-east.`);
  mapF.svg.attr("aria-label", `Heat map of the estimated location effect, on the same colour scale, with the ${dat.n} houses as dots. Highest ${fmt(hiF.max, 2)} at ${hiF.maxAt}; ` +
    `lowest ${fmt(hiF.min, 2)} at ${hiF.minAt}. Mean squared error against the truth ${fmt(m, 4)}.`);
}

// ---------- tables ----------
let lastStats = "";
function drawStats(st) {
  const fin = run.hist[run.hist.length - 1].co, Ffin = terms.map((t, j) => t.values(fin[j]));
  const lamTxt = t => (t.kind === "pspline" ? fmtE(t.lam) : "–");
  const rows = terms.map((t, j) => `<tr><th scope="row">${TERM_LABEL[j]}</th><td>${fmt(t.df, 1)}</td><td>${lamTxt(t)}</td>` +
    `<td>${fmt(mse(st.F[j], dat.truth[j]), 4)}</td><td>${fmt(mse(Ffin[j], dat.truth[j]), 4)}</td></tr>`).join("");
  const t1 = `<div class="table-wrap"><table class="stats"><caption>Mean squared error of each term against the truth</caption>` +
    `<thead><tr><th scope="col">Term</th><th scope="col">df</th><th scope="col">λ</th><th scope="col">now</th><th scope="col">converged</th></tr></thead>` +
    `<tbody>${rows}</tbody></table></div>`;
  const co = st.h.co;
  const t2 = `<div class="table-wrap"><table class="stats"><caption>House-type effects and size slope (now)</caption>` +
    `<thead><tr><th scope="col">Effect</th><th scope="col">houses</th><th scope="col">true</th><th scope="col">estimated</th></tr></thead><tbody>` +
    TYPES.map((t, k) => `<tr><th scope="row">${t}</th><td>${dat.counts[k]}</td><td>${fmt(dat.trueFns[2](k))}</td><td>${fmt(co[2].g[k])}</td></tr>`).join("") +
    `<tr><th scope="row">Size, per 100 m²</th><td>${dat.n}</td><td>${fmt(SIZE_BETA * 100)}</td><td>${fmt(co[1].beta * 100)}</td></tr></tbody></table></div>`;
  const html = t1 + t2;
  if (html !== lastStats) { document.getElementById("stats").innerHTML = html; lastStats = html; }
}

// ---------- controls ----------
const $ = id => document.getElementById(id);
const prog = $("progress");
const curK = () => (S.k === null ? run.hist.length - 1 : Math.min(S.k, run.hist.length - 1));
function progressText(k) {
  const h = run.hist[k], last = run.hist.length - 1;
  if (k === 0) return `Start: every term is 0, so each fitted price is ȳ. ${last} updates (${run.sweeps} sweeps of 4) to converge.`;
  const done = k === last ? (run.converged ? " Converged: no fitted value moved more than 10⁻⁷ in the last sweep." : ` Stopped after ${MAX_SWEEPS} sweeps.`) : "";
  return `Update ${k} of ${last} (sweep ${h.sweep}): re-fitted the ${TERM_LABEL[h.j]} to the partial residual; the largest change in a fitted value was ${fmtE(h.change)}.${done}`;
}
function drawAll() {
  const k = curK(), st = stateAt(k);
  prog.max = run.hist.length - 1; prog.value = k;
  const txt = progressText(k);
  $("progress-out").textContent = txt;
  prog.setAttribute("aria-valuetext", `Update ${k} of ${run.hist.length - 1}`);
  drawPanels(st); drawMaps(st); drawStats(st);
  $("back").disabled = k === 0; $("start").disabled = k === 0;
  const atEnd = k === run.hist.length - 1;
  $("next").disabled = atEnd; $("end").disabled = atEnd && !timer;
}
function setK(k) { const last = run.hist.length - 1; S.k = k >= last ? null : Math.max(0, k); drawAll(); }

function syncControls() {
  $("n-slider").value = S.n; $("n-out").textContent = `n = ${S.n}`;
  $("dfa-slider").value = S.dfA; $("dfa-out").textContent = `df = ${S.dfA}`;
  $("dfl-slider").value = S.dfL; $("dfl-out").textContent = `df = ${S.dfL}`;
  [["n-slider", "n-out"], ["dfa-slider", "dfa-out"], ["dfl-slider", "dfl-out"]].forEach(([s, o]) => $(s).setAttribute("aria-valuetext", $(o).textContent));
}
let pending = null;
function schedule(newData) {
  pending = { newData: newData || (pending && pending.newData) };
  syncControls();
  setTimeout(() => {
    if (!pending) return;
    const p = pending; pending = null;
    stopPlay();
    const wasEnd = S.k === null, frac = wasEnd ? 1 : S.k / (run.hist.length - 1);
    rebuild(p.newData);
    S.k = wasEnd ? null : Math.round(frac * (run.hist.length - 1));
    drawAll();
  }, 0);
}
let timer = null;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function stopPlay() {
  if (timer) clearInterval(timer);
  timer = null;
  $("play").textContent = "Play"; $("play").setAttribute("aria-pressed", "false");
  if ($("stats").getAttribute("aria-live") === "off") { $("stats").setAttribute("aria-live", "polite"); lastStats = ""; }
}
$("play").addEventListener("click", () => {
  if (timer) { stopPlay(); drawAll(); return; }
  if (curK() === run.hist.length - 1) S.k = 0;
  if (reduceMotion) { setK(run.hist.length - 1); return; }
  $("play").textContent = "Pause"; $("play").setAttribute("aria-pressed", "true");
  $("stats").setAttribute("aria-live", "off");
  timer = setInterval(() => {
    const k = curK() + 1;
    if (k >= run.hist.length - 1) { stopPlay(); setK(k); return; }
    setK(k);
  }, 450);
  drawAll();
});
prog.addEventListener("input", () => { stopPlay(); setK(+prog.value); });
$("next").addEventListener("click", () => { stopPlay(); setK(curK() + 1); });
$("back").addEventListener("click", () => { stopPlay(); setK(curK() - 1); });
$("start").addEventListener("click", () => { stopPlay(); setK(0); });
$("end").addEventListener("click", () => { stopPlay(); setK(run.hist.length - 1); });
$("n-slider").addEventListener("input", e => { S.n = +e.target.value; schedule(true); });
$("dfa-slider").addEventListener("input", e => { S.dfA = +e.target.value; schedule(false); });
$("dfl-slider").addEventListener("input", e => { S.dfL = +e.target.value; schedule(false); });
$("resample").addEventListener("click", () => { S.seed += 1; DR = draws(S.seed); schedule(true); });
$("reset").addEventListener("click", () => { Object.assign(S, DEFAULTS, { k: null }); DR = draws(S.seed); schedule(true); });
document.querySelectorAll("input[type=range]").forEach(sl => sl.addEventListener("keydown", event => {
  if (!event.shiftKey) return;
  const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[event.key];
  if (!dir) return;
  event.preventDefault();
  sl.value = Math.min(+sl.max, Math.max(+sl.min, +sl.value + 5 * dir * +sl.step));
  sl.dispatchEvent(new Event("input"));
}));

// Exposed for checking against numpy.
window.__gamTerms = { S, dat: () => dat, terms: () => terms, run: () => run, rebuild, stateAt };

syncControls();
rebuild(true);
colour.domain([-0.5, 0, 0.5]);
drawAll();
