// Bagging regression trees on Example B (lecture notes: Bagging and Random Forests, §1–§4).
// f(x) = sin(12(x + 0.2)) / (x + 0.2), x ~ U[0, 1], noise variance 1, n = 100.
// Every tree is an actual fully grown CART regression tree (sklearn's DecisionTreeRegressor()
// with its defaults: split until a node is pure or all its x values are equal, cutpoints halfway
// between neighbouring x values, x <= cut goes left). Bagging averages B such trees, each grown on
// a bootstrap sample. Bias and variance come from R = 200 simulated training sets.

const N = 100;
const SIGMA = 1;
const R = 200;                 // simulated training sets for bias and variance
const R_SHOW = 30;             // fits drawn in the bottom plots
const B_VALUES = [1, 2, 3, 5, 10, 20, 50, 100, 200];
const B_MAX = B_VALUES[B_VALUES.length - 1];
const N_TREES_DRAWN = 20;      // faint bootstrap trees in the top plot
const G = 401;                 // grid on [0, 1] for drawing and for bias/variance
const f = x => Math.sin(12 * (x + 0.2)) / (x + 0.2);
const grid = Float64Array.from({ length: G }, (_, j) => j / (G - 1));
const fGrid = Float64Array.from(grid, f);

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
// Each training set r has its own data stream and bootstrap stream, so any one set can be rebuilt.
const dataSeed = r => 1000003 * (r + 1) + 17;
const bootSeed = r => 7919 * (r + 1) + 104729;

// One training set, sorted by x.
function makeData(r) {
  const rand = mulberry32(dataSeed(r));
  const rnorm = normalSampler(rand);
  const x = Float64Array.from({ length: N }, () => rand());
  const y = Float64Array.from(x, v => f(v) + SIGMA * rnorm());
  const order = d3.range(N).sort((a, b) => x[a] - x[b]);
  return { x: Float64Array.from(order, i => x[i]), y: Float64Array.from(order, i => y[i]) };
}

// ---------- a fully grown 1-D regression tree ----------
// X (sorted ascending) and Y are the n points the tree is grown on, repeats allowed.
// Writes the tree's prediction at every grid point into out.
const MAXN = N;
const S1 = new Float64Array(MAXN + 1), S2 = new Float64Array(MAXN + 1);
function fitTree(X, Y, n, out) {
  S1[0] = 0; S2[0] = 0;
  for (let i = 0; i < n; i++) { S1[i + 1] = S1[i] + Y[i]; S2[i + 1] = S2[i] + Y[i] * Y[i]; }
  const sse = (a, b) => { const s = S1[b] - S1[a]; return S2[b] - S2[a] - s * s / (b - a); };
  const cuts = [], vals = [];          // leaves in left-to-right order: upper cutpoint and mean
  function grow(a, b, upper) {         // node holds points a..b-1
    const m = b - a;
    const mean = (S1[b] - S1[a]) / m;
    let best = -1, bestSse = Infinity;
    // A node is a leaf if it has one point, or its y's are equal (pure), or its x's are equal.
    if (m >= 2 && sse(a, b) / m > 1e-12) {
      for (let k = a + 1; k < b; k++) {
        if (X[k] === X[k - 1]) continue;
        const s = sse(a, k) + sse(k, b);
        if (s < bestSse) { bestSse = s; best = k; }
      }
    }
    if (best < 0) { cuts.push(upper); vals.push(mean); return; }
    const c = (X[best - 1] + X[best]) / 2;
    grow(a, best, c);                  // x <= c
    grow(best, b, upper);              // x > c
  }
  grow(0, n, Infinity);
  let leaf = 0;
  for (let j = 0; j < G; j++) {
    while (grid[j] > cuts[leaf]) leaf++;
    out[j] = vals[leaf];
  }
}

// Bootstrap sample of a sorted training set, kept sorted: draw counts, then expand in x order.
const counts = new Int32Array(N), BX = new Float64Array(N), BY = new Float64Array(N);
function bootstrapTree(D, rand, out) {
  counts.fill(0);
  for (let i = 0; i < N; i++) counts[Math.floor(rand() * N)]++;
  let k = 0;
  for (let i = 0; i < N; i++) for (let c = 0; c < counts[i]; c++) { BX[k] = D.x[i]; BY[k] = D.y[i]; k++; }
  fitTree(BX, BY, N, out);
}

// ---------- simulation ----------
// single[r]: the tree on training set r. bagged[bi][r]: the average of the first B_VALUES[bi]
// bootstrap trees on training set r. The first B trees of the 200 are used for every B, so moving
// the slider adds trees to the same average, as in practice.
const single = [], bagged = B_VALUES.map(() => []);
const tmp = new Float64Array(G), sum = new Float64Array(G);
function simulateSet(r) {
  const D = makeData(r);
  const t = new Float64Array(G);
  fitTree(D.x, D.y, N, t);
  single[r] = t;
  const rand = mulberry32(bootSeed(r));
  sum.fill(0);
  let bi = 0;
  for (let b = 1; b <= B_MAX; b++) {
    bootstrapTree(D, rand, tmp);
    for (let j = 0; j < G; j++) sum[j] += tmp[j];
    if (b === B_VALUES[bi]) { bagged[bi][r] = Float64Array.from(sum, s => s / b); bi++; }
  }
}

// Squared bias and variance averaged over the grid, from R fits. The naive squared gap between
// the mean fit and f also contains Var/R of simulation noise, so subtract it.
function biasVar(fits) {
  let b2 = 0, v = 0;
  const mean = new Float64Array(G);
  for (let j = 0; j < G; j++) {
    let s = 0, ss = 0;
    for (let r = 0; r < R; r++) { const p = fits[r][j]; s += p; ss += p * p; }
    const m = s / R, vj = (ss - R * m * m) / (R - 1);
    mean[j] = m;
    v += vj;
    b2 += (m - fGrid[j]) ** 2 - vj / R;
  }
  return { bias2: Math.max(0, b2 / G), variance: v / G, mean };
}
let statsTree = null;
const statsBag = [];

// ---------- formatting ----------
const fmt3 = d3.format(".3f");
const fmt2 = d3.format(".2f");

// ---------- plots ----------
const Y_DOMAIN = [-5.5, 6.5];
function makePlot(container, W, H, margin, labelTitle) {
  const svg = d3.select(container).append("svg")
    .attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img");
  const id = container.slice(1);
  const title = svg.append("title").attr("id", `${id}-title`);
  const desc = svg.append("desc").attr("id", `${id}-desc`);
  svg.attr("aria-labelledby", `${id}-title ${id}-desc`);
  const x = d3.scaleLinear().domain([0, 1]).range([margin.left, W - margin.right]);
  const y = d3.scaleLinear().domain(Y_DOMAIN).range([H - margin.bottom, margin.top]);
  svg.append("defs").append("clipPath").attr("id", `${id}-clip`)
    .append("rect").attr("x", margin.left).attr("y", margin.top)
    .attr("width", W - margin.left - margin.right).attr("height", H - margin.top - margin.bottom);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(5));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6));
  svg.append("text").attr("class", "axis-label")
    .attr("x", (margin.left + W - margin.right) / 2).attr("y", H - 6)
    .attr("text-anchor", "middle").text("x");
  if (labelTitle) {
    svg.append("text").attr("class", "axis-label")
      .attr("transform", `translate(12,${(margin.top + H - margin.bottom) / 2}) rotate(-90)`)
      .attr("text-anchor", "middle").text(labelTitle);
  }
  const area = svg.append("g").attr("clip-path", `url(#${id}-clip)`);
  // Tree predictions are step functions on the grid; curveStep puts each jump between grid points.
  const step = d3.line().x((d, j) => x(grid[j])).y(d => y(d)).curve(d3.curveStep);
  return { svg, title, desc, x, y, area, step };
}
const smoothF = p => d3.line().x(d => p.x(d)).y(d => p.y(f(d)))(d3.range(0, 1.0001, 0.0025));

// Top: one training set.
const topPlot = makePlot("#one", 720, 380, { top: 12, right: 14, bottom: 40, left: 44 }, "y");
const topPoints = topPlot.area.append("g");
const topTrees = topPlot.area.append("g");
topPlot.area.append("path").attr("fill", "none").attr("stroke", "#111").attr("stroke-width", 2.5)
  .attr("d", smoothF(topPlot));
const topSingle = topPlot.area.append("path").attr("fill", "none").attr("stroke", "#c64600")
  .attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3");
const topBag = topPlot.area.append("path").attr("fill", "none").attr("stroke", "#1a5fb4")
  .attr("stroke-width", 3).attr("stroke-linejoin", "round");

// Bottom: 30 training sets, single tree vs bagged.
function makeManyPlot(sel, ylab) {
  const p = makePlot(sel, 360, 300, { top: 10, right: 10, bottom: 38, left: ylab ? 40 : 30 }, ylab);
  p.fits = p.area.append("g");
  p.area.append("path").attr("fill", "none").attr("stroke", "#111").attr("stroke-width", 2.5)
    .attr("d", smoothF(p));
  p.mean = p.area.append("path").attr("fill", "none").attr("stroke", "#c64600")
    .attr("stroke-width", 2.5).attr("stroke-dasharray", "6 4");
  return p;
}
const leftPlot = makeManyPlot("#left", "fitted value");
const rightPlot = makeManyPlot("#right", null);

// ---------- state and update ----------
const slider = document.getElementById("b-slider");
const statsEl = document.getElementById("stats");
let shown = 0;                 // training set in the top plot
let shownTrees = [];           // its first N_TREES_DRAWN bootstrap trees
let shownData = null;
let liveTimer = null;

function loadShown() {
  shownData = makeData(shown);
  const rand = mulberry32(bootSeed(shown));
  shownTrees = d3.range(N_TREES_DRAWN).map(() => { const o = new Float64Array(G); bootstrapTree(shownData, rand, o); return o; });
}

// Average squared distance from f over the grid, for describing a single curve.
const ise = curve => d3.mean(curve, (v, j) => (v - fGrid[j]) ** 2);
// Number of jumps in a step curve on the grid.
const jumps = curve => { let c = 0; for (let j = 1; j < G; j++) if (Math.abs(curve[j] - curve[j - 1]) > 1e-9) c++; return c; };

function update() {
  const bi = +slider.value, B = B_VALUES[bi];
  const t = statsTree, s = statsBag[bi];

  // Top plot.
  topPoints.selectAll("circle").data(shownData.x).join("circle")
    .attr("cx", d => topPlot.x(d)).attr("cy", (d, i) => topPlot.y(shownData.y[i])).attr("r", 2.8)
    .attr("fill", "none").attr("stroke", "#666").attr("stroke-width", 1.1);
  topTrees.selectAll("path").data(shownTrees.slice(0, Math.min(B, N_TREES_DRAWN))).join("path")
    .attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 1)
    .attr("stroke-opacity", B === 1 ? 0 : 0.3).attr("d", topPlot.step);
  const bagCurve = bagged[bi][shown], singleCurve = single[shown];
  topSingle.attr("d", topPlot.step(singleCurve));
  topBag.attr("d", topPlot.step(bagCurve));
  topPlot.title.text(`Training set ${shown + 1}: one fully grown tree and the average of ${B} bootstrap tree${B === 1 ? "" : "s"}`);
  topPlot.desc.text(
    `x from 0 to 1, y from −5.5 to 6.5. Open circles are the ${N} training points. The thick black curve is the true f, ` +
    `which starts near 3.4, dips to −2.5 near x = 0.19, and then oscillates with shrinking size. ` +
    `The dashed step curve is one fully grown tree fitted to all the data: it has ${jumps(singleCurve)} jumps and passes ` +
    `through every training point, and its average squared distance from f is ${fmt2(ise(singleCurve))}. ` +
    (B === 1 ? "The thick blue step curve is a single tree grown on one bootstrap sample. "
      : `The faint curves are ${Math.min(B, N_TREES_DRAWN)} of the bootstrap trees, and the thick blue curve is the average of all ${B}. `) +
    `The thick blue curve has ${jumps(bagCurve)} jumps, ${B >= 20 ? "many more but much smaller than the single tree's, " : ""}` +
    `and its average squared distance from f is ${fmt2(ise(bagCurve))}.`);

  // Bottom plots.
  const drawMany = (p, fits, mean) => {
    p.fits.selectAll("path").data(fits.slice(0, R_SHOW)).join("path")
      .attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 1)
      .attr("stroke-opacity", 0.45).attr("d", p.step);
    p.mean.attr("d", p.step(mean));
  };
  drawMany(leftPlot, single, t.mean);
  drawMany(rightPlot, bagged[bi], s.mean);
  document.getElementById("right-heading").textContent = `Bagged, B = ${B}, per training set`;
  leftPlot.title.text("Thirty fully grown trees, one per training set");
  leftPlot.desc.text(
    `x from 0 to 1, fitted value from −5.5 to 6.5. ${R_SHOW} thin step curves, each a fully grown tree on its own training set, ` +
    `scatter widely around the true f. The dashed curve, the average of all ${R} trees, lies almost on f. ` +
    `Squared bias ${fmt3(t.bias2)}, variance ${fmt3(t.variance)}.`);
  rightPlot.title.text(`Thirty bagged fits with B = ${B}, one per training set`);
  rightPlot.desc.text(
    `Same axes. ${R_SHOW} thin curves, each the average of ${B} bootstrap trees on its own training set. ` +
    (B >= 20 ? "They are smooth and form a much narrower band than the single trees. "
      : B === 1 ? "Each is a single bootstrap tree, as jagged as the single trees. "
      : "They are smoother and less spread out than the single trees. ") +
    `The dashed average of all ${R} lies almost on f. Squared bias ${fmt3(s.bias2)}, variance ${fmt3(s.variance)}.`);

  // Table.
  const testMse = d => 1 + d.bias2 + d.variance;
  const rows = [
    ["Single fully grown tree", t],
    [`Bagged, B = ${B}`, s],
  ];
  d3.select("#rows").selectAll("tr").data(rows).join("tr")
    .html(([name, d]) => `<th scope="row">${name}</th><td>${fmt3(d.bias2)}</td><td>${fmt3(d.variance)}</td><td>${fmt3(testMse(d))}</td>`);

  // Slider text and stats line (debounced so dragging doesn't flood a screen reader).
  document.getElementById("b-out").textContent = `B = ${B}`;
  slider.setAttribute("aria-valuetext", `B = ${B} tree${B === 1 ? "" : "s"}`);
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => {
    document.getElementById("s-b").textContent = B;
    document.getElementById("s-tb").textContent = fmt3(t.bias2);
    document.getElementById("s-tv").textContent = fmt3(t.variance);
    document.getElementById("s-bb").textContent = fmt3(s.bias2);
    document.getElementById("s-bv").textContent = fmt3(s.variance);
  }, statsEl.dataset.ready ? 300 : 0);
  statsEl.dataset.ready = "1";
}

function updateShownHeading() {
  document.getElementById("one-heading").textContent = `One training set (number ${shown + 1} of ${R})`;
}

slider.addEventListener("input", update);
document.getElementById("next").addEventListener("click", () => {
  shown = (shown + 1) % R;
  loadShown(); updateShownHeading(); update();
});
document.getElementById("reset").addEventListener("click", () => {
  shown = 0; slider.value = 0;
  loadShown(); updateShownHeading(); update();
});

// ---------- run the simulation in chunks, so the page stays responsive ----------
const statusEl = document.getElementById("status");
function run() {
  let r = 0;
  const chunk = () => {
    const t0 = performance.now();
    while (r < R && performance.now() - t0 < 40) simulateSet(r++);
    if (r < R) {
      statusEl.textContent = `Simulating ${R} training sets… ${Math.round(100 * r / R)}%`;
      setTimeout(chunk, 0);
      return;
    }
    statsTree = biasVar(single);
    B_VALUES.forEach((B, bi) => { statsBag[bi] = biasVar(bagged[bi]); });
    statusEl.textContent = "";
    for (const id of ["b-slider", "next", "reset"]) document.getElementById(id).disabled = false;
    loadShown(); updateShownHeading(); update();
  };
  chunk();
}
run();
