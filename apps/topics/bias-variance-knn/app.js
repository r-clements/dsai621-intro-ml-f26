// Bias–variance tradeoff for KNN regression (lecture notes §4.3, ESL eq. 7.10).
// f(x) = sin(2πx), σ = 0.3, n = 100 fixed training inputs. All the error numbers are the
// exact expectations from the boxed formula, with the x's held fixed:
//   Err(x0) = σ² + [f(x0) − mean of f over the K neighbours]² + σ²/K
//   E[train error] = mean_i b_i² + σ²(1 − 1/K)
// The 20 fitted curves are actual KNN fits to 20 simulated training sets.

// The 100 fixed training inputs (sorted), as used for the lecture table.
// X_FIXED_BEGIN
// rng = np.random.default_rng(621); after two earlier draws for another simulation
// (standard_normal((4000, 500, 10)), then random((4000, 500, 1))), x = np.sort(rng.random(100)).
const X_FIXED = [
  0.0018507314733824387, 0.015157340123289398, 0.03230512476012304, 0.04534997450809242,
  0.05804869051811101, 0.07021313495971226, 0.07339812715016925, 0.07981969772244035,
  0.08239842474794401, 0.08308614681059612, 0.09221591543180141, 0.09673083440230679,
  0.12396901963687323, 0.16459127214356395, 0.19079671758357042, 0.19650058488580968,
  0.20580836045188988, 0.23560710066647683, 0.24373155963526638, 0.2484764431626859,
  0.2543390669943665, 0.2660220767604299, 0.2678036767979488, 0.28219609059024375,
  0.28328518250793, 0.28784931819292636, 0.2974372861462937, 0.30130178694942844,
  0.32587437189342616, 0.3264806134849806, 0.33335848803571533, 0.3466463009966032,
  0.3678572022599397, 0.3688269944098014, 0.37325567709483265, 0.3965197234465657,
  0.4090096013814505, 0.4113914746384255, 0.42658155898228034, 0.42931243227074944,
  0.43704228061278383, 0.4749623419643012, 0.518060533777227, 0.5202077891956756,
  0.5226081570637578, 0.5399966095697266, 0.5664559810207656, 0.5716827989793655,
  0.575853597616296, 0.5795365975485803, 0.5803627635021686, 0.5861680617755907,
  0.5869378395415205, 0.5963406505042099, 0.5993822317688274, 0.6089412061676024,
  0.619477409091569, 0.6241635994210122, 0.6271478483476962, 0.6467251699824421,
  0.6675391420720551, 0.6751477651854888, 0.6869069961227611, 0.6947367446071977,
  0.722399634557678, 0.7336100002336726, 0.7405425328281064, 0.7442373101211037,
  0.7460764914831162, 0.7494277467391813, 0.7536753684401353, 0.7558572661215011,
  0.7617646078784176, 0.7672199861897588, 0.7831696344927715, 0.8004716329262372,
  0.8027769785265219, 0.821664291473035, 0.8226889492949352, 0.851989536634628,
  0.8577986869404017, 0.8604725131275157, 0.8608199088175874, 0.868979418710175,
  0.8728974099952774, 0.8742088347566236, 0.880046344406884, 0.9104040273670809,
  0.9114072709743581, 0.9249598387808742, 0.9268661159192265, 0.9272650097797943,
  0.9291898565004925, 0.9321414861481421, 0.9361428748488747, 0.9449458607852956,
  0.9553784561105889, 0.9580665663705953, 0.9688789592646687, 0.9907120862553642,
];
// X_FIXED_END

const N = X_FIXED.length;
const SIGMA = 0.3;
const SIGMA2 = SIGMA * SIGMA;
const N_SETS = 20;
const N_TEST = 2001;      // test grid for bias² and test error, as in the lecture table
const N_PLOT = 501;       // grid for drawing the fitted curves
const TABLE_KS = [1, 3, 5, 10, 25, 50, 100];
const f = x => Math.sin(2 * Math.PI * x);

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

// ---------- neighbour orderings ----------
// In 1-D the K nearest neighbours of x0 among sorted x's are a contiguous window, so grow it
// one point at a time from the insertion position, taking the closer side (left on a tie).
// Returns the training indices in order of distance from x0.
function neighbourOrder(x0, xs) {
  const order = new Int32Array(xs.length);
  let hi = d3.bisectLeft(xs, x0);
  let lo = hi - 1;
  for (let k = 0; k < xs.length; k++) {
    if (lo < 0) order[k] = hi++;
    else if (hi >= xs.length) order[k] = lo--;
    else if (x0 - xs[lo] <= xs[hi] - x0) order[k] = lo--;
    else order[k] = hi++;
  }
  return order;
}

// Running means over a neighbour ordering: out[k-1] = mean of values over the k nearest.
function runningMeans(order, values) {
  const out = new Float64Array(order.length);
  let s = 0;
  for (let k = 0; k < order.length; k++) {
    s += values[order[k]];
    out[k] = s / (k + 1);
  }
  return out;
}

const fx = Float64Array.from(X_FIXED, f);

// ---------- exact decomposition for every K (precomputed once) ----------
const bias2 = new Float64Array(N + 1);
const trainErr = new Float64Array(N + 1);
(function precomputeErrors() {
  for (let j = 0; j < N_TEST; j++) {
    const x0 = j / (N_TEST - 1);
    const m = runningMeans(neighbourOrder(x0, X_FIXED), fx);
    const f0 = f(x0);
    for (let K = 1; K <= N; K++) bias2[K] += (f0 - m[K - 1]) ** 2;
  }
  // Training points: the point itself is its own nearest neighbour (distance 0).
  for (let i = 0; i < N; i++) {
    const m = runningMeans(neighbourOrder(X_FIXED[i], X_FIXED), fx);
    for (let K = 1; K <= N; K++) trainErr[K] += (fx[i] - m[K - 1]) ** 2;
  }
  for (let K = 1; K <= N; K++) {
    bias2[K] /= N_TEST;
    trainErr[K] = trainErr[K] / N + SIGMA2 * (1 - 1 / K);
  }
})();
const variance = K => SIGMA2 / K;
const testErr = K => SIGMA2 + bias2[K] + variance(K);

// ---------- curves on the plotting grid ----------
const plotX = d3.range(N_PLOT).map(j => j / (N_PLOT - 1));
const plotOrders = plotX.map(x0 => neighbourOrder(x0, X_FIXED));
const meanFitByK = plotOrders.map(o => runningMeans(o, fx));   // E[f̂_K(x0)]

let seed = 1;
let ySets = [];       // N_SETS arrays of length N
let fitsByK = [];     // [set][plot point] -> running means over K
function simulate() {
  const rnorm = normalSampler(mulberry32(seed));
  ySets = d3.range(N_SETS).map(() => Float64Array.from(fx, v => v + SIGMA * rnorm()));
  fitsByK = ySets.map(y => plotOrders.map(o => runningMeans(o, y)));
}

// ---------- formatting ----------
const fmt3 = d3.format(".3f");
const fmtDf = df => (Number.isInteger(df) ? String(df) : d3.format(".1f")(df));

// ---------- main panel: fits ----------
const W = 720, H = 360;
const m1 = { top: 12, right: 14, bottom: 42, left: 46 };
const xS = d3.scaleLinear().domain([0, 1]).range([m1.left, W - m1.right]);
const yS = d3.scaleLinear().domain([-2, 2]).range([H - m1.bottom, m1.top]);

const fitsSvg = d3.select("#fits").append("svg")
  .attr("viewBox", `0 0 ${W} ${H}`)
  .attr("role", "img");
const fitsTitle = fitsSvg.append("title").attr("id", "fits-title");
const fitsDesc = fitsSvg.append("desc").attr("id", "fits-desc");
fitsSvg.attr("aria-labelledby", "fits-title fits-desc");

fitsSvg.append("defs").append("clipPath").attr("id", "plot-clip")
  .append("rect")
  .attr("x", m1.left).attr("y", m1.top)
  .attr("width", W - m1.left - m1.right).attr("height", H - m1.top - m1.bottom);

fitsSvg.append("g").attr("class", "axis")
  .attr("transform", `translate(0,${H - m1.bottom})`)
  .call(d3.axisBottom(xS).ticks(5));
fitsSvg.append("g").attr("class", "axis")
  .attr("transform", `translate(${m1.left},0)`)
  .call(d3.axisLeft(yS).ticks(5));
fitsSvg.append("text").attr("class", "axis-label")
  .attr("x", (m1.left + W - m1.right) / 2).attr("y", H - 8)
  .attr("text-anchor", "middle").text("x");
fitsSvg.append("text").attr("class", "axis-label")
  .attr("transform", `translate(14,${(m1.top + H - m1.bottom) / 2}) rotate(-90)`)
  .attr("text-anchor", "middle").text("y");

const plotArea = fitsSvg.append("g").attr("clip-path", "url(#plot-clip)");
const pointLayer = plotArea.append("g");
const fitLayer = plotArea.append("g");
// KNN fits are step functions; curveStep puts each jump halfway between grid points.
const stepLine = d3.line().x((d, j) => xS(plotX[j])).y(d => yS(d)).curve(d3.curveStep);
const smoothLine = d3.line().x(d => xS(d)).y(d => yS(f(d)));

plotArea.append("path")
  .attr("fill", "none").attr("stroke", "#111").attr("stroke-width", 3)
  .attr("d", smoothLine(plotX));
const meanPath = plotArea.append("path")
  .attr("fill", "none").attr("stroke", "#c64600").attr("stroke-width", 2.5)
  .attr("stroke-dasharray", "6 4");

// ---------- bars: decomposition ----------
const BW = 360, BH = 210;
const mb = { top: 10, right: 52, bottom: 34, left: 92 };
const bars = [
  { key: "bias", label: "bias²", fill: "#c64600", pattern: null },
  { key: "var", label: "variance", fill: "#1a5fb4", pattern: "hatch" },
  { key: "noise", label: "σ² (noise)", fill: "#888", pattern: "dots" },
  { key: "test", label: "test error", fill: "#c01c28", pattern: null, outline: true },
];
const barsSvg = d3.select("#bars").append("svg")
  .attr("viewBox", `0 0 ${BW} ${BH}`).attr("role", "img");
const barsTitle = barsSvg.append("title").attr("id", "bars-title");
const barsDesc = barsSvg.append("desc").attr("id", "bars-desc");
barsSvg.attr("aria-labelledby", "bars-title bars-desc");

const defs = barsSvg.append("defs");
defs.append("pattern").attr("id", "hatch").attr("patternUnits", "userSpaceOnUse")
  .attr("width", 6).attr("height", 6).attr("patternTransform", "rotate(45)")
  .call(p => {
    p.append("rect").attr("width", 6).attr("height", 6).attr("fill", "#1a5fb4");
    p.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 6)
      .attr("stroke", "#fff").attr("stroke-width", 2.5);
  });
defs.append("pattern").attr("id", "dots").attr("patternUnits", "userSpaceOnUse")
  .attr("width", 5).attr("height", 5)
  .call(p => {
    p.append("rect").attr("width", 5).attr("height", 5).attr("fill", "#888");
    p.append("circle").attr("cx", 2.5).attr("cy", 2.5).attr("r", 1.1).attr("fill", "#fff");
  });

// Fixed scale so bar lengths are comparable as K moves (max test error is at K = 100).
const barX = d3.scaleLinear().domain([0, 0.65]).range([mb.left, BW - mb.right]);
const barY = d3.scaleBand().domain(bars.map(b => b.key)).range([mb.top, BH - mb.bottom]).padding(0.25);
barsSvg.append("g").attr("class", "axis")
  .attr("transform", `translate(0,${BH - mb.bottom})`)
  .call(d3.axisBottom(barX).ticks(5));
barsSvg.append("line")
  .attr("x1", barX(0)).attr("x2", barX(0)).attr("y1", mb.top).attr("y2", BH - mb.bottom)
  .attr("stroke", "#333");
const barG = barsSvg.append("g").selectAll("g").data(bars).join("g")
  .attr("transform", b => `translate(0,${barY(b.key)})`);
barG.append("text").attr("class", "bar-label")
  .attr("x", mb.left - 6).attr("y", barY.bandwidth() / 2).attr("dy", "0.35em")
  .attr("text-anchor", "end")
  .style("font-weight", b => (b.outline ? 700 : 400))
  .text(b => b.label);
const barRects = barG.append("rect")
  .attr("x", barX(0)).attr("height", barY.bandwidth())
  .attr("fill", b => (b.pattern ? `url(#${b.pattern})` : b.fill))
  .attr("stroke", b => (b.outline ? "#111" : "none")).attr("stroke-width", 1.5);
const barVals = barG.append("text").attr("class", "bar-value")
  .attr("y", barY.bandwidth() / 2).attr("dy", "0.35em");
// Divider between the three components and their sum.
barsSvg.append("line")
  .attr("x1", 8).attr("x2", BW - 8)
  .attr("y1", barY("test") - barY.step() * barY.paddingInner() / 2)
  .attr("y2", barY("test") - barY.step() * barY.paddingInner() / 2)
  .attr("stroke", "#bbb").attr("stroke-dasharray", "3 3");

// ---------- error curves against df ----------
const EW = 360, EH = 250;
const me = { top: 12, right: 14, bottom: 40, left: 44 };
const dfS = d3.scaleLog().domain([1, N]).range([me.left, EW - me.right]);
const errS = d3.scaleLinear().domain([0, 0.65]).range([EH - me.bottom, me.top]);
const errSvg = d3.select("#errors").append("svg")
  .attr("viewBox", `0 0 ${EW} ${EH}`).attr("role", "img");
const errTitle = errSvg.append("title").attr("id", "err-title");
const errDesc = errSvg.append("desc").attr("id", "err-desc");
errSvg.attr("aria-labelledby", "err-title err-desc");

errSvg.append("g").attr("class", "axis")
  .attr("transform", `translate(0,${EH - me.bottom})`)
  .call(d3.axisBottom(dfS).tickValues([1, 2, 5, 10, 20, 50, 100]).tickFormat(d3.format("d")));
errSvg.append("g").attr("class", "axis")
  .attr("transform", `translate(${me.left},0)`)
  .call(d3.axisLeft(errS).ticks(6));
errSvg.append("text").attr("class", "axis-label")
  .attr("x", (me.left + EW - me.right) / 2).attr("y", EH - 6)
  .attr("text-anchor", "middle").text("df = n/K (log scale)");
errSvg.append("text").attr("class", "axis-label")
  .attr("transform", `translate(12,${(me.top + EH - me.bottom) / 2}) rotate(-90)`)
  .attr("text-anchor", "middle").text("expected error");

const Ks = d3.range(1, N + 1);
const errLine = acc => d3.line().x(K => dfS(N / K)).y(K => errS(acc(K)));
errSvg.append("line")
  .attr("x1", dfS(1)).attr("x2", dfS(N)).attr("y1", errS(SIGMA2)).attr("y2", errS(SIGMA2))
  .attr("stroke", "#777").attr("stroke-width", 1.5).attr("stroke-dasharray", "2 3");
errSvg.append("path").attr("fill", "none").attr("stroke", "#c01c28").attr("stroke-width", 2.5)
  .attr("d", errLine(testErr)(Ks));
errSvg.append("path").attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 2.5)
  .attr("stroke-dasharray", "6 4")
  .attr("d", errLine(K => trainErr[K])(Ks));
// Direct labels at the flexible end, so the curves don't depend on the legend's colours.
errSvg.append("text").attr("class", "bar-label").attr("x", dfS(N) - 4).attr("y", errS(testErr(1)) - 6)
  .attr("text-anchor", "end").text("test");
errSvg.append("text").attr("class", "bar-label").attr("x", dfS(N / 4)).attr("y", errS(trainErr[4]) + 16)
  .attr("text-anchor", "middle").text("training");
errSvg.append("text").attr("class", "bar-label").attr("x", dfS(1) + 4).attr("y", errS(SIGMA2) + 14)
  .style("fill", "#555").text("σ²");

const marker = errSvg.append("g");
const markerLine = marker.append("line")
  .attr("y1", me.top).attr("y2", EH - me.bottom).attr("stroke", "#333").attr("stroke-width", 1);
const markerTest = marker.append("circle").attr("r", 5)
  .attr("fill", "#c01c28").attr("stroke", "#fff").attr("stroke-width", 1.5);
const markerTrain = marker.append("rect").attr("width", 9).attr("height", 9)
  .attr("fill", "#1a5fb4").attr("stroke", "#fff").attr("stroke-width", 1.5);
const markerLabel = marker.append("text").attr("class", "bar-label").attr("y", me.top + 10);

// ---------- table ----------
function renderTable(K) {
  const rows = Array.from(new Set([...TABLE_KS, K])).sort((a, b) => a - b);
  d3.select("#table-body").selectAll("tr").data(rows, d => d).join("tr")
    .classed("current", d => d === K)
    .html(d => `<th scope="row">${d}${d === K ? " (current)" : ""}</th>` +
      `<td>${fmtDf(N / d)}</td><td>${fmt3(bias2[d])}</td><td>${fmt3(variance(d))}</td>` +
      `<td>${fmt3(testErr(d))}</td><td>${fmt3(trainErr[d])}</td>`);
}

// ---------- update ----------
const slider = document.getElementById("k-slider");
const statsEl = document.getElementById("stats");
let liveTimer = null;

function update() {
  const K = +slider.value;
  const df = N / K;
  const v = {
    bias: bias2[K], var: variance(K), noise: SIGMA2, test: testErr(K), train: trainErr[K],
  };

  // Main panel
  pointLayer.selectAll("circle").data(ySets[0]).join("circle")
    .attr("cx", (d, i) => xS(X_FIXED[i])).attr("cy", d => yS(d)).attr("r", 2.6)
    .attr("fill", "none").attr("stroke", "#555").attr("stroke-width", 1);
  fitLayer.selectAll("path").data(fitsByK).join("path")
    .attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 1.1).attr("stroke-opacity", 0.55)
    .attr("d", set => stepLine(set.map(m => m[K - 1])));
  meanPath.attr("d", stepLine(meanFitByK.map(m => m[K - 1])));
  // Spread of the 20 fits, averaged over x: an empirical check on σ²/K.
  let spread = 0;
  for (let j = 0; j < N_PLOT; j++) {
    const vals = fitsByK.map(set => set[j][K - 1]);
    spread += d3.variance(vals);
  }
  spread /= N_PLOT;
  fitsTitle.text(`Twenty KNN fits with K = ${K}, and the true function f(x) = sin(2πx)`);
  fitsDesc.text(
    `x from 0 to 1, y from −2 to 2. The thick black curve is sin(2πx), one full wave peaking at 1 near x = 0.25 ` +
    `and bottoming at −1 near x = 0.75. ${N_SETS} thin step-function curves are KNN fits with K = ${K}, ` +
    `each to its own simulated training set; open circles show training set 1. The dashed curve is the average fit. ` +
    (K <= 3 ? "The fits are very jagged and scatter widely around the true curve. "
      : K >= 40 ? "The fits are flat and tightly bundled, and miss the peak and trough by a similar amount. "
      : "The fits follow the wave with moderate scatter. ") +
    `Average squared gap between the dashed curve and the true curve (bias squared): ${fmt3(v.bias)}. ` +
    `Variance of the 20 fits around each other, averaged over x: ${fmt3(spread)} (theory: ${fmt3(v.var)}).`);

  // Bars
  barRects.attr("width", b => barX(v[b.key]) - barX(0));
  barVals.attr("x", b => barX(v[b.key]) + 4).text(b => fmt3(v[b.key]));
  barsTitle.text(`Error decomposition at K = ${K}`);
  barsDesc.text(`Bar chart. bias squared ${fmt3(v.bias)}, variance ${fmt3(v.var)}, ` +
    `noise sigma squared ${fmt3(v.noise)}, test error ${fmt3(v.test)}, which is their sum.`);

  // Error curves
  const px = dfS(df);
  markerLine.attr("x1", px).attr("x2", px);
  markerTest.attr("cx", px).attr("cy", errS(v.test));
  markerTrain.attr("x", px - 4.5).attr("y", errS(v.train) - 4.5);
  const right = px > (me.left + EW - me.right) / 2;
  markerLabel.attr("x", right ? px - 5 : px + 5).attr("text-anchor", right ? "end" : "start")
    .text(`K = ${K}`);
  let best = 1;
  for (let k = 1; k <= N; k++) if (testErr(k) < testErr(best)) best = k;
  errTitle.text(`Test and training error against degrees of freedom n/K, current K = ${K}`);
  errDesc.text(
    `Line chart, df from 1 to 100 on a log scale, error from 0 to 0.65. ` +
    `Test error is U-shaped: ${fmt3(testErr(N))} at df 1 (K = 100), lowest ${fmt3(testErr(best))} at K = ${best} ` +
    `(df ${fmtDf(N / best)}), rising to ${fmt3(testErr(1))} at df 100 (K = 1), and never below sigma squared ${fmt3(SIGMA2)}. ` +
    `Training error falls steadily from ${fmt3(trainErr[N])} at df 1 to ${fmt3(trainErr[1])} at df 100. ` +
    `At the current K = ${K}, df ${fmtDf(df)}: test error ${fmt3(v.test)}, training error ${fmt3(v.train)}.`);

  // Stats line and slider text
  document.getElementById("k-out").textContent = `K = ${K}`;
  slider.setAttribute("aria-valuetext", `K = ${K}, df ${fmtDf(df)}`);
  renderTable(K);

  // Debounce the live region so dragging doesn't flood a screen reader.
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => {
    document.getElementById("s-k").textContent = K;
    document.getElementById("s-df").textContent = fmtDf(df);
    document.getElementById("s-bias").textContent = fmt3(v.bias);
    document.getElementById("s-var").textContent = fmt3(v.var);
    document.getElementById("s-noise").textContent = fmt3(v.noise);
    document.getElementById("s-test").textContent = fmt3(v.test);
    document.getElementById("s-train").textContent = fmt3(v.train);
  }, statsEl.dataset.ready ? 300 : 0);
  statsEl.dataset.ready = "1";
}

slider.addEventListener("input", update);
// Shift+arrow moves K by 10 (Page Up/Down, Home/End work natively).
slider.addEventListener("keydown", event => {
  if (!event.shiftKey) return;
  const delta = { ArrowRight: 10, ArrowUp: 10, ArrowLeft: -10, ArrowDown: -10 }[event.key];
  if (delta === undefined) return;
  event.preventDefault();
  slider.value = Math.min(N, Math.max(1, +slider.value + delta));
  update();
});
document.getElementById("resample").addEventListener("click", () => { seed += 1; simulate(); update(); });
document.getElementById("reset").addEventListener("click", () => {
  seed = 1; slider.value = 10; simulate(); update();
});

simulate();
update();
