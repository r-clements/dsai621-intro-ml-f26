// Random forests on Example C.
// y = x1 + x2 + x3 + noise, x1..x3 standard normal with correlation 0.8, x4, x5 pure noise,
// n = 200, B = 100 trees. The fits were computed with scikit-learn (RandomForestRegressor with
// max_features = m and min_samples_leaf = leaf size; DecisionTreeRegressor for the single tree)
// on R = 100 simulated training sets, and are evaluated along the diagonal x1 = x2 = x3 = t,
// x4 = x5 = 0, where the truth is 3t. rf_data.json (made by tools/rf_app_data.py) stores every
// value multiplied by 100. Wrapped in a function so its names don't clash with app.js.

(() => {

const LEAF_VALUES = [1, 3, 5, 10];
const Y_DOMAIN = [-7.5, 7.5];
const T_DOMAIN = [-2, 2];

const fmt3 = d3.format(".3f");
const fmt2 = d3.format(".2f");
const minus = s => s.replace(/-/g, "−");

let D = null;                   // the loaded data, values rescaled
let grid = null, fGrid = null;
const R_SHOW = 30;
let shown = 0;
let liveTimer = null;

const scale = a => (Array.isArray(a[0]) ? a.map(scale) : Float64Array.from(a, v => v / 100));

// ---------- plots ----------
function makePlot(container, W, H, margin, ylab) {
  const svg = d3.select(container).append("svg")
    .attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img");
  const id = container.slice(1);
  const title = svg.append("title").attr("id", `${id}-title`);
  const desc = svg.append("desc").attr("id", `${id}-desc`);
  svg.attr("aria-labelledby", `${id}-title ${id}-desc`);
  const x = d3.scaleLinear().domain(T_DOMAIN).range([margin.left, W - margin.right]);
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
    .attr("text-anchor", "middle").text("t  (x₁ = x₂ = x₃ = t)");
  if (ylab) {
    svg.append("text").attr("class", "axis-label")
      .attr("transform", `translate(12,${(margin.top + H - margin.bottom) / 2}) rotate(-90)`)
      .attr("text-anchor", "middle").text(ylab);
  }
  const area = svg.append("g").attr("clip-path", `url(#${id}-clip)`);
  // Tree and forest predictions are step functions of t; curveStep draws them as steps.
  const step = d3.line().x((d, j) => x(grid[j])).y(d => y(d)).curve(d3.curveStep);
  const truth = area.append("path").attr("fill", "none").attr("stroke", "#111").attr("stroke-width", 2.5);
  return { svg, title, desc, x, y, area, step, truth };
}

const topPlot = makePlot("#rf-one", 720, 380, { top: 12, right: 14, bottom: 40, left: 44 }, "y");
const topPoints = topPlot.area.append("g");
const topTrees = topPlot.area.append("g");
const topSingle = topPlot.area.append("path").attr("fill", "none").attr("stroke", "#c64600")
  .attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3");
const topForest = topPlot.area.append("path").attr("fill", "none").attr("stroke", "#1a5fb4")
  .attr("stroke-width", 3).attr("stroke-linejoin", "round");
topPlot.truth.raise();

function makeManyPlot(sel, ylab) {
  const p = makePlot(sel, 360, 300, { top: 10, right: 10, bottom: 38, left: ylab ? 40 : 30 }, ylab);
  p.fits = p.area.append("g");
  p.truth.raise();
  p.mean = p.area.append("path").attr("fill", "none").attr("stroke", "#c64600")
    .attr("stroke-width", 2.5).attr("stroke-dasharray", "6 4");
  return p;
}
const leftPlot = makeManyPlot("#rf-left", "fitted value");
const rightPlot = makeManyPlot("#rf-right", null);

// ---------- helpers ----------
const mSlider = document.getElementById("rf-m-slider");
const leafSlider = document.getElementById("rf-leaf-slider");
const statsEl = document.getElementById("rf-stats");
const key = (m, leaf) => `m${m}_l${leaf}`;
const ise = curve => d3.mean(curve, (v, j) => (v - fGrid[j]) ** 2);
const jumps = curve => { let c = 0; for (let j = 1; j < curve.length; j++) if (Math.abs(curve[j] - curve[j - 1]) > 1e-9) c++; return c; };
const forestName = (m, leaf) =>
  m === 5 && leaf === 1 ? "bagging (m = 5, leaf size 1)"
    : `${m === 5 ? "bagging" : "forest"} (m = ${m}, leaf size ${leaf})`;

function drawMany(p, fits, mean) {
  p.fits.selectAll("path").data(fits.slice(0, R_SHOW)).join("path")
    .attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 1)
    .attr("stroke-opacity", 0.45).attr("d", p.step);
  p.mean.attr("d", p.step(mean));
}

// Describe how widely a set of fits spreads, at t = -1, 0, 1.
function spreadAt(fits) {
  return [-1, 0, 1].map(tv => {
    const j = d3.minIndex(grid, g => Math.abs(g - tv));
    const vals = fits.slice(0, R_SHOW).map(c => c[j]);
    return `${minus(fmt2(d3.min(vals)))} to ${minus(fmt2(d3.max(vals)))} at t = ${minus(String(tv))} (truth ${minus(String(3 * tv))})`;
  }).join("; ");
}

// ---------- update ----------
function update() {
  const m = +mSlider.value, leaf = LEAF_VALUES[+leafSlider.value];
  const S = D.settings[key(m, leaf)], bag = D.settings[key(5, 1)], tree = D.tree;
  const top = D.top[shown];
  const name = forestName(m, leaf);

  // Top plot.
  topPoints.selectAll("circle").data(top.s).join("circle")
    .attr("cx", d => topPlot.x(d)).attr("cy", (d, i) => topPlot.y(top.y[i])).attr("r", 2.6)
    .attr("fill", "none").attr("stroke", "#666").attr("stroke-width", 1);
  topTrees.selectAll("path").data(S.trees[shown]).join("path")
    .attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-width", 1)
    .attr("stroke-opacity", 0.3).attr("d", topPlot.step);
  const forestCurve = S.fits[shown];
  topSingle.attr("d", topPlot.step(top.tree));
  topForest.attr("d", topPlot.step(forestCurve));
  topPlot.title.text(`Training set ${shown + 1}: one fully grown tree and the ${name}`);
  topPlot.desc.text(
    `t from −2 to 2 along the diagonal, y from −7.5 to 7.5. Open circles are the ${D.n} training points, ` +
    `each plotted at the mean of its x₁, x₂ and x₃. The thick black line is the truth, 3t. ` +
    `The dashed step curve is one fully grown tree on all the data: it has ${jumps(top.tree)} jumps along the diagonal ` +
    `and its average squared distance from the line is ${fmt2(ise(top.tree))}. The faint curves are 6 of the forest's trees, ` +
    `and the thick blue curve is the average of all ${D.B}: ${jumps(forestCurve)} smaller jumps, average squared distance ` +
    `from the line ${fmt2(ise(forestCurve))}. At the ends, where there are few data, the forest predicts ` +
    `${minus(fmt2(forestCurve[0]))} at t = −2 and ${minus(fmt2(forestCurve[forestCurve.length - 1]))} at t = 2 (truth −6 and 6).`);

  // Bottom plots.
  drawMany(leftPlot, bag.fits, bag.mean);
  drawMany(rightPlot, S.fits, S.mean);
  document.getElementById("rf-right-heading").textContent =
    m === 5 && leaf === 1 ? "Bagging again (m = 5, leaf size 1)"
      : `${m === 5 ? "Bagging" : "Random forest"}, m = ${m}, leaf size ${leaf}`;
  leftPlot.title.text("Thirty bagged fits (m = 5, leaf size 1), one per training set");
  leftPlot.desc.text(
    `t from −2 to 2, fitted value from −7.5 to 7.5. ${R_SHOW} thin step curves, each the average of ${D.B} bootstrap trees ` +
    `on its own training set. They range from ${spreadAt(bag.fits)}. The dashed average of all ${D.R} lies close to the line, ` +
    `falling a little short of it near t = −2 and t = 2. Squared bias ${fmt3(bag.bias2)}, variance ${fmt3(bag.variance)}.`);
  rightPlot.title.text(`Thirty fits of the ${name}, one per training set`);
  rightPlot.desc.text(
    `Same axes. ${R_SHOW} thin curves, each a ${m === 5 ? "bagged fit" : "random forest"} of ${D.B} trees on its own training set. ` +
    `They range from ${spreadAt(S.fits)}. Squared bias ${fmt3(S.bias2)}, variance ${fmt3(S.variance)}` +
    (m === 5 && leaf === 1 ? ", the same as the left plot." : `, against ${fmt3(bag.variance)} for bagging.`));

  // Table.
  const mse = d => 1 + d.bias2 + d.variance;
  const rows = [
    ["Single fully grown tree", tree, false],
    ["Bagging (m = 5, leaf size 1)", bag, false],
    [`Current: ${name}`, S, true],
  ];
  d3.select("#rf-rows").selectAll("tr").data(rows).join("tr")
    .attr("class", d => (d[2] ? "current" : null))
    .html(([nm, d]) => `<th scope="row">${nm}</th><td>${fmt3(d.bias2)}</td><td>${fmt3(d.variance)}</td><td>${fmt3(mse(d))}</td>`);

  // Slider text and stats line (debounced so dragging doesn't flood a screen reader).
  const mText = m === 5 ? "m = 5 (bagging)" : `m = ${m} of 5`;
  const leafText = leaf === 1 ? "1 (fully grown)" : String(leaf);
  document.getElementById("rf-m-out").textContent = mText;
  document.getElementById("rf-leaf-out").textContent = leafText;
  mSlider.setAttribute("aria-valuetext", m === 5 ? "m = 5, all predictors: bagging" : `m = ${m} of 5 predictors`);
  leafSlider.setAttribute("aria-valuetext", leaf === 1 ? "minimum leaf size 1, fully grown trees" : `minimum leaf size ${leaf}`);
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => {
    document.getElementById("rf-s-bb").textContent = fmt3(bag.bias2);
    document.getElementById("rf-s-bv").textContent = fmt3(bag.variance);
    document.getElementById("rf-s-name").textContent = `m = ${m}, leaf size ${leaf}`;
    document.getElementById("rf-s-fb").textContent = fmt3(S.bias2);
    document.getElementById("rf-s-fv").textContent = fmt3(S.variance);
  }, statsEl.dataset.ready ? 300 : 0);
  statsEl.dataset.ready = "1";
}

function updateShownHeading() {
  document.getElementById("rf-one-heading").textContent = `One training set (number ${shown + 1} of ${D.top.length} shown here)`;
}

mSlider.addEventListener("input", update);
leafSlider.addEventListener("input", update);
document.getElementById("rf-next").addEventListener("click", () => {
  shown = (shown + 1) % D.top.length;
  updateShownHeading(); update();
});
document.getElementById("rf-reset").addEventListener("click", () => {
  shown = 0; mSlider.value = 5; leafSlider.value = 0;
  updateShownHeading(); update();
});

// ---------- load ----------
d3.json("rf_data.json").then(raw => {
  grid = Float64Array.from(raw.t);
  fGrid = Float64Array.from(grid, t => 3 * t);
  const fix = s => ({ ...s, mean: scale(s.mean), fits: scale(s.fits), trees: s.trees ? s.trees.map(scale) : undefined });
  D = {
    n: raw.n, R: raw.R, B: raw.B,
    tree: fix(raw.tree),
    settings: Object.fromEntries(Object.entries(raw.settings).map(([k, s]) => [k, fix(s)])),
    top: raw.top.map(d => ({ s: scale(d.s), y: scale(d.y), tree: scale(d.tree) })),
  };
  const line = [T_DOMAIN[0], T_DOMAIN[1]];
  for (const p of [topPlot, leftPlot, rightPlot]) {
    p.truth.attr("d", d3.line().x(t => p.x(t)).y(t => p.y(3 * t))(line));
  }
  document.getElementById("rf-status").textContent = "";
  for (const id of ["m-slider", "leaf-slider", "next", "reset"]) document.getElementById(`rf-${id}`).disabled = false;
  updateShownHeading(); update();
}).catch(err => {
  document.getElementById("rf-status").textContent = `Could not load the simulated fits (${err.message}).`;
});
})();
