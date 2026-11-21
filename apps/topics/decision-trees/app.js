// Decision trees grown by Gini, entropy and classification error, side by side (ISL 8.1, ESL 9.2).
// One problem: x1..x4 ~ U[0, 10]; class 2 when (x1 >= 5 and x2 >= 4) or x3 >= 8; x4 is noise;
// 15% of labels flipped. y = 0 is "class 1", y = 1 is "class 2".
//   CART recursive binary splitting. Candidate cuts on predictor j are midpoints between adjacent
//   distinct values, and both children must keep at least minLeaf points. The best cut of a node
//   maximizes n_t Q_t − n_L Q_L − n_R Q_R (ties: lowest-numbered predictor, then smallest cutpoint,
//   as sklearn scans). Pure nodes are not split.
//   "Step" splits, in each tree, the leaf with the largest n_t-weighted reduction (best-first, like
//   sklearn's max_leaf_nodes), so growing to the end gives the same tree as depth-first growth.

// CORE-BEGIN (pure functions; tested outside the browser against sklearn)
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

const IMP = {
  gini: p => 2 * p * (1 - p),
  entropy: p => (p <= 0 || p >= 1) ? 0 : -(p * Math.log(p) + (1 - p) * Math.log(1 - p)),
  error: p => Math.min(p, 1 - p),
};
// Count-weighted impurity n·Q for a node with c1 class-2 points out of n.
function wimp(n, c1, crit) { return n > 0 ? n * IMP[crit](c1 / n) : 0; }

// D = { x: [x1, ..., xp] (Float64Arrays), y: Uint8Array, n }. idx: array of row indices.
function bestSplit(D, idx, crit, minLeaf) {
  const n = idx.length;
  let c1 = 0;
  for (const i of idx) c1 += D.y[i];
  if (c1 === 0 || c1 === n) return null;                      // pure
  const parent = wimp(n, c1, crit);
  let best = null;
  for (let j = 0; j < D.x.length; j++) {
    const xs = D.x[j];
    const ord = idx.slice().sort((a, b) => xs[a] - xs[b]);
    let nL = 0, kL = 0;
    for (let p = 0; p < n - 1; p++) {
      nL++; kL += D.y[ord[p]];
      const a = xs[ord[p]], b = xs[ord[p + 1]];
      if (!(b > a) || nL < minLeaf || n - nL < minLeaf) continue;
      const gain = parent - wimp(nL, kL, crit) - wimp(n - nL, c1 - kL, crit);
      if (!best || gain > best.gain + 1e-12) best = { j, s: (a + b) / 2, gain, dQ: gain / n };
    }
  }
  return best;
}

function newTree(D, idx) {
  const tree = { nodes: [], D };
  tree.root = addNode(tree, idx, 0);
  return tree;
}
function addNode(tree, idx, depth) {
  let c1 = 0;
  for (const i of idx) c1 += tree.D.y[i];
  const node = { id: tree.nodes.length, idx, n: idx.length, c1, depth, split: null, left: null, right: null };
  tree.nodes.push(node);
  return node;
}
function splitNode(tree, node, b) {
  const xs = tree.D.x[b.j];
  node.split = b;
  node.left = addNode(tree, node.idx.filter(i => xs[i] < b.s), node.depth + 1);
  node.right = addNode(tree, node.idx.filter(i => xs[i] >= b.s), node.depth + 1);
}
function leavesOf(tree) {
  const out = [];
  (function walk(nd) { if (!nd.split) { out.push(nd); return; } walk(nd.left); walk(nd.right); })(tree.root);
  return out;
}
// One best-first step: split the leaf with the largest n_t-weighted reduction. Returns the node, or null.
function growStep(tree, crit, minLeaf) {
  let best = null;
  for (const nd of leavesOf(tree)) {
    const b = bestSplit(tree.D, nd.idx, crit, minLeaf);
    if (b && (!best || b.gain > best.b.gain + 1e-12)) best = { nd, b };
  }
  if (!best) return null;
  splitNode(tree, best.nd, best.b);
  return best.nd;
}
const majority = nd => (2 * nd.c1 > nd.n ? 1 : 0);              // ties go to class 1 (y = 0)
function predict(tree, X, i) {
  let nd = tree.root;
  while (nd.split) nd = X.x[nd.split.j][i] < nd.split.s ? nd.left : nd.right;
  return majority(nd);
}
function errorRate(tree, X) {
  let e = 0;
  for (let i = 0; i < X.n; i++) if (predict(tree, X, i) !== X.y[i]) e++;
  return e / X.n;
}
// CORE-END

// ---------- data ----------
const P = 4, N = 400, N_TEST = 5000, NOISE = 0.15;
// Class 2 above a wavy curve in (x1, x2), but only where x3 >= 2; x4 is noise. No small tree fits the wave.
const wave = x1 => 5 + 3 * Math.sin(0.8 * x1);
const truth = x => (x[1] >= wave(x[0]) && x[2] >= 2 ? 1 : 0);
function makeData(n, seed) {
  const rand = mulberry32(seed);
  const x = Array.from({ length: P }, () => new Float64Array(n)), y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < P; j++) { x[j][i] = 10 * rand(); row.push(x[j][i]); }
    y[i] = truth(row) ^ (rand() < NOISE ? 1 : 0);
  }
  return { x, y, n };
}
const TEST = makeData(N_TEST, 424242);

const CRITS = [
  { key: "gini", name: "Gini index", by: "the Gini index" },
  { key: "entropy", name: "Entropy", by: "entropy" },
  { key: "error", name: "Classification error", by: "classification error" },
];

// ---------- stopping rules ----------
// Every rule picks a final tree: a subtree of the tree grown until the nodes are pure (or too small).
// Step then reveals that final tree's splits best-first (largest n_t-weighted reduction first).
const RULES = {
  depth:  { label: "Maximum depth", short: "maximum depth", values: [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15], start: 3 },
  leaf:   { label: "Minimum leaf size", short: "minimum leaf size", values: [1, 2, 3, 5, 8, 10, 15, 20, 30, 50], start: 5 },
  leaves: { label: "Maximum number of leaves", short: "maximum leaves", values: [2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 80], start: 5 },
  prune:  { label: "Pruning penalty α", short: "α", values: [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40], start: 5 },
};
const S = { rule: "leaf", vi: RULES.leaf.start, seed: 1, steps: 0 };   // steps = Infinity once grown fully
const curValue = () => RULES[S.rule].values[S.vi];
const fmtV = (r, v) => (r === "prune" ? d3.format("~g")(v) : String(v));
const settingText = (r, v) => `${RULES[r].short} ${r === "prune" ? "= " : ""}${fmtV(r, v)}`;

// Grow until pure or too small, depth-first (the same tree best-first growth reaches).
function growFull(D, crit, minLeaf) {
  const nodes = [];
  (function rec(idx, depth) {
    let c1 = 0;
    for (const i of idx) c1 += D.y[i];
    const nd = { id: nodes.length, n: idx.length, c1, depth, bs: null, left: null, right: null };
    nodes.push(nd);
    const b = bestSplit(D, idx, crit, minLeaf);
    if (b) {
      nd.bs = b;
      const xs = D.x[b.j];
      nd.left = rec(idx.filter(i => xs[i] < b.s), depth + 1);
      nd.right = rec(idx.filter(i => xs[i] >= b.s), depth + 1);
    }
    return nd;
  })(Array.from({ length: D.n }, (_, i) => i), 0);
  return { root: nodes[0], nodes };
}
// Internal nodes of `inc` (a Set of ids) in best-first order.
function bestFirst(root, inc) {
  const order = [], front = [root];
  for (;;) {
    let bi = -1;
    front.forEach((nd, i) => { if (inc.has(nd.id) && (bi < 0 || nd.bs.gain > front[bi].bs.gain + 1e-12)) bi = i; });
    if (bi < 0) return order;
    const nd = front[bi];
    order.push(nd);
    front.splice(bi, 1, nd.left, nd.right);
  }
}
// Cost-complexity pruning with the training misclassification count (ISL 8.1.2):
// keep a split only if its subtree's errors + α·leaves beat the node as a leaf. Ties prune.
function pruneKeep(root, alpha) {
  const keep = new Set();
  (function rec(nd) {
    const asLeaf = Math.min(nd.c1, nd.n - nd.c1) + alpha;
    if (!nd.bs) return { cost: asLeaf, ids: [] };
    const L = rec(nd.left), R = rec(nd.right);
    if (L.cost + R.cost < asLeaf - 1e-9) return { cost: L.cost + R.cost, ids: [nd.id, ...L.ids, ...R.ids] };
    return { cost: asLeaf, ids: [] };
  })(root).ids.forEach(id => keep.add(id));
  return keep;
}
// The final tree for a rule and value: { G (grown tree), inc (Set of internal-node ids) }.
function finalTree(D, crit, r, v, grownCache) {
  const grown = m => {
    const key = `${crit}|${m}`;
    if (!grownCache[key]) grownCache[key] = growFull(D, crit, m);
    return grownCache[key];
  };
  const G = grown(r === "leaf" ? v : 1);
  const internal = G.nodes.filter(nd => nd.bs);
  let inc;
  if (r === "leaf") inc = new Set(internal.map(nd => nd.id));
  else if (r === "depth") inc = new Set(internal.filter(nd => nd.depth < v).map(nd => nd.id));
  else if (r === "leaves") inc = new Set(bestFirst(G.root, new Set(internal.map(nd => nd.id))).slice(0, v - 1).map(nd => nd.id));
  else inc = pruneKeep(G.root, v);
  return { G, inc };
}
function walk(G, inc, X, i) {
  let nd = G.root;
  while (nd.bs && inc.has(nd.id)) nd = X.x[nd.bs.j][i] < nd.bs.s ? nd.left : nd.right;
  return majority(nd);
}

let D, grownNow = {}, T = {};   // T[crit] = { tree (the displayed view), last (id of the latest split), done }
function rebuild(newData) {
  if (newData || !D) { D = makeData(N, S.seed); grownNow = {}; }
  for (const c of CRITS) {
    const { G, inc } = finalTree(D, c.key, S.rule, curValue(), grownNow);
    const order = bestFirst(G.root, inc);
    const k = Math.min(S.steps, order.length);
    const shown = new Set(order.slice(0, k).map(nd => nd.id));
    const nodes = [];
    const view = (function v(nd) {
      const o = { id: nd.id, n: nd.n, c1: nd.c1, depth: nd.depth, split: shown.has(nd.id) ? nd.bs : null, left: null, right: null };
      nodes.push(o);
      if (o.split) { o.left = v(nd.left); o.right = v(nd.right); }
      return o;
    })(G.root);
    const byId = Object.fromEntries(nodes.map(o => [o.id, o]));
    T[c.key] = { tree: { root: view, nodes }, last: k ? order[k - 1].id : null, byId, done: k >= order.length, total: order.length + 1 };
  }
}

// ---------- formatting ----------
const XN = ["x₁", "x₂", "x₃", "x₄"];
const f2 = v => d3.format(".2f")(v);
const f3 = v => d3.format(".3f")(Math.abs(v) < 5e-4 ? 0 : v);
const nLeaves = k => `${k} ${k === 1 ? "leaf" : "leaves"}`;
const rule = nd => `${XN[nd.split.j]} < ${f2(nd.split.s)}`;
const counts = nd => `${nd.n - nd.c1} ○ / ${nd.c1} ▲`;
const CLASS_COLOR = ["#3d3846", "#c64600"];
const tri = d3.symbol().type(d3.symbolTriangle);
const circ = d3.symbol().type(d3.symbolCircle);

// ---------- one panel per criterion ----------
const panels = {};
for (const c of CRITS) {
  const host = d3.select("#trees").append("section").attr("aria-labelledby", `h-${c.key}`);
  host.append("h3").attr("id", `h-${c.key}`).text(`Split by ${c.by}`);
  const sub = host.append("p").attr("class", "tree-sub");
  const svg = host.append("figure").attr("class", "panel").style("margin", "0").append("svg").attr("role", "img");
  const pat = svg.append("defs").append("pattern").attr("id", `hatch-${c.key}`).attr("width", 7).attr("height", 7)
    .attr("patternUnits", "userSpaceOnUse").attr("patternTransform", "rotate(45)");
  pat.append("rect").attr("width", 7).attr("height", 7).attr("fill", "#fbe3d3");
  pat.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 7).attr("stroke", "#eab08c").attr("stroke-width", 2);
  const links = svg.append("g").attr("aria-hidden", "true");
  const nodes = svg.append("g").attr("aria-hidden", "true");
  const det = host.append("details");
  det.append("summary").text("Rules as text");
  const pre = det.append("pre");
  panels[c.key] = { sub, svg, links, nodes, pre };
}


// ---------- bias, variance and test error for each setting of the rule (simulated) ----------
// SIM_R training sets of N points. For each setting we record every tree's predictions on SIM_TEST
// fresh x's. With f*(x) the Bayes classifier and 15% label noise, a prediction's expected 0-1 loss is
// 0.15 + 0.7·1[ŷ ≠ f*(x)]. The majority vote of the SIM_R trees is the main prediction, and
//   test error = 0.15 + 0.7·P(tree ≠ f*),  bias = 0.7·P(main ≠ f*),  variance = test − 0.15 − bias.
const SIM_R = 51, SIM_TEST = 2000, BAYES = NOISE;
const SIM_X = makeData(SIM_TEST, 777);
const SIM_F = Uint8Array.from({ length: SIM_TEST }, (_, i) => truth(SIM_X.x.map(col => col[i])));
const simCache = {};
function simulateRule(r) {
  if (simCache[r]) return simCache[r];
  const vals = RULES[r].values, out = {};
  for (const c of CRITS) {
    const votes = vals.map(() => new Uint16Array(SIM_TEST));
    const acc = vals.map(() => ({ leaves: 0, train: 0, wrong: 0 }));
    for (let s = 0; s < SIM_R; s++) {
      const Ds = makeData(N, 100000 + s), cache = {};
      vals.forEach((v, vi) => {
        const { G, inc } = finalTree(Ds, c.key, r, v, cache);
        const a = acc[vi], vt = votes[vi];
        a.leaves += inc.size + 1;
        let tr = 0;
        for (let i = 0; i < Ds.n; i++) tr += walk(G, inc, Ds, i) !== Ds.y[i];
        a.train += tr / Ds.n;
        let w = 0;
        for (let i = 0; i < SIM_TEST; i++) { const p = walk(G, inc, SIM_X, i); vt[i] += p; w += p !== SIM_F[i]; }
        a.wrong += w / SIM_TEST;
      });
    }
    const rows = vals.map((v, vi) => {
      let mainWrong = 0;
      for (let i = 0; i < SIM_TEST; i++) mainWrong += (2 * votes[vi][i] > SIM_R ? 1 : 0) !== SIM_F[i];
      const a = acc[vi];
      const test = BAYES + (1 - 2 * BAYES) * a.wrong / SIM_R, bias = (1 - 2 * BAYES) * mainWrong / SIM_TEST;
      return { v, leaves: a.leaves / SIM_R, train: a.train / SIM_R, test, bias, variance: test - BAYES - bias, excess: test - BAYES };
    });
    let b = 0;
    rows.forEach((q, i) => { if (q.test < rows[b].test - 1e-12) b = i; });
    out[c.key] = { rows, best: rows[b] };
  }
  return (simCache[r] = out);
}

const BV = {
  W: 720, H: 260, m: { t: 24, r: 14, b: 40, l: 48 },
  lines: [
    { key: "bias", name: "bias", color: "#c64600", width: 1.8, dash: "7 4", sym: d3.symbolCircle },
    { key: "variance", name: "variance", color: "#1a5fb4", width: 1.8, dash: "9 3 2 3", sym: d3.symbolSquare },
    { key: "excess", name: "test error − 0.15", color: "#222", width: 2.6, dash: null, sym: d3.symbolTriangle },
    { key: "train", name: "training error", color: "#1e7a3c", width: 1.6, dash: "2 3", sym: d3.symbolCross },
  ],
};
const bvPanels = {};
for (const c of CRITS) {
  const host = d3.select("#bv").append("section").attr("class", "bv-panel").attr("aria-labelledby", `bvh-${c.key}`);
  host.append("h3").attr("id", `bvh-${c.key}`).text(`Split by ${c.by}`);
  const svg = host.append("svg").attr("role", "img").attr("viewBox", `0 0 ${BV.W} ${BV.H}`);
  const g = svg.append("g").attr("aria-hidden", "true");
  const sub = host.append("p").attr("class", "bv-sub");
  const det = host.append("details");
  det.append("summary").text("Values as text");
  const tbl = det.append("table").attr("class", "data small");
  const hr = tbl.append("thead").append("tr");
  const tb = tbl.append("tbody");
  bvPanels[c.key] = { svg, g, sub, hr, tb };
}

function drawBV() {
  const sim = simulateRule(S.rule), R = RULES[S.rule], v = curValue();
  const { H, m } = BV;
  const W = Math.max(340, Math.min(BV.W, document.getElementById("bv").clientWidth || BV.W));   // keep text readable on phones
  const all = CRITS.flatMap(c => sim[c.key].rows);
  const x = d3.scaleLog().domain([d3.min(all, q => q.leaves) / 1.08, d3.max(all, q => q.leaves) * 1.08]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(all, q => q.variance)), d3.max(all, q => Math.max(q.train, q.excess, q.bias)) * 1.05]).nice().range([H - m.b, m.t]);
  const ticks = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200].filter(t => t >= x.domain()[0] && t <= x.domain()[1]);
  for (const c of CRITS) {
    const s = sim[c.key], pn = bvPanels[c.key], g = pn.g;
    const cur = s.rows[S.vi];
    pn.svg.attr("viewBox", `0 0 ${W} ${H}`);
    g.selectAll("*").remove();
    g.append("g").attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickValues(ticks).tickFormat(d3.format("d")).tickSizeOuter(0));
    g.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
    g.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("font-size", 12)
      .text("mean number of leaves (log scale)");
    if (y.domain()[0] < 0) g.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", "#bbb");
    g.append("line").attr("x1", x(cur.leaves)).attr("x2", x(cur.leaves)).attr("y1", m.t).attr("y2", H - m.b)
      .attr("stroke", "#777").attr("stroke-dasharray", "3 3");
    const label = `now: ${settingText(S.rule, v)}`, half = label.length * 3.6;
    g.append("text").attr("x", Math.min(W - m.r - half, Math.max(m.l + half, x(cur.leaves)))).attr("y", 13)
      .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 700)
      .attr("stroke", "#fff").attr("stroke-width", 3).attr("paint-order", "stroke").text(label);
    for (const L of BV.lines) {
      g.append("path").datum(s.rows).attr("fill", "none").attr("stroke", L.color).attr("stroke-width", L.width)
        .attr("stroke-dasharray", L.dash).attr("d", d3.line().x(q => x(q.leaves)).y(q => y(q[L.key])));
      g.append("g").selectAll("path").data(s.rows).join("path")
        .attr("transform", q => `translate(${x(q.leaves)},${y(q[L.key])})`)
        .attr("d", (q, i) => d3.symbol().type(L.sym).size(i === S.vi ? 150 : 40)())
        .attr("fill", L.color).attr("stroke", (q, i) => (i === S.vi ? "#000" : "#fff")).attr("stroke-width", (q, i) => (i === S.vi ? 1.6 : 0.8));
    }
    const sentence = `At ${settingText(S.rule, v)} the trees have ${d3.format(".1f")(cur.leaves)} leaves on average: ` +
      `test error ${f3(cur.test)} = 0.15 + bias ${f3(cur.bias)} + variance ${f3(cur.variance)}, training error ${f3(cur.train)}. ` +
      `Lowest test error ${f3(s.best.test)} at ${settingText(S.rule, s.best.v)}.`;
    pn.sub.text(sentence);
    pn.svg.attr("aria-label", `Bias, variance, test error minus 0.15 and training error against the mean number of leaves, ` +
      `for trees split by ${c.by} and stopped by ${R.short}. ${sentence} The values are under Values as text.`);
    pn.hr.selectAll("th").data([R.label, "Mean leaves", "Training error", "Bias", "Variance", "Test error"])
      .join("th").attr("scope", "col").text(d => d);
    pn.tb.selectAll("tr").data(s.rows).join(e => { const t = e.append("tr"); for (let i = 0; i < 6; i++) t.append("td"); return t; })
      .attr("class", (q, i) => (i === S.vi ? "current" : null))
      .selectAll("td").data((q, i) => [fmtV(S.rule, q.v) + (i === S.vi ? " (now)" : ""), d3.format(".1f")(q.leaves), f3(q.train), f3(q.bias), f3(q.variance), f3(q.test)])
      .join("td").text(d => d);
  }
}

function rulesText(nd, indent, prefix) {
  const pad = "  ".repeat(indent);
  if (!nd.split) return `${pad}${prefix}leaf: class ${majority(nd) + 1} (${nd.n} points: ${counts(nd)})\n`;
  return `${pad}${prefix}${rule(nd)}?  (${nd.n} points: ${counts(nd)}; Δ = ${f3(nd.split.dQ)})\n` +
    rulesText(nd.left, indent + 1, "yes → ") + rulesText(nd.right, indent + 1, "no  → ");
}

function drawTree(c) {
  const { tree, last } = T[c.key];
  const pn = panels[c.key];
  const root = d3.hierarchy(tree.root, nd => (nd.split ? [nd.left, nd.right] : null));
  const L = root.leaves().length;
  const compact = L > 9;
  const W = 720, rowH = compact ? 52 : 66, top = 26;
  const H = top + root.height * rowH + 30;
  pn.svg.attr("viewBox", `0 0 ${W} ${H}`);
  d3.tree().size([W - 70, root.height * rowH]).separation((a, b) => (a.parent === b.parent ? 1 : 1.15))(root);
  const X = d => d.x + 35, Y = d => d.y + top;
  pn.links.selectAll("path").data(root.links(), d => d.target.data.id).join("path")
    .attr("fill", "none").attr("stroke", "#777").attr("stroke-width", 1.4)
    .attr("d", d => `M${X(d.source)},${Y(d.source)} V${(Y(d.source) + Y(d.target)) / 2} H${X(d.target)} V${Y(d.target)}`);
  // yes / no labels on the first split only
  pn.links.selectAll("text").data(root.children ? root.children.map((ch, i) => ({ ch, t: i ? "no" : "yes" })) : [])
    .join("text").attr("font-size", 10).attr("fill", "#555").attr("text-anchor", "middle")
    .attr("x", d => X(d.ch)).attr("y", d => (Y(root) + Y(d.ch)) / 2 - 4).text(d => d.t);
  const g = pn.nodes.selectAll("g.n").data(root.descendants(), d => d.data.id).join(enter => {
    const e = enter.append("g").attr("class", "n");
    e.append("rect"); e.append("path"); e.append("text").attr("class", "t1"); e.append("text").attr("class", "t2");
    return e;
  }).attr("transform", d => `translate(${X(d)},${Y(d)})`);
  const box = d => (d.data.split ? (compact ? [62, 30] : [92, 34]) : (compact ? [18, 18] : [92, 34]));
  g.select("rect").attr("x", d => -box(d)[0] / 2).attr("y", d => -box(d)[1] / 2)
    .attr("width", d => box(d)[0]).attr("height", d => box(d)[1]).attr("rx", 4)
    .attr("fill", d => (d.data.split ? "#f2f2f4" : majority(d.data) ? `url(#hatch-${c.key})` : "#fff"))
    .attr("stroke", d => (d.data.id === last ? "#1a5fb4" : "#555"))
    .attr("stroke-width", d => (d.data.id === last ? 3.5 : 1.2));
  g.select("path").attr("d", d => (d.data.split ? null : majority(d.data) ? tri.size(compact ? 34 : 46)() : circ.size(compact ? 34 : 46)()))
    .attr("transform", d => (compact ? "translate(0,0)" : "translate(-33,0)"))
    .attr("fill", d => (majority(d.data) ? CLASS_COLOR[1] : "#fff")).attr("stroke", d => (majority(d.data) ? "none" : CLASS_COLOR[0]))
    .attr("stroke-width", 1.4);
  g.select("text.t1").attr("text-anchor", "middle").attr("font-weight", 700)
    .attr("font-size", compact ? 10 : 11).attr("y", d => (d.data.split ? -2 : 4)).attr("x", d => (d.data.split || compact ? 0 : 8))
    .text(d => (d.data.split ? rule(d.data) : compact ? "" : counts(d.data)));
  g.select("text.t2").attr("text-anchor", "middle").attr("font-size", compact ? 9 : 10).attr("y", 10).attr("fill", "#333")
    .text(d => (d.data.split ? `Δ ${f3(d.data.split.dQ)}` : ""));
  g.order();
  pn.pre.text(rulesText(tree.root, 0, ""));
}


// ---------- render ----------
function draw() {
  const res = CRITS.map(c => {
    const { tree, last, byId } = T[c.key];
    const leaves = leavesOf(tree).length;
    const tr = errorRate(tree, D), te = errorRate(tree, TEST);
    const used = [...new Set(tree.nodes.filter(nd => nd.split).map(nd => nd.split.j))].sort().map(j => XN[j]);
    return { c, leaves, tr, te, used, lastNd: last !== null ? byId[last] : null };
  });
  for (const r of res) {
    drawTree(r.c);
    const pn = panels[r.c.key], t = T[r.c.key];
    pn.sub.text(`${nLeaves(r.leaves)} of ${t.total}, training error ${f3(r.tr)}, test error ${f3(r.te)}` +
      (r.lastNd ? `. Latest split (outlined): ${rule(r.lastNd)}, Δ = ${f3(r.lastNd.split.dQ)}.` : "."));
    pn.svg.attr("aria-label", `Tree split by ${r.c.by}, ${nLeaves(r.leaves)}` +
      (r.leaves > 1 ? `, first split ${rule(t.tree.root)}` : "") + ". The full rules are under Rules as text.");
  }
  d3.select("#rows").selectAll("tr").data(res).join(e => {
    const t = e.append("tr"); t.append("th").attr("scope", "row");
    for (let i = 0; i < 4; i++) t.append("td");
    return t;
  }).call(tr => {
    tr.select("th").text(r => r.c.name);
    tr.selectAll("td").data(r => [r.leaves, f3(r.tr), f3(r.te), r.used.join(", ") || "none"]).join("td").text(v => v);
  });
  const anyGrow = CRITS.some(c => !T[c.key].done);
  document.getElementById("step").disabled = !anyGrow;
  document.getElementById("growfull").disabled = !anyGrow;
  document.getElementById("reset").disabled = S.steps === 0;
  document.getElementById("stats").textContent =
    `Sample ${S.seed}, stopped by ${settingText(S.rule, curValue())}. ` +
    res.map(r => `${r.c.name}: ${nLeaves(r.leaves)}, test error ${f3(r.te)}` +
      (r.lastNd ? `, latest split ${rule(r.lastNd)}` : "")).join(". ") + "." +
    (anyGrow ? "" : " Every tree has reached the stopping rule.");
  scheduleBV();
}
// The simulation for a new rule takes up to a few seconds, so draw the trees first.
let bvTimer;
function scheduleBV() {
  clearTimeout(bvTimer);
  if (simCache[S.rule]) { drawBV(); return; }
  for (const c of CRITS) bvPanels[c.key].sub.text("Simulating 51 training sets for this stopping rule…");
  document.getElementById("bv").setAttribute("aria-busy", "true");
  bvTimer = setTimeout(() => { drawBV(); document.getElementById("bv").removeAttribute("aria-busy"); }, 50);
}

// ---------- controls ----------
const slider = document.getElementById("param-slider");
function syncSlider() {
  const R = RULES[S.rule], v = curValue();
  document.getElementById("param-label").textContent = R.label;
  slider.max = R.values.length - 1;
  slider.value = S.vi;
  slider.setAttribute("aria-valuetext", fmtV(S.rule, v));
  document.getElementById("param-out").textContent = fmtV(S.rule, v);
}
document.querySelectorAll("input[name=rule]").forEach(el => el.addEventListener("change", () => {
  S.rule = el.value; S.vi = RULES[S.rule].start;
  syncSlider(); rebuild(); draw();
}));
slider.addEventListener("input", () => { S.vi = +slider.value; syncSlider(); rebuild(); draw(); });
document.getElementById("step").addEventListener("click", () => {
  if (!Number.isFinite(S.steps)) return;
  S.steps += 1; rebuild(); draw();
});
document.getElementById("growfull").addEventListener("click", () => { S.steps = Infinity; rebuild(); draw(); });
document.getElementById("reset").addEventListener("click", () => { S.steps = 0; rebuild(); draw(); });
document.getElementById("resample").addEventListener("click", () => { S.seed += 1; rebuild(true); draw(); });

let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(scheduleBV, 150); });

syncSlider();
rebuild();
draw();
