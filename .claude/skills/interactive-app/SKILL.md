---
name: interactive-app
description: Build a new standalone interactive D3 demo for a DSAI 621 topic in apps/topics/ and register it on the landing page. Use when asked for an app, demo, visualization, simulation or interactive for a concept.
---

# Interactive app (D3, GitHub Pages)

## Structure

- `apps/topics/<kebab-slug>/index.html` + `app.js`. D3 v7 from `https://d3js.org/d3.v7.min.js`
  and nothing else unless it's needed (for 3D, follow how the regression course's
  `ssr-surface`/`optimization-methods` did it).
- Add an `<li>` to `apps/index.html`: link text = title, `p.desc` = one sentence saying what the
  student does and what they will see.
- Deploys automatically when `apps/**` is pushed to `main`.

## Reference

Read `../../dsai601/regression_f26/apps/topics/ols-fit/` (small, canonical) and one richer app
close to the requested one (e.g. `regression-diagnostics`, `sampling-distributions`,
`logistic-regression`). Match their look: system font, max-width ~760px, `#1a5fb4` accent,
`← All topics` back link, short instructions paragraph, live stats line.

## Before coding

Write me a 3–5 line spec: the one idea the app teaches, what the student manipulates (sliders,
drag, click), the linked views, and what they should notice. Wait for my OK unless I've said to
just build it.

## Requirements

- **Correct math.** Implement the algorithm honestly (actual ridge path, actual CART splits,
  actual k-means iterations). Use a seeded RNG (e.g. mulberry32) so a "Reset" reproduces the
  same data. Sanity-check numbers against sklearn in a scratch Python script when feasible.
- **Accessibility.** Everything doable with the mouse must also work from the keyboard (Tab to
  points or controls, arrows to move, Shift for bigger steps). Labelled `<input type="range">`
  controls; `role="img"` + `aria-label` on SVGs; stats in an `aria-live="polite"` region; a
  keyboard/screen-reader instruction paragraph. Never encode meaning in colour alone (use a
  shape or dash pattern too).
- **Performance.** Keep updates under ~16ms; precompute or throttle anything heavy.
- No build step and no frameworks. It must work from `python3 -m http.server`.

## Verify

Serve `apps/` with `python3 -m http.server 8000` (in the background) and open
`http://localhost:8000/topics/<slug>/` in the browser pane. Exercise every control and check the
console for errors. Also check phone width (375px) and keyboard-only use. Stop the server
afterwards.
