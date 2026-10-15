---
name: handwritten-notes
description: Transcribe a scan of handwritten DSAI 621 class notes into an accessible HTML page (MathML, described SVG figures, WCAG 2.2 AA) and publish it to notes/. Use when given a handwritten-notes PDF or images.
---

# Handwritten notes → accessible HTML

## Inputs and outputs

- Input: a scan, usually dropped in `source/handwritten/ML_week_<W>_day_<D>.pdf`. If the file has
  another name, ask which week/day it is, or infer it from the content and the syllabus.
- Working copy: `source/handwritten/ML_week_<W>_day_<D>.html`
- Published (once I approve): copy the PDF to `notes/hand-written/` and the HTML to `notes/html/`
  under the same base name. Verify the copies with `md5 -q`.

## Reference

Match the established format exactly:
`../../dsai601/regression_f26/source/class_notes/LR_week_2_day_2.html`. Read its `<head>` and
`<style>` and copy them; reuse that stylesheet unchanged except for the course name. That gives
the skip link, the TOC nav, dark mode, focus styles, and the colour tokens with checked contrast.

## Transcription rules

1. Read every page of the PDF (use `pages` in chunks of ≤20). Don't skim. Every equation, arrow,
   margin note and figure is content.
2. Header: `DSAI 621 — Intro to Machine Learning · Fall 2026 · Week W, Day D`, an `<h1>` title
   that names the topic, and the standard lede saying it's an accessible transcription.
3. Structure: an `<h2>` for each major topic in the notes, plus a TOC `nav` linking to them.
4. Math: all of it in MathML, `<math alttext="spoken form">`. Use `display="block"` for display
   equations. No images of math, no LaTeX-in-HTML.
5. Figures: redraw sketches as inline SVG with `role="img"`, `aria-labelledby` pointing to a
   `<title>` and `<desc>`, a `<figcaption>`, and a `<details>` long description of what the
   sketch shows (axes, shapes, what changes). Number figures consecutively through the week.
6. Second-colour annotations and boxed asides become labelled callouts (`<div class="note">`),
   so no meaning depends on colour.
7. Faithfulness: transcribe what's written. If something looks wrong (a dropped sign or index),
   keep it and add a visible "Possible error in original" callout, then list it for me at the
   end. Ask me about illegible passages; don't guess.
8. Footer: source filename, page count, and the WCAG 2.2 AA statement.

## Check before reporting done

- Every `<math>` has `alttext`, every `<svg>` has title and desc, and headings don't skip levels.
- Open it in the browser pane (`python3 -m http.server` from the folder) and check that it
  renders in both light and dark mode.
- Report: page count transcribed, figure count, and any flagged errors or illegible spots.
