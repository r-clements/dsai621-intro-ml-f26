---
name: lecture-notes
description: Write typeset DSAI 621 lecture notes for a topic (markdown draft, then LaTeX, then PDF) with instructor commentary boxes. Use when asked for lecture notes, a lecture outline, or notes to teach from.
---

# Lecture notes

## Files

`source/lecture_notes/<topic_slug>.md`, then `.tex`, then `.pdf`. These are my teaching notes, so
they stay in `source/`. Work in that order: draft the `.md` and get my feedback on it before
converting to `.tex`.

## Reference

Match `../../dsai601/regression_f26/source/new_notes/model_selection.md` and `.tex`. Read both
before starting; the `.tex` preamble (commentary box, `thmbox`, accent colour) is reused as-is.

## Structure

1. Title and an italic one-paragraph hook: the question this lecture answers.
2. **Notation** paragraph (ISL/ESL conventions; flag where DSAI 601/Kutner differed).
3. **Pairs with:** links to the earlier lectures this builds on, including DSAI 601 ones.
4. Numbered sections. Each one gives the idea in words, then the math, then a picture or
   example, then a short sklearn snippet when relevant.
5. `> 💬 **Commentary:**` blocks (in the .tex, `\Commentary{...}`) for what I should *say* or
   do at the board: motivation, where students get stuck, what to draw, timing.
6. Derivations are complete, with no "it can be shown". Use `thmbox` for lemmas and results.
7. End with a summary, common misconceptions, and ISL/ESL readings by section.

## Content standards

- Correctness first: check derivations step by step, and check numeric examples by running them.
- Cite ISL/ESL sections precisely. If unsure of a section number, check the TOC PDFs in `../`.
- Suggest which `apps/` demo, lab or HW problem pairs with each section. Propose a new app if
  a section would benefit from one.

## Build

```bash
cd source/lecture_notes && lualatex <topic_slug>.tex && lualatex <topic_slug>.tex
```

Clear all errors and undefined references. These notes are for me, so tagged-PDF isn't required.
If a student needs them, convert them through the `homework` preamble rules.
