---
name: homework
description: Write a DSAI 621 homework, practice quiz, or exam (plus its solutions) as accessible tagged-PDF LaTeX and build it. Use when asked for a homework, problem set, exam, midterm, final, or printable quiz.
---

# Homework / exam (LaTeX, accessible PDF)

## Files

- Homework: `source/homeworks/HW<n>.tex` and `HW<n>_sol.tex`
- Exams: `source/exams/ML-Midterm-2026.tex`, `ML-Final-2026.tex` (+ `_sol`)
- Printable practice quizzes: `source/quizzes/ML-Quiz<n>-practice.tex` (+ `_sol`)

Start from `template.tex` in this skill folder. It is a tested PDF/UA-2 preamble. Write the
student version first, then copy it to `_sol` and uncomment `\solutionstrue`, so both files share
one body. Update `pdftitle`, `\title` and the due date in both.

## Preamble rules (these are what keep the tagging working)

- Keep `\DocumentMetadata{...}` first and `unicode-math` loaded. Without them there's no MathML.
- Use `\[...\]` for display math, never `$$...$$`.
- Don't add `subfigure`, `fontenc[OT1]`, `dsfont`, `bm`, `mdframed` or `framed`: they break
  tagging or text extraction. For bold vectors, `unicode-math` provides `\symbf`.
- Every `\includegraphics` needs `alt={...}` describing what the figure shows.
- Put `\tagpdfsetup{table/header-rows={1}}` before any `tabular` that has a header row.

## Content conventions

- Mix theory (derivations, proofs) with computation (sklearn on a dataset in `data/`). Point to
  the matching lecture and ISL/ESL section in each problem.
- Homework is due Sunday 11:59pm on Gradescope. Ask me for the date if it isn't obvious from the
  syllabus schedule.
- Exams are closed book, with one 8.5x11 cheat sheet and no technology. Keep exam questions
  answerable by hand, and include space or instructions for showing work.
- Solutions should show the full reasoning a grader needs, not just final answers.

## Build and check

```bash
tools/build_pdf.sh source/homeworks/HW1.tex source/homeworks/HW1_sol.tex
```

This runs two LuaLaTeX passes and then the structural accessibility check. Fix any errors or
FAILs before reporting done. Tell me that veraPDF (`verapdf -f ua2`) is the authoritative check
if it's installed.
