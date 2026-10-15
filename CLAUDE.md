# DSAI 621 — Introduction to Machine Learning, Fall 2026

Instructor: Robert Clements (USF MSDS). The syllabus is `syllabus/dsai-621-fall-2026-syllabus.md`
and is the source of truth for topics, the schedule, exam dates and policies.

Textbooks: ISL (*Introduction to Statistical Learning*) and ESL (*Elements of Statistical
Learning*). Cite sections as "ISL 6.2" / "ESL 3.4". The ISL website and ESL table of contents PDFs
are in `../` (one level up).

Students have just taken DSAI 601 (Regression; course at `../../dsai601/regression_f26/`). Assume
they know OLS, the hat matrix, MLE, logistic regression, and model selection by AIC/BIC/C_p. Build
on that material and refer back to it; don't re-teach it.

## Layout

| Path | What | Public? |
|---|---|---|
| `apps/` | Interactive D3 demos, deployed to GitHub Pages from `main` | yes |
| `notes/hand-written/` | Scans of handwritten class notes, as posted | yes |
| `notes/html/` | Accessible HTML transcriptions of those notes | yes |
| `notebooks/` | Student lab notebooks (no solutions) | yes |
| `data/` | Datasets used by labs, homeworks and apps | yes |
| `syllabus/` | Syllabus `.md` (and `.pdf`) | yes |
| `tools/` | Build and check scripts | yes |
| `source/handwritten/` | Raw scans as they arrive, before transcription | **no** |
| `source/lecture_notes/` | Typeset lecture notes: `.md` + `.tex` + `.pdf` | **no** |
| `source/homeworks/` | `HWn.tex` / `HWn_sol.tex` | **no** |
| `source/quizzes/`, `source/exams/` | Quizzes and exams with solutions | **no** |
| `source/labs/` | Lab notebooks *with* solutions | **no** |

`source/` is gitignored. Solutions, exams and drafts go there and are never committed. Anything
outside `source/` is pushed to a public GitHub repo, so don't put solutions there.

## Naming

- Class notes: `ML_week_<W>_day_<D>` (e.g. `ML_week_3_day_2.pdf` / `.html`). Use `day_1_2` when
  one set of notes spans both days.
- Homeworks: `HW<n>.tex`, `HW<n>_sol.tex`. Exams: `ML-Midterm-2026.tex`, `ML-Final-2026.tex`
  (`_sol` suffix for solutions). Practice quizzes: `ML-Quiz<n>-practice.tex`.
- Apps: `apps/topics/<kebab-case-slug>/`. Notebooks: `snake_case.ipynb`, with the solutions copy
  named `<name>_solutions.ipynb` in `source/labs/`.

## Standing requirements

- **Accessibility (WCAG 2.2 AA)** applies to everything students receive. HTML notes use MathML
  with `alttext`, inline SVG figures with title, description and a long description, and nothing
  that relies on colour alone. Apps must work by keyboard and have screen-reader text. PDFs are
  built as tagged PDF/UA-2 with LuaLaTeX (see the `homework` skill).
- **Python** with numpy, pandas, scikit-learn, matplotlib/plotnine. Use `uv` for the environment
  (`uv sync`, `uv add <pkg>`), never plain pip.
- **Notation**: follow ISL/ESL. Use $n$ observations and $p$ predictors, $X$ is $n\times p$, and
  write $\hat f$, $\lambda$ for the tuning parameter, and $\mathrm{Err}$ / $\overline{\mathrm{err}}$
  for test and training error. Note: this differs from DSAI 601, which used Kutner's "$p$ includes
  the intercept". Point this out the first time it matters.
- When I share handwritten notes, the notes are the authority on content. Transcribe what's
  there, and flag anything that looks like an error in a callout instead of silently fixing it.

## Skills

Recurring tasks have project skills in `.claude/skills/`. Use them:

- `handwritten-notes`: turn a scan into accessible HTML in `notes/html/`
- `lecture-notes`: write typeset lecture notes (md → tex → pdf)
- `interactive-app`: build a new D3 demo in `apps/`
- `quiz`: write an interactive review quiz (app) or a printable practice quiz (LaTeX)
- `homework`: write a homework or exam with solutions as accessible LaTeX
- `lab`: write a Jupyter lab notebook, with a separate solutions copy

## Tooling on this machine

- LuaLaTeX from TeX Live with LaTeX kernel **2024-11-01**. Tagging uses the `testphase` key, not
  `tagging=on`; see the `homework` skill.
- No node/npx. Preview apps with `python3 -m http.server` from `apps/`.
- No pandoc.
- `gh` is installed. The GitHub repo is `r-clements/dsai621-intro-ml-f26` once it's created.
