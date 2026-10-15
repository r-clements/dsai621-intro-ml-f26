---
name: lab
description: Write a DSAI 621 in-class lab as a Jupyter notebook with TODOs for students, plus a separate executed solutions copy. Use when asked for a lab, notebook, coding exercise or in-class activity.
---

# Lab notebook

## Files

- Student version: `notebooks/<snake_name>.ipynb` (public, so no solutions anywhere in it,
  including outputs and metadata)
- Solutions: `source/labs/<snake_name>_solutions.ipynb`
- Data: `data/` (small CSVs; check the license, and prefer ISL datasets, `sklearn.datasets` or
  simulated data)

## Reference

`../../dsai601/regression_f26/notebooks/logistic_gradient_descent.ipynb` and its `_solutions`
twin. Same tone: a title cell, a one-line course/week line, short motivation, then numbered
`---` sections.

## Shape

1. Title + `**DSAI 621 — Week W.**` + what students build and how they'll know it worked.
2. Setup cell: imports and `rng = np.random.default_rng(<seed>)`. Read data with relative paths
   (`../data/...`).
3. Implement-it sections: function stubs with docstrings, `# TODO` and
   `raise NotImplementedError`. Keep them vectorized, with no hidden helper code.
4. A check-it section that compares against sklearn (`np.allclose`, or a plot overlay), so
   students get immediate feedback.
5. An explore-it section: vary a hyperparameter and interpret it, with markdown prompts.
6. Short-answer questions in markdown at the end, tied to the lecture.

## Process

1. Write the solutions notebook first and run it end to end:
   `uv run jupyter nbconvert --to notebook --execute --inplace source/labs/<name>_solutions.ipynb`.
2. Derive the student version from it: replace solution bodies with TODO stubs, then clear all
   outputs (`uv run jupyter nbconvert --clear-output --inplace notebooks/<name>.ipynb`).
3. Check that the student notebook runs up to the first TODO without errors, and that
   `grep` finds no solution code or outputs in it.
4. If a new package is needed, `uv add <pkg>` (it goes into `pyproject.toml` for students).
