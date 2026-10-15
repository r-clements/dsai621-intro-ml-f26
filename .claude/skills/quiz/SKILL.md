---
name: quiz
description: Write a DSAI 621 review quiz, either as an interactive self-check app in apps/ (click an answer, see why) or as a printable practice quiz. Use when asked for a quiz, review questions, practice problems or a self-check.
---

# Quiz

Ask which form if it's unclear:

- **Interactive review quiz** (default for "review quiz" / "self-check"): an app in
  `apps/topics/<topic>-quiz/`.
- **Printable practice quiz / exam practice**: LaTeX. Use the `homework` skill, file
  `source/quizzes/ML-Quiz<n>-practice.tex` + `_sol`.

## Interactive quiz

Copy the structure of `../../dsai601/regression_f26/apps/topics/mlr-fundamentals-quiz/`
(`index.html` + `app.js`) closely. Questions live in a `QUESTIONS` array:

```js
{ section: 0, type: 'mcq' | 'tf', q: "...", options: [...], correct: <index>, explain: "..." }
```

with a `SECTIONS` list naming each week/day. Register the quiz in `apps/index.html` like any app.

## Writing good questions

- Base them on what was actually taught. Read the relevant `notes/html/` and
  `source/lecture_notes/` first, and tag each question's `section` by lecture.
- Test understanding, not recall: misconceptions, "what happens if…", reading a plot,
  choosing a method. Mix ~40% true/false with MCQ.
- Distractors should be plausible errors students actually make, and never jokes.
- `explain` says why the right answer is right *and* why the tempting wrong one is wrong. It can
  cross-reference other questions.
- Vary the position of the correct answer. Check the answer distribution before finishing.
- Math in questions uses HTML entities, `<sup>` and `<sub>`, as in the reference.

## Verify

Serve and click through every question in the browser pane: each `correct` index matches the
explanation, the score tally works, and it works from the keyboard. Give me the count per section
and the answer-position distribution.
