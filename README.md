# Introduction to Machine Learning - DSAI 621

## This Week: Week 3 (Mon 11/2 & Thu 11/5)

**Quiz 2 is Thursday 11/5 at 3:40PM.** It covers ridge and the lasso, logistic regression, and LDA. Bring your hand-written cheat sheet (8.5x11, one side, with your name on it). No phones, calculators or other devices.

**Topics:** LDA, QDA and naive Bayes; polynomial regression and splines.

**Read:** ISL 4.4–4.5 and 7.1–7.4; ESL 4.3, 6.6.3 and 5.1–5.2.

**Work on:**
- [Quiz 2 practice questions](problem_sets/ML-Quiz2-practice.pdf)
- Finish [Practice Problems 3](problem_sets/PP3.pdf) ([solutions](problem_sets/PP3_sol.pdf))
- [Practice Problems 4](problem_sets/PP4.pdf) ([solutions](problem_sets/PP4_sol.pdf))
- Practice Problems 5 (released Friday 11/6)

The practice problems aren't turned in, but the quizzes draw on them, so try each problem before you look at the solutions.

**Notebooks** (run `git pull` and then `uv sync` first):
- Finish [`notebooks/lda_vs_logistic.ipynb`](notebooks/lda_vs_logistic.ipynb) and [`notebooks/qda_vs_naive_bayes.ipynb`](notebooks/qda_vs_naive_bayes.ipynb)
- Splines notebook (released before Thursday's class)

**Interactive demos:**
- [LDA, QDA, Naive Bayes and Logistic Regression](https://robertclements.org/dsai621-intro-ml-f26/topics/lda-qda-boundaries/)
- [Basis Expansions: Polynomials, Step Functions and Splines](https://robertclements.org/dsai621-intro-ml-f26/topics/basis-expansions/)
- For Quiz 2 review: [Shrinkage: Ridge, Lasso and Elastic Net](https://robertclements.org/dsai621-intro-ml-f26/topics/shrinkage-paths/)
- [All demos](https://robertclements.org/dsai621-intro-ml-f26/)

<details>
<summary>Past weeks</summary>

### Week 2 (Mon 10/26 & Thu 10/29)

**Topics:** The curse of dimensionality, ridge regression and the lasso; classification with logistic regression, generative models and LDA.

**Read:** ISL 6.2 and 4.4; ESL 3.4 and 4.3.

- Class notes, Day 1: [handwritten](notes/hand-written/ML_week_2_day_1.pdf) · [accessible HTML](notes/html/ML_week_2_day_1.html)
- Class notes, Day 2: [handwritten](notes/hand-written/ML_week_2_day_2.pdf) · [accessible HTML](notes/html/ML_week_2_day_2.html)
- [Practice Problems 2](problem_sets/PP2.pdf) ([solutions](problem_sets/PP2_sol.pdf))
- Notebook: [`notebooks/ridge_lasso.ipynb`](notebooks/ridge_lasso.ipynb)
- Interactive demo: [Shrinkage: Ridge, Lasso and Elastic Net](https://robertclements.org/dsai621-intro-ml-f26/topics/shrinkage-paths/)

### Week 1 (Thu 10/22)

**Topic:** What is machine learning? The bias-variance tradeoff and the curse of dimensionality.

**Read:** ISL 2.1–2.2 and 3.5; ESL 2.1–2.5 and 7.2–7.3.

- Class notes: [handwritten](notes/hand-written/ML_week_1_day_1.pdf) · [accessible HTML](notes/html/ML_week_1_day_1.html)
- [Practice Problems 1](problem_sets/PP1.pdf) ([solutions](problem_sets/PP1_sol.pdf))
- Interactive demo: [Bias–Variance Tradeoff with KNN Regression](https://robertclements.org/dsai621-intro-ml-f26/topics/bias-variance-knn/)

</details>

## Robert Clements

**Email**: rclements [at] usfca [dot] edu

**Class Time**: Mon/Thurs 10:00AM-12:00PM or 1:00PM-3:00PM in SFD 529

**Office Hours**: Mon 9:00AM-9:50AM; Thur 3:00PM-3:50PM

**Textbooks (both free)**: [Introduction to Statistical Learning](https://www.statlearning.com) (ISL) and [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) (ESL)

**Syllabus**: [Syllabus link](syllabus/dsai-621-fall-2026-syllabus.md)


## Using This Repo

```
git clone https://github.com/r-clements/dsai621-intro-ml-f26.git

# cd into the repo

uv sync
```

## What's Here

- `notes/`: class notes (handwritten scans plus accessible HTML versions)
- `problem_sets/`: weekly practice problems and solutions
- `notebooks/`: lab notebooks
- `data/`: datasets
- `apps/`: interactive demos, also published on GitHub Pages

## Material Covered

| # | Topic | ESL Ch. | ISL Ch. |
|---|---|---|---|
| 1 | Bias-variance tradeoff, curse of dimensionality | 2, 7 | 2, 3 |
| 2 | Ridge, lasso, elastic net | 3 | 6 |
| 3 | LDA, QDA, naive Bayes vs. logistic regression | 4 | 4 |
| 4 | Basis expansions: polynomial regression, splines | 5 | 7 |
| 5 | Smoothing splines | 5 | 7 |
| 6 | Generalized additive models (GAMs) | 9 | 7 |
| 7 | Trees (CART) | 9 | 8 |
| 8 | Bagging and Random Forests | 8, 15 | 8 |
| 9 | Boosting | 10 | 8 |
| 10 | Probability calibration | — | — |
| 11 | Clustering: k-means, hierarchical | 14.3 | 12 |

## Quizzes

Six in-class quizzes (check the Google calendar for most up-to-date quiz dates), plus an optional cumulative bonus quiz during finals week. See the [syllabus](syllabus/dsai-621-fall-2026-syllabus.md) for details.
