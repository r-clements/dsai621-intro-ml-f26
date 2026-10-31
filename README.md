# Introduction to Machine Learning - DSAI 621

## This Week: Week 2 (Mon 10/26 & Thu 10/29)

**Quiz 1 is this Thursday, 10/29.** Bring your hand-written cheat sheet (8.5x11, one side, with your name on it). No phones, calculators or other devices.

**Topic:** Shrinkage: ridge regression, the lasso and the elastic net.

**Read:** ISL 6.2 (and 6.1 for a review of subset selection); ESL 3.4.

**Work on:**
- [Practice Problems 2](problem_sets/PP2.pdf) ([solutions](problem_sets/PP2_sol.pdf)). These aren't turned in, but the quizzes draw on them, so try each problem before you look at the solutions.
- [Practice Problems 3](problem_sets/PP3.pdf) ([solutions](problem_sets/PP3_sol.pdf)): problems 1, 2, 7 and 8.
- Notebook: [`notebooks/ridge_lasso.ipynb`](notebooks/ridge_lasso.ipynb). Run `git pull` and then `uv sync` first.
- Interactive demo: [Shrinkage: Ridge, Lasso and Elastic Net](https://robertclements.org/dsai621-intro-ml-f26/topics/shrinkage-paths/). 
- If you haven't finished them yet, the Week 1 problems are below under Past weeks.

<details>
<summary>Past weeks</summary>

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
| 1 | Bias-variance tradeoff, curse of dimensionality | 2 | 2 |
| 2 | Ridge, lasso, elastic net | 3 | 6 |
| 3 | LDA, QDA, naive Bayes vs. logistic regression | 4 | 4 |
| 4 | Basis expansions: polynomial regression, splines | 5 | 7 |
| 5 | Kernel smoothing / local regression | 6 | 7 |
| 6 | Trees (CART) | 9.2 | 8 |
| 7 | Bagging and Random Forests | 8.7, 15 | 8 |
| 8 | Boosting | 10 | 8 |
| 9 | Probability calibration | — | — |
| 10 | Clustering: k-means, hierarchical | 14.3 | 12 |
| 11 | PCA + recommender systems intro | 14.5 | 12 |

## Quizzes

Six in-class quizzes (check the Google calendar for most up-to-date quiz dates), plus an optional cumulative bonus quiz during finals week. See the [syllabus](syllabus/dsai-621-fall-2026-syllabus.md) for details.
