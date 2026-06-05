# Hyperparameter Tuning with GridSearchCV

## The Problem with Manual Tuning

Every classifier you have trained has at least one hyperparameter: `k` in KNN, `C` in logistic regression, `max_depth` in a decision tree. The week 3 assignment asked you to try a few values of `k` by hand and pick the best one. This works when there are only a handful of values to try and one hyperparameter to tune. It breaks down quickly when you have multiple hyperparameters, because you end up with a combinatorial explosion of combinations to test.

More importantly, manual tuning has a subtle data leakage risk. If you look at test-set performance to choose hyperparameters, the test set is no longer a fresh holdout — it has influenced your choices. The standard solution is a dedicated *validation set* or, more robustly, *cross-validation*: you tune on the training data alone and reserve the test set for a single final evaluation.

`GridSearchCV` automates exactly this process. You define a grid of hyperparameter values to try, and it evaluates every combination using cross-validation on the training data. At the end, it tells you which combination performed best — without ever touching the test set.

## Setting Up: The Weather Classifier

We will continue with the weather classifier from the previous lesson. Run the same data-loading and label-engineering code to get `X_train`, `X_test`, `y_train`, and `y_test`:

```python
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Fetch data
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.23,
    "longitude": -80.84,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

def label_running_day(row):
    return int(
        7 <= row["temperature_2m_max"] <= 26
        and row["temperature_2m_min"] >= 0
        and row["precipitation_sum"] < 3.0
        and row["wind_speed_10m_max"] < 30
    )

df["good_for_running"] = df.apply(label_running_day, axis=1)

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## GridSearchCV on a Single Model

The simplest case: tuning one hyperparameter on one model. Let's find the best `C` for logistic regression.

`GridSearchCV` takes a model, a parameter grid (a dictionary mapping parameter names to lists of values to try), the number of cross-validation folds, and a scoring metric. It fits a new model for every combination of parameters and every fold, then reports which combination had the best mean CV score:

```python
# Scale first — required before passing to GridSearchCV
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

param_grid = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best C:      {grid_search.best_params_['C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
```

A few things to notice:

- `scoring="roc_auc"` tells GridSearchCV to optimize for AUC — a better choice than accuracy for this problem, because AUC does not depend on a threshold choice. You can use `"f1"`, `"accuracy"`, `"precision"`, `"recall"`, or any other metric scikit-learn supports.
- `n_jobs=-1` uses all available CPU cores to run folds in parallel, which speeds things up substantially.
- `best_params_` gives the winning configuration. `best_score_` gives its mean CV score across folds.

After the search finishes, `grid_search.best_estimator_` is the model retrained on the full training set using the best parameters:

```python
best_lr = grid_search.best_estimator_

y_pred  = best_lr.predict(X_test_scaled)
y_probs = best_lr.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}")
```

## Inspecting the Full Results

`GridSearchCV` stores the complete results from every parameter combination in `cv_results_`. Converting it to a DataFrame makes it easy to inspect:

```python
results = pd.DataFrame(grid_search.cv_results_)
print(
    results[["param_C", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .to_string(index=False)
)
```

The `std_test_score` column tells you how stable the score is across folds. Two configurations with similar mean scores but different standard deviations are not equivalent — the one with lower standard deviation is more reliable.

## Pipeline + GridSearchCV

Manually scaling before passing to GridSearchCV works, but it has a hidden problem: you fit the scaler on the full training set before the CV folds are created. This means the scaler has seen the held-out fold during fitting — a mild but real form of data leakage.

The correct approach is to bundle the scaler and model into a Pipeline and pass the *unscaled* training data to `GridSearchCV`. The pipeline's `fit` method will fit the scaler and then the model, and this happens inside each fold, so the scaler never sees the fold's held-out data:

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

# Pipeline parameter names use the format "stepname__parametername"
param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"Best C:      {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
```

The `__` separator is how you reach inside a pipeline step when naming parameters: `"clf__C"` means "the `C` parameter of the step named `clf`". This pattern extends naturally to any number of steps and any combination of parameters.

`grid_search.best_estimator_` is now a fully fitted Pipeline — scaler and model together. You can call `.predict()` or `.predict_proba()` directly on raw, unscaled data:

```python
best_pipe = grid_search.best_estimator_

y_pred  = best_pipe.predict(X_test)         # no manual scaling needed
y_probs = best_pipe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}")
```

This is the correct, leak-free way to tune a model.

## Searching Across Multiple Hyperparameters

GridSearchCV evaluates every combination in the grid — so adding parameters multiplies the number of experiments. With 6 values of `C` and 4 values of `n_neighbors`, you have 10 total combinations across two models, each evaluated 5 times: 50 fits. This is manageable. With large grids, runtimes grow quickly, and a tool like `RandomizedSearchCV` (which samples from the grid rather than exhausting it) becomes attractive.

For the weather classifier, let's search across both a logistic regression and a KNN:

```python
# You can pass a list of dicts — each dict is a separate sub-grid
param_grid = [
    {
        "clf":     [LogisticRegression(max_iter=1000, random_state=42)],
        "clf__C":  [0.01, 0.1, 1.0, 10.0],
    },
    {
        "clf":              [KNeighborsClassifier()],
        "clf__n_neighbors": [3, 5, 7, 11],
    },
]

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"Best model:  {type(grid_search.best_estimator_['clf']).__name__}")
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
```

This searches logistic regression (4 values of `C`) and KNN (4 values of `k`) and selects whichever combination performs best in cross-validation.

## Choosing the Right Scoring Metric

The `scoring` parameter determines what GridSearchCV optimizes. The right choice depends on your problem:

| Situation | Recommended scoring |
|-----------|-------------------|
| Unknown threshold, comparing models | `"roc_auc"` |
| Balanced classes, threshold fixed at 0.5 | `"f1"` |
| False positives are very costly | `"precision"` |
| False negatives are very costly | `"recall"` |
| Class imbalance, multi-class | `"f1_macro"` or `"roc_auc_ovr"` |

For the weather classifier, `"roc_auc"` is the natural choice: the class balance is reasonable and we care about the model's overall discrimination ability more than its behavior at any specific threshold.

## Check for Understanding

1. You manually try `C=1.0` and get 89% accuracy on the test set, then try `C=10.0` and get 91%. You report 91%. What is wrong with this approach?

    - A. Nothing — picking the better model is always correct
    - B. The test set has now influenced your hyperparameter choice, so the 91% figure is optimistic
    - C. C should never be greater than 1.0
    - D. You need more folds to make this valid

    <details><summary><strong>Click to reveal answer</strong></summary>
    Correct answer: B. Using test-set performance to select hyperparameters is a form of data leakage — the test set is no longer an unbiased estimate of generalization performance. Use cross-validation on the training set to tune, then evaluate on the test set once.
    </details>

2. When passing a Pipeline to GridSearchCV, parameter names follow the pattern `"stepname__parametername"`. If your pipeline has a step named `"classifier"` containing a `KNeighborsClassifier`, how do you refer to the `n_neighbors` parameter in the `param_grid`?

    - A. `"n_neighbors"`
    - B. `"KNeighborsClassifier__n_neighbors"`
    - C. `"classifier__n_neighbors"`
    - D. `"classifier.n_neighbors"`

    <details><summary><strong>Click to reveal answer</strong></summary>
    Correct answer: C. Pipeline parameters are always `"stepname__parametername"`, using a double underscore as the separator.
    </details>

3. GridSearchCV with `cv=5` trains 5 models per parameter combination. You have a grid with 6 values of `C` and 4 values of `n_neighbors` across two models. How many total model fits does the search require?

    - A. 10
    - B. 24
    - C. 50
    - D. 120

    <details><summary><strong>Click to reveal answer</strong></summary>
    Correct answer: C. 6 (LR) + 4 (KNN) = 10 combinations × 5 folds = 50 fits.
    </details>

## Lesson Wrap-Up

GridSearchCV turns hyperparameter tuning from guesswork into a systematic, reproducible process. Passing a Pipeline to GridSearchCV — rather than pre-scaled arrays — ensures the preprocessing steps are properly isolated within each fold, avoiding data leakage. `best_estimator_` gives you the winning model, retrained on the full training set, ready to use.

In the next lesson, you will save this tuned pipeline to disk so it can be loaded and used in a completely separate script — the model persistence pattern that makes trained models actually deployable.
