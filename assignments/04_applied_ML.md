# Week 4 Assignments

This week's assignments build on the classifier skills from weeks 2–3 and introduce three new tools: ROC curves for threshold-independent evaluation, GridSearchCV for systematic hyperparameter tuning, and joblib for saving and loading trained models.

The mini-project asks you to build a weather classifier from scratch — the same classifier introduced in the lessons — and save it to disk. In later weeks, you will load this saved model and use it as a component inside a larger data pipeline.

As always, the warmup exercises build muscle memory for the core mechanics. Try to work through them without AI assistance. The mini-project is intentionally open-ended: there is no single right answer, and your choices and reasoning matter as much as the final output.

---

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_04/`. Inside that folder, create:

1. `warmup_04.py` — for the warmup exercises
2. `train_weather_classifier.py` — the training script for the mini-project
3. `predict_weather.py` — the prediction script for the mini-project
4. `models/` — directory containing your saved `.pkl` and metadata files
5. `outputs/` — for any plots your code saves

When finished, commit and open a PR as described in the [assignments README](https://github.com/Code-the-Dream-School/python-200/blob/main/assignments/README.md).

**Primary submission**: A link to your open GitHub PR.

---

# Part 1: Warmup Exercises

Put all warmup exercises in a single file: `warmup_04.py`. Use comments to label each section (e.g., `# --- ROC and AUC ---` and `# Q1`). Use `print()` to display all outputs and save figures to `outputs/`.

Run this setup block at the top of `warmup_04.py`:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## ROC and AUC

### ROC Question 1

Train a `LogisticRegression(max_iter=1000, random_state=42)` on the raw (unscaled) training data and a `KNeighborsClassifier(n_neighbors=5)` on the scaled training data. For each model:
- Compute predicted probabilities on the test set using `.predict_proba()`
- Compute and print the AUC score using `roc_auc_score`

Add a comment: which model has higher AUC? What does that tell you about which model better separates the two classes, independently of any threshold choice?

### ROC Question 2

Plot both ROC curves on the same axes. Label each curve with the model name and its AUC score. Add the random-classifier diagonal. Save to `outputs/roc_comparison.png`.

Add a comment: at the point on each curve where TPR = 0.80, which model has the lower FPR? What does that mean practically — if you needed to catch 80% of positives, which model would produce fewer false alarms?

### ROC Question 3

Using the logistic regression from Q1, find the threshold that achieves the highest F1 score on the test set. To do this:

1. Get `fpr`, `tpr`, and `thresholds` from `roc_curve(y_test, y_probs_lr)`.
2. For each threshold, compute `y_pred = (y_probs_lr >= threshold).astype(int)` and calculate the F1 score.
3. Print the threshold, TPR, FPR, and F1 at the optimum.

Add a comment: how does this optimal threshold compare to the default 0.5? In a real application, when would you choose a threshold lower than 0.5?

---

## GridSearchCV

### GridSearch Question 1

Build a Pipeline with a `StandardScaler` and a `LogisticRegression(max_iter=1000)`. Use `GridSearchCV` with `cv=5` and `scoring="roc_auc"` to search over `C` values `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`.

Print:
- The best `C` value
- The best CV AUC score
- The test AUC of the best estimator

Add a comment: did the grid search pick the same `C` you would have guessed by default? By how much did the test AUC change compared to the default `C=1.0`?

### GridSearch Question 2

Run a second grid search using the same Pipeline, but this time replace the `LogisticRegression` with a `DecisionTreeClassifier(random_state=42)` and search over `max_depth` values `[2, 3, 5, 8, None]`.

Print the best `max_depth` and best CV AUC, then print the test AUC.

Add a comment: compare the best AUC from Q1 (logistic regression) to this one (decision tree). Which model would you bring into further development? Is AUC the only thing you would consider?

### GridSearch Question 3

Look at the `cv_results_` from either grid search. Print the mean and standard deviation of the CV AUC for each parameter value, sorted from best to worst.

Add a comment: find a case where two parameter values have similar mean scores but different standard deviations. If you had to choose between them, which would you pick and why?

---

## joblib

### joblib Question 1

Take the best Pipeline from GridSearch Question 1 (the logistic regression pipeline). Save it to `models/warmup_model.pkl` using `joblib.dump`. Then load it back in the same script using `joblib.load` and confirm it makes identical predictions to the original:

```python
loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_lr_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")
```

Add a comment: what would break if you saved only the logistic regression model (without the scaler) and then called `.predict(X_test)` on the loaded model, where `X_test` is unscaled?

### joblib Question 2

Demonstrate a minimal version of the train/predict split. In the same `warmup_04.py`:

1. Save the best logistic regression Pipeline to `models/warmup_model.pkl` (you may have already done this in Q1).
2. Add a clearly labeled section — use a comment like `# --- Simulated prediction script ---`.
3. In that section, load the model fresh from disk and use it to predict on three manually constructed rows:

```python
import numpy as np

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])
```

Print the predicted class and the probability for each row. Add a comment: what do you expect the all-zeros row to predict? Why?

---

# Part 2: Mini-Project — Build the Weather Classifier

This project takes you through the full ML workflow — data, labels, training, evaluation, and persistence — using real weather data from the Open-Meteo API. The model you build here will be reused in a later week, when it becomes a transform step inside a cloud data pipeline.

Your work should be split across two files, as described below.

---

## File 1: `train_weather_classifier.py`

This script does all the heavy lifting: data loading, label engineering, model selection, final evaluation, and saving.

### Step 1: Fetch the Data

Use the Open-Meteo historical API to download one year of daily weather data for a location of your choice. The API is free and requires no key. Use these four daily variables:

- `temperature_2m_max` — daily high temperature (°C)
- `temperature_2m_min` — daily low temperature (°C)
- `precipitation_sum` — total precipitation (mm)
- `wind_speed_10m_max` — maximum wind speed (km/h)

An example request for Charlotte, NC:

```python
import requests
import pandas as pd

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
```

You are encouraged to pick your own city — the latitude and longitude are the only things that change. Print a summary of the dataset once it is loaded.

### Step 2: Engineer Labels

Define what "good for running" means in terms of your four numeric features. The lesson uses these thresholds as a starting point:

| Feature | "Good for running" range |
|---|---|
| `temperature_2m_max` | 7 – 26 °C (45–79°F) |
| `temperature_2m_min` | ≥ 0 °C (above freezing) |
| `precipitation_sum` | < 3.0 mm |
| `wind_speed_10m_max` | < 30 km/h |

You may adjust these thresholds to reflect your city's climate or your own preferences — just document your choices in comments. Print the class distribution after labeling, and add a comment: what fraction of days in your dataset are labeled "good for running"? Does that seem reasonable given the climate where you chose?

### Step 3: Train and Tune

Split the data into train (80%) and test (20%) sets, stratifying on the label.

Use `GridSearchCV` with a `Pipeline(StandardScaler, LogisticRegression)` to search over at least five values of `C`. Use `cv=5` and `scoring="roc_auc"`. Print:

- The best `C` value and best CV AUC
- A full classification report on the test set
- The test AUC

Then plot and save the ROC curve for the best estimator to `outputs/weather_roc.png`.

### Step 4: Reflect on Evaluation

Add a comment block (at least 4–6 sentences) addressing the following:

- What does the AUC score tell you about this model's quality? Is it surprisingly good, surprisingly bad, or about what you expected?
- Look at the precision and recall in the classification report. Which type of error (false positive vs. false negative) is more common? What would this mean in practice — would you rather the app over-recommend running or under-recommend it?
- If you were setting the threshold for a real app, would you use the default 0.5? What would you change it to and why?

### Step 5: Save the Model

Save the best Pipeline to `models/weather_classifier.pkl` using `joblib.dump`. Save a metadata file to `models/weather_classifier_metadata.json` that includes:

- Python version
- scikit-learn version
- Feature names (in order)
- Best hyperparameters from GridSearchCV
- Test AUC
- Your city (latitude and longitude)
- A brief description of the label thresholds you used

Print a confirmation message once both files are saved.

---

## File 2: `predict_weather.py`

This script simulates how the trained model would be used in production. It should have **no training code** — no `fit`, no `GridSearchCV`, no raw data loading from the API.

### Task 1: Load and Verify

Load the Pipeline from `models/weather_classifier.pkl`. Load the metadata from `models/weather_classifier_metadata.json` and print the model's key metadata (city, features, test AUC).

### Task 2: Predict on New Data

Create a DataFrame of at least five hypothetical days, covering a range of conditions: clearly good days, clearly bad days, and at least one borderline case. Use the feature names from your metadata to make sure the columns match exactly.

For each day, print:
- The four input feature values
- The predicted label (good / skip)
- The model's confidence (probability of "good for running")

### Task 3: Reflect

Add a comment block answering the following:

1. Pick the borderline case you included. What was the probability? Would you describe the model's answer as confident or uncertain? How would you handle a day where the model says 0.52?
2. The training script and the prediction script are completely separate. What would break if someone ran `predict_weather.py` before `train_weather_classifier.py`? How would you make the error message more helpful?
3. In a production system, the prediction script might run daily to classify tomorrow's weather forecast. What would need to change in `predict_weather.py` to support that? (You do not need to implement this — just describe it in a comment.)

---

# Optional Extensions

## Extension A: Try a Second City (Low)

Pick a second city with a noticeably different climate from your first (e.g., Miami if you chose Minneapolis). Pull the same year of data, apply the same label thresholds, and compare the class distributions. Which city has more "good for running" days? Does the model's AUC change when applied to that city's data? Why might it?

## Extension B: Feature Engineering (Moderate)

The four raw features work well, but there may be additional signal in derived features. Consider adding:
- A `temp_range` feature: `temperature_2m_max - temperature_2m_min`
- Seasonal indicators: month of year encoded as a numeric variable
- Lagged precipitation: whether the previous day was wet

Retrain with the expanded feature set and compare the AUC. Which added features (if any) improved the model? Update your metadata file to document the new feature list.

## Extension C: Multiple Years (Moderate)

Expand the training data to two or three years instead of one. Does more training data improve the model's AUC? Is the improvement large or small? Add a brief comment with your finding.