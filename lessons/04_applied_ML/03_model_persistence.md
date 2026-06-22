# Model Persistence with joblib

## Training Once, Predicting Many Times

Every script you have written so far follows the same pattern: load data, train a model, evaluate it, done. Training and prediction happen in the same run. This is fine for learning, but it is not how models work in production.

In a real data pipeline, training and prediction are separated — sometimes by hours, sometimes by months:

- A model is trained offline, on historical data, at regular intervals (weekly, monthly).
- The trained model is saved to disk.
- A separate script — the one that actually runs in production — loads the saved model and uses it to make predictions on new data.

This separation is fundamental. It means the cost of training (potentially slow, resource-intensive) is paid once, while the cost of prediction (fast, lightweight) is paid many times. It also means a model can be shared across scripts, deployed to different environments, or rolled back to an earlier version if something goes wrong.

`joblib` is the standard library for saving and loading scikit-learn models. It is already installed when you install scikit-learn.

## Saving a Model: joblib.dump

`joblib.dump` takes two arguments: the object to save and the path where it should be written.

```python
import joblib
import os

os.makedirs("models", exist_ok=True)

# Assuming best_pipe is the fitted Pipeline from the GridSearchCV lesson
joblib.dump(best_pipe, "models/weather_classifier.pkl")
print("Model saved.")
```

The `.pkl` extension is conventional — it stands for "pickle", the underlying serialization format. The file is a binary snapshot of the model object: all the learned weights, the scaler's mean and standard deviation, and the pipeline structure.

## Loading a Model: joblib.load

`joblib.load` takes the file path and returns the original Python object:

```python
clf = joblib.load("models/weather_classifier.pkl")
print(type(clf))
```

The loaded object is ready to use immediately — no retraining required.

## Why Save the Full Pipeline, Not Just the Model

This is a common mistake and worth understanding clearly.

A trained logistic regression model knows its coefficients, but it does not know the scale of the data it was trained on. If you save only the model and later call `.predict()` on raw, unscaled data, the predictions will be wrong — silently wrong, with no error message. The model does not know it received unscaled inputs; it just produces bad outputs.

The scaler is learned state too: it remembers the training set's mean and standard deviation for each feature. Those values must be applied to any new data in exactly the same way they were applied to the training data. If you throw the scaler away after training, you have no way to reproduce that transformation consistently.

Saving a full Pipeline solves this cleanly. The pipeline bundles the scaler and the model into a single object. When you call `.predict()` on the pipeline with raw data, it automatically applies the scaler first, then the model — in the same order they were trained. One file, no manual bookkeeping.

```python
# This is what you want — full pipeline
joblib.dump(best_pipe, "models/weather_classifier.pkl")

# This is what you want to avoid — model only, no scaler
# joblib.dump(best_pipe["clf"], "models/lr_only.pkl")
```

## Using a Loaded Model in a Separate Script

Here is what the prediction side looks like — a script that has no knowledge of how the model was trained:

```python
import joblib
import pandas as pd

# Load the pipeline
clf = joblib.load("models/weather_classifier.pkl")

# Create a DataFrame with the same features the model was trained on
new_days = pd.DataFrame({
    "temperature_2m_max": [22.0, 5.0, 30.0, 14.0],
    "temperature_2m_min": [12.0, -3.0, 20.0, 8.0],
    "precipitation_sum":  [0.5,   0.0,  8.0, 1.0],
    "wind_speed_10m_max": [15.0,  10.0, 45.0, 20.0],
})

# Predict — the pipeline applies scaling automatically
predictions = clf.predict(new_days)
probabilities = clf.predict_proba(new_days)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "good" if pred == 1 else "skip"
    print(f"Day {i+1}: {label} ({prob:.2f} probability)")
```

Notice what the prediction script does *not* contain: no StandardScaler, no `fit_transform`, no training data. It simply loads the file and calls `.predict()`. The pipeline handles everything else.

## The Versioning Problem

Serialized models have a limitation worth knowing: they are tied to the Python and scikit-learn versions that created them. A `.pkl` file created with scikit-learn 1.3 may fail to load in scikit-learn 1.5.

In production, this is managed by pinning dependency versions (a `requirements.txt` or a Docker image that specifies exact library versions). For this course, what matters is that you document the versions used when you save a model. A simple metadata file alongside the `.pkl` is enough:

```python
import sklearn
import sys

metadata = {
    "python_version":   sys.version,
    "sklearn_version":  sklearn.__version__,
    "features":         FEATURES,
    "label":            "good_for_running",
    "trained_on":       "2023 Open-Meteo data, Charlotte NC",
    "label_thresholds": {
        "temperature_2m_max": "7–26°C",
        "temperature_2m_min": ">= 0°C",
        "precipitation_sum":  "< 3.0 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

import json
with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved.")
```

This metadata answers three questions anyone picking up this model will have: what library versions it needs, what input features it expects (in what order), and what it was trained on.

## A Complete Save-and-Load Workflow

Putting it all together: here is the full sequence from training to saving to loading and predicting.

### Training script (run once)

```python
import requests
import pandas as pd
import joblib
import json
import sklearn
import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

os.makedirs("models", exist_ok=True)

# --- Load and label data ---
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

# --- Tune and train ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])
param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_pipe = grid_search.best_estimator_
y_probs = best_pipe.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs)
print(f"Best C: {grid_search.best_params_['clf__C']}")
print(f"Test AUC: {test_auc:.3f}")

# --- Save model and metadata ---
joblib.dump(best_pipe, "models/weather_classifier.pkl")

metadata = {
    "python_version":  sys.version,
    "sklearn_version": sklearn.__version__,
    "features":        FEATURES,
    "label":           "good_for_running",
    "best_params":     grid_search.best_params_,
    "test_auc":        round(test_auc, 4),
    "trained_on":      "2023 Open-Meteo, Charlotte NC (lat 35.23, lon -80.84)",
}
with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model and metadata saved to models/")
```

### Prediction script (run many times)

```python
import joblib
import pandas as pd

clf = joblib.load("models/weather_classifier.pkl")

new_days = pd.DataFrame({
    "temperature_2m_max": [18.0, 2.0, 28.0],
    "temperature_2m_min": [10.0, -2.0, 19.0],
    "precipitation_sum":  [0.0,   0.0, 12.0],
    "wind_speed_10m_max": [18.0,  8.0, 35.0],
})

preds = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Day {i+1}: {label}  (confidence: {prob:.2f})")
```

## Check for Understanding

1. You save only the trained `LogisticRegression` model (not the `StandardScaler`) and later load it to make predictions on raw data. What happens?

    - A. The model raises an error because the scale is wrong
    - B. The model produces predictions, but they are likely wrong because the input scale does not match training
    - C. The model automatically rescales the input
    - D. Nothing changes because logistic regression does not need scaling

    <details><summary><strong>Click to reveal answer</strong></summary>
    Correct answer: B. Scikit-learn models do not validate that inputs are scaled correctly — they simply compute with whatever numbers you give them. If the scale differs from training, the predictions will be silently wrong. Saving the full Pipeline avoids this.
    </details>

2. A colleague loads your saved Pipeline and calls `clf.predict(X_new)` on raw, unscaled data. What does the Pipeline do?

    - A. Raises an error: the pipeline needs pre-scaled data
    - B. Applies the scaler, then the model, automatically
    - C. Applies only the model, skipping the scaler
    - D. Returns probabilities rather than hard predictions

    <details><summary><strong>Click to reveal answer</strong></summary>
    Correct answer: B. Calling `.predict()` on a Pipeline runs each step in order: the scaler transforms the raw data, and the transformed data is passed to the model. Your colleague does not need to know anything about the preprocessing.
    </details>

3. Which of the following should you document alongside a saved model file?

    - A. The Python and scikit-learn versions used during training
    - B. The feature names and their expected order
    - C. What the model was trained on (data source, date range)
    - D. All of the above

    <details><summary><strong>Click to reveal answer</strong></summary>
    Correct answer: D. All three. The version information ensures the file can be loaded in the target environment. The feature list ensures inputs are prepared correctly. The training provenance records what the model knows and when.
    </details>

## Lesson Wrap-Up

`joblib` is the standard way to save and load scikit-learn models. Saving the full Pipeline — scaler and model together — ensures predictions on raw data are always correct without any manual preprocessing. A metadata file alongside the model records the context needed to use it safely.

The weather classifier you have now trained, tuned, and saved is more than a homework exercise. In weeks 9 and 10, it will become a live component of a data pipeline: a transform step that takes raw weather records and produces predictions. The `.pkl` file you created today is what makes that possible.
