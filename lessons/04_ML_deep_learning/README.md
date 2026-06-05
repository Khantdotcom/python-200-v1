# Week 4: Applied Machine Learning

Welcome to Week 4! Over the past two weeks you trained classifiers and evaluated them using accuracy, precision, recall, F1, and the confusion matrix. Those tools are essential — but they leave two important questions unanswered.

The first question is **how do you know if 0.5 is the right decision threshold?** Every classifier we have built outputs a probability, and we have always converted that to a prediction by asking "is the probability above 0.5?" That cutoff is a convention, not a law. Depending on what kind of errors you can tolerate, a different threshold might serve you better. This week introduces a tool — the ROC curve — that lets you visualize your classifier's behavior across every possible threshold at once, and a single summary metric, AUC, that captures model quality independently of any threshold choice.

The second question is **how do you actually use a trained model?** So far, every script we have written trains a model and immediately uses it in the same session. In a real project, training and prediction happen at different times, in different scripts, or on different machines. A model needs to be saved to disk after training and loaded later for use. This week introduces `joblib`, the standard tool for this, and shows why saving the entire preprocessing pipeline — not just the model weights — is the right approach.

In between, the week covers one more skill that pays for itself immediately: **systematic hyperparameter tuning with GridSearchCV**. Rather than guessing a value for `k` or `C` and hoping for the best, GridSearchCV automates the search by evaluating every candidate in a grid using cross-validation and surfaces the best configuration with a single call.

The running example throughout the week is a weather classifier: a model that takes daily weather features and predicts whether conditions are good for an outdoor run. You will build this classifier, tune it, and save it to disk. In later weeks, that saved file becomes a reusable pipeline component — a concrete first step toward the end-to-end data pipeline you will build by the end of the course.

## Topics

1. [ROC Curves and AUC](./01_roc_auc.md)
Beyond accuracy: understanding the threshold problem, visualizing classifier performance with ROC curves, and using AUC to compare models without committing to a single cutoff.

2. [Hyperparameter Tuning with GridSearchCV](./02_gridsearch.md)
Systematic hyperparameter search using cross-validated grid search. Covers `GridSearchCV`, `param_grid`, `best_params_`, and how to combine a Pipeline with a grid search for a clean, reproducible tuning workflow.

3. [Model Persistence with joblib](./03_model_persistence.md)
Saving a trained model to disk with `joblib.dump` and loading it in a separate script with `joblib.load`. Covers why you should serialize the full sklearn Pipeline rather than the model alone, and what to document to keep a saved model usable over time.

## Week 4 Assignments

Once you finish the lessons, head to the [assignments](../../assignments/README.md) to practice these skills and build the weather classifier you will reuse later in the course.
