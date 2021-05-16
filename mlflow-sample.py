"""
Examples

$ mlflow ui

$ python mlflow-sample.py

open http://localhost:5000
"""

import mlflow
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import optuna
from optuna.integration.mlflow import MLflowCallback


X, y = load_iris(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
classes = np.unique(y)

mlflc = MLflowCallback(
    tracking_uri="http://localhost:5000",
    metric_name="my metric score",
)


@mlflc.track_in_mlflow()
def objective(trial):
    alpha = trial.suggest_float("alpha", 0.0, 10.0)
    clf = SGDClassifier(alpha=alpha)
    n_train_iter = 100
    mlflow.log_metric("brilliant_metric", 1.0)

    for step in range(n_train_iter):
        clf.partial_fit(X_train, y_train, classes=classes)

        intermediate_value = clf.score(X_valid, y_valid)
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return clf.score(X_valid, y_valid)


study = optuna.create_study(
    study_name="my_study",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
)
study.optimize(objective, n_trials=10, callbacks=[mlflc])
