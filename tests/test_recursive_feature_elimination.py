import numpy as np
import pytest
from feature_engine.selection import RecursiveFeatureElimination
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )


@pytest.fixture(scope="module")
def experiment():
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    selector = RecursiveFeatureElimination(
        estimator=make_logistic_model(),
        scoring="roc_auc",
        cv=3,
        threshold=0.001,
    )
    # Fit feature selection on the training split only.
    selector.fit(X_train, y_train)
    selected_features = list(selector.transform(X_train).columns)
    dropped_features = selector.features_to_drop_

    baseline_model = make_logistic_model().fit(X_train, y_train)
    selected_model = make_logistic_model().fit(
        X_train[selected_features], y_train
    )
    baseline_probabilities = baseline_model.predict_proba(X_test)[:, 1]
    selected_probabilities = selected_model.predict_proba(
        X_test[selected_features]
    )[:, 1]

    return {
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "selector": selector,
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "baseline_probabilities": baseline_probabilities,
        "selected_probabilities": selected_probabilities,
        "baseline_auc": roc_auc_score(y_test, baseline_probabilities),
        "selected_auc": roc_auc_score(y_test, selected_probabilities),
    }


def test_dataset_and_rfe_use_training_split_only(experiment):
    assert experiment["X"].shape[1] == 30
    assert len(experiment["X_train"]) + len(experiment["X_test"]) == len(
        experiment["X"]
    )
    assert set(experiment["X_train"].index).isdisjoint(
        experiment["X_test"].index
    )
    assert list(experiment["selector"].feature_names_in_) == list(
        experiment["X_train"].columns
    )


def test_rfe_selects_a_nonempty_partition_of_original_features(experiment):
    original_features = set(experiment["X"].columns)
    selected = set(experiment["selected_features"])
    dropped = set(experiment["dropped_features"])

    assert 0 < len(selected) < len(original_features)
    assert selected <= original_features
    assert dropped <= original_features
    assert selected.isdisjoint(dropped)
    assert selected | dropped == original_features


def test_models_predict_valid_test_probabilities(experiment):
    for key in ("baseline_probabilities", "selected_probabilities"):
        probabilities = experiment[key]

        assert probabilities.shape == (len(experiment["X_test"]),)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_selected_model_roc_auc_is_reasonable_against_baseline(experiment):
    baseline_auc = experiment["baseline_auc"]
    selected_auc = experiment["selected_auc"]

    assert 0 <= baseline_auc <= 1
    assert 0 <= selected_auc <= 1
    # Match the notebook's stated tolerance for maintaining test performance.
    assert selected_auc >= baseline_auc - 0.01
