"""
This module contains all the business logic for the learning of diploma project data pipeline.
Each function is a self-contained task that can be called by the Airflow DAG.

Location: /opt/airflow/src/diploma_project_ml_tasks.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from diploma_project_evaluation import evaluate_and_save
from diploma_project_plots import plot_prediction_analysis

# Here we are setting default parameters if grid search isn't used
DEFAULT_PARAMS = {
    "randomforest": {
        "n_estimators":      300,
        "max_depth":         None,
        "min_samples_split": 2,
        "min_samples_leaf":  1,
        "max_features":      "sqrt",
    },
    "xgboost": {
        "n_estimators":    200,
        "max_depth":       6,
        "learning_rate":   0.1,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
    },
    "catboost": {
        "iterations":    500,
        "depth":         6,
        "learning_rate": 0.03,
        "l2_leaf_reg":   3,
        "subsample":     0.8,
    },
    "lightgbm": {
        "n_estimators":    500,
        "max_depth":       10,
        "learning_rate":   0.05,
        "num_leaves":      31,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      0.1,
    },
}

GRID_SEARCH_PARAMS = {
    "randomforest": {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    },
    "xgboost": {
        "n_estimators":    [100, 300],
        "max_depth":       [3, 6, 9],
        "learning_rate":   [0.01, 0.1],
        "subsample":       [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_alpha":       [0, 0.1],
        "reg_lambda":      [1, 5],
    },
    "catboost": {
        "iterations":    [200, 600],
        "depth":         [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "l2_leaf_reg":   [1, 3, 5],
        "subsample":     [0.6, 0.8],
    },
    "lightgbm": {
        "n_estimators":    [200, 500],
        "max_depth":       [-1, 10, 20],
        "learning_rate":   [0.01, 0.05],
        "num_leaves":      [31, 63, 127],
        "subsample":       [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_alpha":       [0, 0.1],
        "reg_lambda":      [0, 0.1],
    },
}



# Here we are peforming grid search with k-fold cross validation to improve models accuracy
def optimize_hyperparameters(X_train, y_train, model_type):
    print(f"GridSearch for {model_type} on {X_train.shape[0]} samples")

    model_classes = {
        "randomforest": RandomForestRegressor(random_state=42),
        "xgboost": XGBRegressor(random_state=42),
        "catboost": CatBoostRegressor(random_state=42, verbose=0),
        "lightgbm": LGBMRegressor(random_state=42, verbose=-1),
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    grid_search = GridSearchCV(
        estimator=model_classes[model_type],
        param_grid=GRID_SEARCH_PARAMS[model_type],
        cv=5,
        scoring="r2",
        verbose=2,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV R2:  {grid_search.best_score_:.4f}")

    return grid_search.best_params_, grid_search.best_score_


def compute_sample_weights(y_train):
    # Dynamiczne ustalanie wag dla modelu
    price_bins = pd.qcut(
        y_train,
        q=6,
        labels=False,
        duplicates='drop'
    )

    price_bins = pd.Series(price_bins).fillna(0).astype(int).values

    bin_counts = np.bincount(price_bins, minlength=4)
    bin_counts  = np.where(bin_counts == 0, 1, bin_counts)

    weights = 1.0 / bin_counts[price_bins]
    weights /= weights.mean()   # normalizacja: średnia waga = 1.0
    return weights

def build_and_fit_model(model_type, params, X_train, y_train):
    if model_type == "randomforest":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_type == "xgboost":
        model = XGBRegressor(**params, random_state=42)
    elif model_type == "catboost":
        model = CatBoostRegressor(**params, random_state=42, verbose=0)
    elif model_type == "lightgbm":
        model = LGBMRegressor(**params, random_state=42, verbose=-1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    #model.fit(X_train, y_train)

    weights = compute_sample_weights(y_train)
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def train_model(data_dir, model_type="randomforest", use_tuning=False):

    print(f"\nTraining {model_type} (Tuning: {use_tuning})")

    X_train = pd.read_csv(f"{data_dir}/train_x.csv")
    y_train = pd.read_csv(f"{data_dir}/train_y.csv").values.flatten()
    X_val = pd.read_csv(f"{data_dir}/val_x.csv")
    y_val = pd.read_csv(f"{data_dir}/val_y.csv").values.flatten()
    X_test = pd.read_csv(f"{data_dir}/test_x.csv")
    y_test = pd.read_csv(f"{data_dir}/test_y.csv").values.flatten()

    cv_r2 = None

    if use_tuning:
        params, cv_r2 = optimize_hyperparameters(X_train, y_train, model_type)
    else:
        params = DEFAULT_PARAMS.get(model_type)
        if params is None:
            raise ValueError(f"No default parameters defined for: {model_type}")

    print(f"Parameters: {params}")

    model = build_and_fit_model(model_type, params, X_train, y_train)

    plot_prediction_analysis(model, X_test, y_test, model_type)

    evaluate_and_save(model,
                      X_val, y_val,
                      X_test, y_test,
                      X_train,
                      model_type, params,
                      cv_r2, use_tuning)

    model_dir = "/opt/airflow/models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{model_dir}/{model_type}_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved to {model_path}\n")
    return model_path
