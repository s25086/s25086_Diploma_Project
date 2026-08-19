"""
This module contains all the business logic for the learning of diploma project data pipeline.
Each function is a self-contained task that can be called by the Airflow DAG.

Location: /opt/airflow/src/diploma_project_ml_tasks.py
"""

import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

from diploma_project_evaluation import evaluate_and_save
from diploma_project_plots import plot_prediction_analysis, plot_learning_curve

# Domyślne parametry dla każdego z modeli
DEFAULT_PARAMS = {
    "randomforest": {
        "n_estimators":      300,
        "max_depth":         None,
        "min_samples_split": 2,
        "min_samples_leaf":  1,
        "max_features":      "sqrt",
    },
    "xgboost": {
        "n_estimators":    1000,
        "max_depth":       6,
        "learning_rate":   0.1,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
    },
    "catboost": {
        "iterations":    1000,
        "depth":         6,
        "learning_rate": 0.03,
        "l2_leaf_reg":   3,
        "subsample":     0.8,
    },
    "lightgbm": {
        "n_estimators":    1000,
        "max_depth":       10,
        "learning_rate":   0.05,
        "num_leaves":      31,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      0.1,
    },
}

# Wartości parametrów zdefiniowane do wyszukiwania metodą grid search
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

# Early stopping: liczba kolejnych drzew/iteracji bez poprawy wyniku na zbiorze walidacyjnym,
# po której trening jest przerywany.
EARLY_STOPPING_ROUNDS = 50

# Early stopping tylko dla drzew budowanych iteracyjnie, w random forest drzewa budowane
# są niezależnie od siebie
BOOSTING_MODELS = {"xgboost", "catboost", "lightgbm"}



# Implementacja wyszukiwania grid search dla każdego z modeli,
# w celu polepszenia zdolności przewidywania każdego z modeli
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

def build_and_fit_model(model_type, params, X_train, y_train, X_val=None, y_val=None):
    # Sprawdzenie czy zbiór walidacyjny istnieje
    use_early_stopping = (
            model_type in BOOSTING_MODELS and X_val is not None and y_val is not None
    )

    if model_type == "randomforest":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_type == "xgboost":
        model = XGBRegressor(
            **params,
            random_state=42,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if use_early_stopping else None,
            eval_metric="rmse",
        )
    elif model_type == "catboost":
        model = CatBoostRegressor(**params, random_state=42, verbose=0)
    elif model_type == "lightgbm":
        model = LGBMRegressor(**params, random_state=42, verbose=-1)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    fit_kwargs = {}

    if use_early_stopping:
        if model_type == "xgboost":
            # Ostatni zbiór podany w eval_set jest tym, na podstawie którego
            # XGBoost decyduje o zatrzymaniu treningu - zbiór treningowy dołączamy
            # tylko po to, by później narysować krzywą uczenia (train vs val),
            # zbiór walidacyjny musi zostać ostatni.
            fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]
            fit_kwargs["verbose"] = False
        elif model_type == "catboost":
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
            fit_kwargs["verbose"] = False
        elif model_type == "lightgbm":
            fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]
            fit_kwargs["eval_metric"] = "rmse"
            fit_kwargs["callbacks"] = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]

    model.fit(X_train, y_train, **fit_kwargs)
    return model

def get_early_stopping_summary(model, model_type, params):
    # Zwraca ile drzew faktycznie wykorzystano w porównaniu do zaplanowanej
    # maksymalnej liczby (n_estimators / iterations). Przydatne do opisania
    # efektu early stopping w rozdziale 1.6/1.7 pracy.
    if model_type not in BOOSTING_MODELS:
        return None

    if model_type == "xgboost":
        requested = params.get("n_estimators")
        best_iter = getattr(model, "best_iteration", None)
        actual = (best_iter + 1) if best_iter is not None else requested
    elif model_type == "lightgbm":
        requested = params.get("n_estimators")
        actual = getattr(model, "best_iteration_", None)
        actual = actual if actual is not None else requested
    elif model_type == "catboost":
        requested = params.get("iterations")
        best_iter = model.get_best_iteration()
        actual = (best_iter + 1) if best_iter is not None else requested
    else:
        return None

    if requested is None or actual is None:
        return None

    return {
        "patience_rounds": EARLY_STOPPING_ROUNDS,
        "requested_rounds": int(requested),
        "actual_rounds": int(actual),
        "stopped_early": bool(actual < requested),
    }


# Trening i zapis wyników modeli
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

    model = build_and_fit_model(model_type, params, X_train, y_train, X_val, y_val)

    early_stopping_info = get_early_stopping_summary(model, model_type, params)
    if early_stopping_info:
        print(f"Early stopping: {early_stopping_info['actual_rounds']}/"
              f"{early_stopping_info['requested_rounds']} drzew "
              f"(patience={early_stopping_info['patience_rounds']}, "
              f"zatrzymano wcześniej: {'tak' if early_stopping_info['stopped_early'] else 'nie'})")
        plot_learning_curve(model, model_type)

    plot_prediction_analysis(model, X_test, y_test, model_type)

    evaluate_and_save(model,
                      X_val, y_val,
                      X_test, y_test,
                      X_train,
                      model_type, params,
                      cv_r2, use_tuning,
                      early_stopping_info=early_stopping_info)


    model_dir = "/opt/airflow/models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{model_dir}/{model_type}_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved to {model_path}\n")
    return model_path
