"""
Rysowanie wykresów dla modeli po utworzeniu
Location: /opt/airflow/src/diploma_project_plots.py
"""

import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from datetime import datetime

def _eur_formatter(x, pos):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"

def plot_prediction_analysis(model, X_test, y_test, model_type,
                             save_dir="/opt/airflow/reports/evaluation_reports"):
    # Funkcja do rysowania wykresów do mierzenia precyzyjności modeli
    # Wykresy zapisywane są jako dwa oddzielne pliki
    y_pred = model.predict(X_test)
    y_true = np.array(y_test)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []

    # Wykres 1: Predicted vs Actual
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    ax1.scatter(y_true, y_pred, alpha=0.25, s=8, color='steelblue',
                label='Predykcje', rasterized=True)

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax1.plot([min_val, max_val], [min_val, max_val],
             color='red', linewidth=1.8, linestyle='--', label='Idealna predykcja (y=x)')

    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 300)
    ax1.plot(x_line, p(x_line), color='darkorange',
             linewidth=1.6, linestyle='-',
             label=f'Rzeczywisty trend modelu\n(slope={z[0]:.2f})')

    r2 = r2_score(y_true, y_pred)
    ax1.set_title(f'Predicted vs Actual - {model_type.upper()}\nR² = {r2:.4f}')
    ax1.set_xlabel('Rzeczywista cena')
    ax1.set_ylabel('Predykcja')
    ax1.xaxis.set_major_formatter(FuncFormatter(_eur_formatter))
    ax1.yaxis.set_major_formatter(FuncFormatter(_eur_formatter))
    ax1.legend(fontsize=9)

    path1 = f"{save_dir}/{model_type}_pred_vs_actual_{timestamp}.png"
    fig1.savefig(path1, bbox_inches='tight', dpi=150)
    plt.close(fig1)
    saved_paths.append(path1)

    # Wykres 2: Przewidywalna wartość vs Faktyczna cena
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    err_pct = ((y_pred - y_true) / y_true) * 100

    ax2.scatter(y_true, err_pct, alpha=0.25, s=8, color='darkorange', rasterized=True)
    ax2.axhline(0, color='red', linewidth=1.8, linestyle='--', label='Zero błędu')
    ax2.axhline(20, color='gray', linewidth=0.9, linestyle=':', label='±20%')
    ax2.axhline(-20, color='gray', linewidth=0.9, linestyle=':')

    median_err = float(np.median(err_pct))
    ax2.axhline(median_err, color='green', linewidth=1.2, linestyle='--',
                label=f'Mediana: {median_err:+.1f}%')

    within_20 = np.mean(np.abs(err_pct) <= 20) * 100
    ax2.set_title(f'Błąd predykcji vs cena rzeczywista — {model_type.upper()}\n{within_20:.1f}% predykcji w +-20%')
    ax2.set_xlabel('Rzeczywista cena')
    ax2.set_ylabel('Błąd predykcji w %')
    ax2.set_ylim(-100, 100)
    ax2.xaxis.set_major_formatter(FuncFormatter(_eur_formatter))
    ax2.legend(fontsize=9)

    path2 = f"{save_dir}/{model_type}_error_vs_actual_{timestamp}.png"
    fig2.savefig(path2, bbox_inches='tight', dpi=150)
    plt.close(fig2)
    saved_paths.append(path2)

    print(f"Wykresy zapisane:\n  1. {path1}\n  2. {path2}")
    return tuple(saved_paths)

# Funkcje krzywej uczenia
_LEARNING_CURVE_EXTRACTORS = {
    "xgboost":  lambda m: (m.evals_result()['validation_0']['rmse'],
                            m.evals_result()['validation_1']['rmse']),
    "lightgbm": lambda m: (m.evals_result_['training']['rmse'],
                            m.evals_result_['valid_1']['rmse']),
    "catboost": lambda m: (m.get_evals_result()['learn']['RMSE'],
                            m.get_evals_result()['validation']['RMSE']),
}

def plot_learning_curve(model, model_type, save_dir="/opt/airflow/reports/evaluation_reports"):
    extractor = _LEARNING_CURVE_EXTRACTORS.get(model_type)
    if extractor is None:
        return None

    train_curve, val_curve = extractor(model)
    best_iter = int(np.argmin(val_curve))

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(train_curve)), train_curve, color='steelblue',
             linewidth=1.5, label='Zbiór treningowy')
    ax.plot(range(len(val_curve)), val_curve, color='darkorange',
             linewidth=1.5, label='Zbiór walidacyjny')
    ax.axvline(best_iter, color='green', linestyle='--', linewidth=1.3,
               label=f'Early stopping (iteracja {best_iter})')

    ax.set_title(f'Krzywa uczenia — {model_type.upper()}')
    ax.set_xlabel('Iteracja (numer drzewa)')
    ax.set_ylabel('RMSE')
    ax.yaxis.set_major_formatter(FuncFormatter(_eur_formatter))
    ax.legend(fontsize=9)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{save_dir}/{model_type}_learning_curve_{timestamp}.png"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Krzywa uczenia zapisana: {path}")
    return path