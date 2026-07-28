
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from pathlib import Path
from datetime import datetime

def _eur_formatter(x, pos):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"


def plot_prediction_analysis(model, X_test, y_test, model_type, save_dir="/opt/airflow/reports/evaluation_reports"):
    # Function is drawing charts that shows graphically prediction error
    # Charts are being placed in evaluation_reports inside reports folder

    y_pred = model.predict(X_test)
    y_true = np.array(y_test)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Wykres 1: Predicted vs Actual
    ax1 = fig.add_subplot(gs[0])

    ax1.scatter(y_true, y_pred, alpha=0.25, s=8, color='steelblue',
                label='Predykcje', rasterized=True)

    # Linia idealna, gdzie powinny leżeć punkty
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax1.plot([min_val, max_val], [min_val, max_val],
             color='red', linewidth=1.8, linestyle='--', label='Idealna predykcja (y=x)')

    # Linia trendu modelu, gdzie faktycznie leżą predykcje
    # Odchylenie od czerwonej = systematyczny błąd modelu
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 300)
    ax1.plot(x_line, p(x_line),color='darkorange',
             linewidth=1.6, linestyle='-',
             label=f'Rzeczywisty trend modelu\n(slope={z[0]:.2f})')

    r2 = r2_score(y_true, y_pred)
    ax1.set_title(f'Predicted vs Actual — {model_type.upper()}\nR² = {r2:.4f}')
    ax1.set_xlabel('Rzeczywista cena')
    ax1.set_ylabel('Predykcja')
    ax1.xaxis.set_major_formatter(FuncFormatter(_eur_formatter))
    ax1.yaxis.set_major_formatter(FuncFormatter(_eur_formatter))
    ax1.legend(fontsize=8)

    # Wykres 2: Percentile error vs Actual price
    ax2 = fig.add_subplot(gs[1])

    err_pct = ((y_pred - y_true) / y_true) * 100

    ax2.scatter(y_true, err_pct, alpha=0.25, s=8, color='darkorange', rasterized=True)
    ax2.axhline(0, color='red', linewidth=1.8, linestyle='--', label='Zero błędu')
    ax2.axhline(20, color='gray', linewidth=0.9, linestyle=':', label='±20%')
    ax2.axhline(-20, color='gray', linewidth=0.9, linestyle=':')

    median_err = float(np.median(err_pct))
    ax2.axhline(median_err, color='green', linewidth=1.2, linestyle='--',
                label=f'Mediana: {median_err:+.1f}%')

    within_20 = np.mean(np.abs(err_pct) <= 20) * 100
    ax2.set_title(f'Błąd predykcji vs cena rzeczywista\n{within_20:.1f}% predykcji w +-20%')
    ax2.set_xlabel('Rzeczywista cena')
    ax2.set_ylabel('Błąd predykcji w %')
    ax2.set_ylim(-100, 100)
    ax2.xaxis.set_major_formatter(FuncFormatter(_eur_formatter))
    ax2.legend(fontsize=8)

    plt.suptitle(f'Analiza predykcji: {model_type.upper()}', fontsize=13, y=1.01)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{save_dir}/{model_type}_prediction_analysis_{timestamp}.png"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Wykres zapisany: {path}")
    return path