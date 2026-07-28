"""
Moduł zawiera logikę biznesową dla potoku danych w projekcie dyplomowym.
Każda funkcja to niezależne zadanie, które jest uruchamiane przez Airflow DAG.

Lokalizacja: /opt/airflow/src/validation_and_transformation/diploma_project_validation_tasks.py
"""

from pathlib import Path
import subprocess
import pandas as pd


# Weryfikacja obecności lokalnego pliku CSV z danymi w folderze projektu
def verify_local_data(raw_csv_path: str) -> str:
    print("Weryfikacja obecności pliku z danymi")
    raw_csv = Path(raw_csv_path)

    if not raw_csv.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego w lokalizacji: {raw_csv}")

    print(f"Plik wejściowy gotowy do użycia: {raw_csv}")
    return str(raw_csv)

# Wstępna weryfikacja struktury i jakości danych przy użyciu biblioteki pandas
def validate_data(csv_path: str, min_rows: int = 40000, required_cols: list = None) -> dict:
    print("2: Walidacja danych")
    if required_cols is None:
        required_cols = ['Make', 'Model', 'Body', 'Mileage_km', 'Price', 'Year', 'Country', 'Condition',
                         'Fuel_Type', 'Fuel_Consumption_l', 'Drivetrain', 'Gearbox', 'Gears', 'Power_hp',
                         'Engine_Size_cc', 'Cylinders', 'Seats', 'Doors', 'Color', 'Upholstery',
                         'Full_Service_History', 'Non_Smoker_Vehicle', 'Previous_Owners', 'Seller', 'Image_url']
    df = pd.read_csv(csv_path)

    # Walidacja 1: Sprawdzenie minimalnej liczby wierszy
    assert len(df) >= min_rows, f"Zbyt mało wierszy: {len(df)} < {min_rows}"
    print(f"Liczba wierszy: {len(df):,} (minimum: {min_rows:,})")

    # Walidacja 2: Sprawdzenie obecności wymaganych kolumn
    missing = [col for col in required_cols if col not in df.columns]
    assert len(missing) == 0, f"Brakujące kolumny w zbiorze: {missing}"
    print(f"Wszystkie wymagane kolumny są obecne: {required_cols}")

    # Walidacja 3: Sprawdzenie stopnia wartości null w danych
    for col in required_cols:
        null_pct = df[col].isna().sum() / len(df)
        assert null_pct < 0.5, f"Zbyt wysoki odsetek wartości null w kolumnie {col}: {null_pct:.1%}"
    print(f"Odsetek wartości null akceptowalny")

    print(f"\nPodsumowanie zbioru danych:")
    print(f"  Kolumny: {list(df.columns)}")
    print(f"  Rozmiar: {df.shape}")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns)
    }


# Uruchomienie zewnętrznego skryptu PySpark do transformacji danych
def spark_transform(csv_path: str, spark_script: str, processed_csv_path: str) -> str:
    print("3: Transformacja danych w Apache Spark")
    processed_csv = Path(processed_csv_path)
    processed_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["python", spark_script, csv_path, str(processed_csv)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Błąd Spark:", e.stderr)
        raise e

    return str(processed_csv)