"""
Przetworzenie i podział danych
Lokalizacja: /opt/airflow/src/diploma_project_data_split.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
import json


def split_data(input_path, output_dir):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Kodowanie kolumn kategorycznych
    categorical_cols = ['Brand', 'Model', 'Body', 'Country', 'Condition', 'Fuel_Type',
                        'Gearbox', 'Color', 'Non_Smoker_Vehicle', 'Seller',
                        'Market_Segment', 'Classic_Vehicle']

    mappings = {}

    # Tworzenie kopii danych dla modelu Random forest, który wymaga zamienienia danych kategorycznych na liczbowe
    df_encoded = df.copy()

    for col in categorical_cols:
        if col in df.columns:
            # 1. Konfiguracja dla df_encoded (Random Forest)
            df_encoded[col] = df_encoded[col].astype('category')

            # Ekstrakcja mapowania z docelowych kategorii Pandas
            mappings[col] = {str(k): int(i) for i, k in enumerate(df_encoded[col].cat.categories)}

            # Zakodowanie do wartości numerycznych (tylko w skopiowanym df)
            df_encoded[col] = df_encoded[col].cat.codes

            # 2. Konfiguracja dla df (CatBoost, LightGBM, XGBoost) by wartości dla nich pozostały w formacie tekstowym.
            df[col] = df[col].astype(str)

    # Utworzenie katalogu wyjściowego i zapisanie słownika do JSON
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)

    print(f"Zapisano słownik mapowań do {output_dir}/mappings.json")

    # Ustawienie kolumny docelowej i kolumn do usunięcia
    y = df['Price']

    cols_to_drop = ['Price', 'Year', 'Mileage']

    # Utworzenie dwóch wariantów wektorów wejściowych (X)
    X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    X_encoded = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns])

    # Finalny wymiar wektora wejściowego (oba mają tyle samo kolumn)
    print(f"Final input vector dimension: {X_raw.shape[1]} columns")
    with open(f"{output_dir}/feature_info.json", "w", encoding="utf-8") as f:
        json.dump({"n_features": X_raw.shape[1], "feature_names": list(X_raw.columns)}, f, indent=4)

    # Podział zbioru danych na train, test, validation:
    # Podział danych 1:1 dla (CatBoost, LightGBM, XGBoost)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X_raw, y, test_size=0.30, random_state=42)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.50, random_state=42)

    # Podział danych ENCODED (Dla Random Forest)
    # Target (y) jest taki sam, więc ignorowane są zwracane wartości by nie powielać ich w pamięci
    X_train_enc, X_temp_enc, _, _ = train_test_split(X_encoded, y, test_size=0.30, random_state=42)
    X_val_enc, X_test_enc, _, _ = train_test_split(X_temp_enc, y_temp, test_size=0.50, random_state=42)

    # Zapis wszystkich wariantów danych do plików CSV
    datasets = {
        # Zbiory surowe dla nowoczesnych algorytmów boostingu
        'train_x_raw': X_train_raw,
        'val_x_raw': X_val_raw,
        'test_x_raw': X_test_raw,

        # Zbiory numeryczne dla tradycyjnego Random Forest
        'train_x_encoded': X_train_enc,
        'val_x_encoded': X_val_enc,
        'test_x_encoded': X_test_enc,

        # Zbiory wektora docelowego (wspólne)
        'train_y': y_train,
        'val_y': y_val,
        'test_y': y_test
    }

    for name, data in datasets.items():
        data.to_csv(f"{output_dir}/{name}.csv", index=False)

    print(f"Split complete: Train {len(X_train_raw)}, Val {len(X_val_raw)}, Test {len(X_test_raw)}")
    return output_dir