"""
Data preparation and splitting.
Location: /opt/airflow/src/diploma_project_data_split.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path


def split_data(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Auto coding string columns
    categorical_cols = ['Brand', 'Model', 'Body', 'Country', 'Condition', 'Fuel_Type',
                        'Gearbox', 'Color', 'Non_Smoker_Vehicle', 'Seller',
                        'Market_Segment', 'Classic_Vehicle']

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Setting target column and columns to remove from the dataset
    y = df['Price']

    cols_to_drop = ['Price', 'Year', 'Mileage']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Separating datasets for train, test, validation
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    datasets = {
        'train_x': X_train, 'train_y': y_train,
        'val_x': X_val, 'val_y': y_val,
        'test_x': X_test, 'test_y': y_test
    }

    for name, data in datasets.items():
        data.to_csv(f"{output_dir}/{name}.csv", index=False)

    print(f"Split complete: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    return output_dir