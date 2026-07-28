"""
Orkiestracja: Weryfikacja lokalnego zbioru -> Walidacja Pandas -> Transformacja Spark -> Podział Danych -> Trening Modeli ML

Zbiór danych: EU Used Cars (30k+ wierszy)
Źródło: Lokalny folder środowiska

Lokalizacja DAG: /opt/airflow/dags/diploma_project_pipeline.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow.decorators import dag, task

sys.path.insert(0, '/opt/airflow/src/validation_and_transformation')
sys.path.insert(0, '/opt/airflow/src/data_split_and_model_training')

from diploma_project_validation_tasks import verify_local_data, validate_data, spark_transform
from diploma_project_data_split import split_data
from diploma_project_ml_tasks import train_model

# Definicja ścieżek
DATA_DIR = Path("/opt/airflow/data")
RAW_CSV = DATA_DIR / "raw" / "fullGas.csv"
PROCESSED_CSV = DATA_DIR / "processed" / "fullGas_processed.csv"
SPARK_SCRIPT = "/opt/airflow/src/validation_and_transformation/diploma_project_transform.py"
ML_READY_DIR = str(DATA_DIR / "ml_splits")

# Stałe używane we wstępnej weryfikacji
MIN_ROWS = 40000
REQUIRED_COLUMNS = ['Make', 'Model', 'Body', 'Mileage_km', 'Price', 'Year', 'Country', 'Condition',
                    'Fuel_Type', 'Gearbox', 'Power_hp', "Engine_Size_cc", 'Cylinders',
                    'Seats', 'Doors', 'Color', 'Non_Smoker_Vehicle', 'Seller']

MODEL_CONFIG = {
    "randomforest": {"enabled": True, "tune": False},
    "xgboost":      {"enabled": True, "tune": False},
    "catboost":     {"enabled": True, "tune": False},
    "lightgbm":     {"enabled": True, "tune": False},
}


default_args = {
    'owner': 'student',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='diploma_project_pipeline',
    default_args=default_args,
    description='End-to-end Diploma project ml pipeline: Validation, Transformation, Data splitting, Model training.',
    schedule="@weekly",
    start_date=datetime(2026, 4, 1),
    dagrun_timeout=timedelta(hours=24),
    catchup=False,
    tags=['pro2d', 'diploma', 'spark', 'random_forest', 'xgboost', 'catboost', 'lightgbm'],
)
def diploma_project_pipeline():

    # Sprawdzenie obecności pliku wejściowego w podanej ścieżce lokalnej
    @task()
    def verify_input_file():
        return verify_local_data(raw_csv_path=str(RAW_CSV))

    # Walidacja danych pod kątem kompletności i struktury
    @task()
    def validate(csv_path: str):
        return validate_data(csv_path=csv_path, min_rows=MIN_ROWS, required_cols=REQUIRED_COLUMNS)

    # Przetworzenie danych narzędziem Apache Spark
    @task()
    def transform(csv_path: str):
        return spark_transform(csv_path=csv_path, spark_script=SPARK_SCRIPT, processed_csv_path=str(PROCESSED_CSV))

    # Podział gotowych danych na zbiory uczące/testowe/walidacyjne
    @task()
    def prepare_splits(processed_path: str):
        return split_data(processed_path, ML_READY_DIR)

    # Równoległe treningi i ocena modeli
    @task()
    def train_task(split_path: str, model_type: str):
        config = MODEL_CONFIG.get(model_type, {"enabled": False, "tune": False})

        if not config["enabled"]:
            print(f"Skipping {model_type} training")
            return None

        return train_model(
            data_dir=split_path,
            model_type=model_type,
            use_tuning=config["tune"]  # Przekazujemy flagę z naszego słownika
        )

    # Ustalanie logiki zależności przepływu
    raw_data = verify_input_file()
    validation = validate(raw_data)

    transformed_data = transform(raw_data)
    transformed_data.set_upstream(validation)

    splits = prepare_splits(transformed_data)

    model_tasks = [
        train_task.override(task_id=f"train_{mtype}")(splits, mtype)
        for mtype in MODEL_CONFIG.keys()
    ]

diploma_project_pipeline()