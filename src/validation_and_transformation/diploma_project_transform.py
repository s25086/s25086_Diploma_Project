"""
Wykonuje czyszczenie danych, rzutowanie typów oraz inżynierię cech na danych o samochodach.

Użycie:
    python diploma_project_transform.py <input_csv> <output_csv>

Lokalizacja: /opt/airflow/src/validation_and_transformation/diploma_project_transform.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, floor, avg, count
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql import functions as F

import sys
import os
from datetime import datetime

def main():
    if len(sys.argv) != 3:
        print("Usage: python diploma_project_transform.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("Rozpoczęcie transformacji w Sparku")
    print(f"Wejście:  {input_path}")
    print(f"Wyjście: {output_path}")


    # Tworzenie sesji pyspark
    spark = SparkSession.builder \
        .appName("Diploma_project_Transform") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    # Ograniczenie logów do poziomu ostrzeżeń, aby poprawić czytelność w konsoli
    spark.sparkContext.setLogLevel("WARN")

    print("\nUtworzono sesję Spark")


    # Wczytywanie pliku CSV
    print("\nWczytywanie pliku CSV")
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(input_path)

    initial_count = df.count()
    print(f"Wczytano {initial_count:,} wierszy")

    print("\nPoczątkowy schemat danych:")
    df.printSchema()

    print("\nPrzykładowe dane (pierwsze 3 wiersze):")
    df.show(3, truncate=False)


    print("Transformacja 1. Filtrowanie i czyszczenie")

    # Usuwanie kolumn z mało istotnymi danymi lub ze zbyt dużą ilością wartości null
    columns_to_drop = ['Fuel_Consumption_l', 'Previous_Owners', 'Gears', 'Cylinders',
                       'Upholstery', 'Engine_Size_cc', 'Drivetrain', 'Image_url']

    df_dropped_columns = df.drop(*columns_to_drop)

    # Usunięcie wierszy z wartością null w kolumnach
    df_removed_null_rows = df_dropped_columns.filter(
        col("Gearbox").isNotNull() &
        col("Model").isNotNull() &
        col("Seller").isNotNull() &
        col("Fuel_Type").isNotNull() &
        col("Body").isNotNull()
    )

    # Obliczenie mediany dla poszczególnych kolumn (środkowy kwantyl z dokładnością do 0.01)
    medians = df_removed_null_rows.approxQuantile(['Year', 'Doors', 'Seats'], [0.5], 0.01)

    median_year = medians[0][0]
    median_doors = medians[1][0]
    median_seats = medians[2][0]

    # Uzupełnienie braków medianą
    df_replaced_with_median = df_removed_null_rows.na.fill({
        'Year': median_year,
        'Doors': median_doors,
        'Seats': median_seats
    })
    # Uzupełnienie braków w kolumnach kategorycznych wartością "Unknown"
    df_replaced_with_unknown = df_replaced_with_median.na.fill({
        "Color": "Unknown",
        "Country": "Unknown"
    })

    # Usuwanie wartości powyżej lub poniżej określonego progu
    df_cleaned_rows = df_replaced_with_unknown.filter(
        (col("Mileage_km") < 999999) &
        (col("Price") < 750000) &
        (col("Price") > 51) &
        (col("Power_hp") < 1203) &
        (col("Seats") < 10) &
        (col("Doors") < 10)
    )

    # Usuwanie zduplikowanych kolumn
    df_clean = df_cleaned_rows.dropDuplicates()

    print(f"Po drop columns: {df_dropped_columns.count():,}")
    print(f"Po filtrze outlierów: {df_cleaned_rows.count():,}")
    print(f"Po usunięciu null rows: {df_removed_null_rows.count():,}")
    print(f"Po fill median: {df_replaced_with_median.count():,}")
    print(f"Po fill unknown: {df_replaced_with_unknown.count():,}")
    print(f"Po dropDuplicates: {df_clean.count():,}")

    rows_after_clean = df_clean.count()
    rows_removed = initial_count - rows_after_clean
    print(f"Liczba wierszy po czyszczeniu: {rows_after_clean:,}")
    print(f"Usunięte wiersze: {rows_removed:,} ({rows_removed / initial_count * 100:.1f}%)")

    print("\nTransformacja 2. Rzutowanie typów danych")

    # Rzutowanie kolumn na docelowe typy oraz mapowanie wartości logicznych
    df_typed = df_clean \
        .withColumn("Mileage_km", col("Mileage_km").cast(IntegerType())) \
        .withColumn("Price", col("Price").cast(FloatType())) \
        .withColumn("Year", col("Year").cast(IntegerType())) \
        .withColumn("Power_hp", col("Power_hp").cast(IntegerType())) \
        .withColumn("Seats", col("Seats").cast(IntegerType())) \
        .withColumn("Doors", col("Doors").cast(IntegerType())) \
        .withColumn("Non_Smoker_Vehicle", when(col("Non_Smoker_Vehicle") == True, "Yes").otherwise("No"))
    print("Rzutowanie typów zakończone")

    print("\nSchemat po rzutowaniu:")
    df_typed.printSchema()

    # Ujednolicenie nazw kolumn w celu ułatwienia dalszej analizy
    df_renamed_columns = (df_typed
                          .withColumnRenamed("Make", "Brand")
                          .withColumnRenamed("Mileage_km", "Mileage")
                          .withColumnRenamed("Power_hp", "Horsepower"))

    # Ujednolicenie i pełne rozwinięcie nazw krajów
    df_columns_renamed = df_renamed_columns.withColumn(
        "Country",
        F.when(F.col("Country") == "DE", "Deutschland")
        .when(F.col("Country") == "IT", "Italy")
        .when(F.col("Country") == "NL", "Netherlands")
        .when(F.col("Country") == "BE", "Belgium")
        .when(F.col("Country") == "ES", "Spain")
        .when(F.col("Country") == "FR", "France")
        .when(F.col("Country") == "AT", "Austria")
        .when(F.col("Country") == "LU", "Luxembourg")
        .otherwise(F.col("Country"))  # Zostaw oryginalną wartość, jeśli nie pasuje do wzorca
    )

    # Zmiana nazw marki i modeli na duże litery w celu uniknięcia duplikatów wynikających z innej wielkości liter
    df_standardized = df_columns_renamed \
        .withColumn("Brand", F.upper(F.col("Brand"))) \
        .withColumn("Model", F.upper(F.col("Model")))

    print("Transformacja 3. Inżynieria cech (Feature Engineering)")

    # 1. Obliczenie wieku pojazdu i szacowanego rocznego przebiegu
    current_year = datetime.now().year
    df_derived = df_standardized.withColumn(
        "Vehicle_Age",
        current_year - F.col("Year")
    ).withColumn(
        "Annual_Distance_Avg",
        F.col("Mileage") / F.when(F.col("Vehicle_Age") <= 0, 1).otherwise(F.col("Vehicle_Age"))
    )

    # 2. Definicja list dla segmentów rynku
    luxury_brands = [
        "BENTLEY", "ASTON MARTIN", "ROLLS-ROYCE", "LAMBORGHINI",
        "FERRARI", "MCLAREN", "MASERATI", "PORSCHE", "CORVETTE"
    ]

    premium_brands = [
        "ALFA ROMEO", "JAGUAR", "VOLVO", "AUDI", "BMW",
        "MERCEDES-BENZ", "LEXUS", "LAND ROVER", "MINI", "TESLA", "CUPRA"
    ]

    # 3. Kategoryzacja pojazdów na segmenty rynkowe
    df_derived = df_derived.withColumn(
        "Market_Segment",
        F.when(F.col("Brand").isin(luxury_brands), "Luxury")
        .when(F.col("Brand").isin(premium_brands), "Premium")
        .otherwise("Standard")
    )

    # 4. Identyfikacja pojazdów klasycznych
    df_final = df_derived.withColumn(
        "Classic_Vehicle",
        F.when(F.col("Vehicle_Age") >= 25, "Yes").otherwise("No")
    )

    # Zliczenie wierszy po wszystkich transformacjach
    final_count = df_final.count()
    print(f"\nKońcowa liczba wierszy: {final_count:,}")

    print("\nKońcowy schemat danych:")
    df_final.printSchema()

    # Zapis danych
    print(f"\nZapisywanie do {output_path}")

    # Zapisanie wyniku do pojedynczego pliku CSV
    temp_path = "/tmp/spark_csv_output_temp"
    df_final.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(temp_path)

    # Przeniesienie wygenerowanego pliku CSV do docelowej lokalizacji i usunięcie folderów tymczasowych
    import shutil
    csv_files = [f for f in os.listdir(temp_path) if f.endswith('.csv')]

    if csv_files:
        src_file = os.path.join(temp_path, csv_files[0])

        # Usunięcie pliku docelowego, jeżeli już istnieje
        if os.path.exists(output_path):
            os.remove(output_path)

        shutil.move(src_file, output_path)

        # Usunięcie folderu tymczasowego utworzonego przez Sparka
        shutil.rmtree(temp_path)

    print(f"Zapisano w: {output_path}")

    print("Transformacja w Sparku zakończona sukcesem")

    spark.stop()


if __name__ == "__main__":
    main()