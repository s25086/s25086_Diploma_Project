import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Konfiguracja strony
st.set_page_config(page_title="Wycena Samochodów Używanych", layout="wide")
st.title("System predykcyjny do analizy i wyceny samochodów używanych na rynku europejskim")

# 1. Definicja ścieżek i ładowanie zasobów
from pathlib import Path
import os

IN_DOCKER = os.path.exists("/app/models")

if IN_DOCKER:
    BASE_DIR = Path("/app")
    MAPPINGS_FILE = BASE_DIR / "data" / "ml_splits" / "mappings.json"
    MODEL_DIR = BASE_DIR / "models"
else:
    # Jeśli dany plik nie znajduje się w dockerze to przeszukiwane jest środowisko lokalne
    SCRIPT_DIR = Path(__file__).parent.resolve()  # Katalog /streamlit
    PROJECT_ROOT = SCRIPT_DIR.parent              # Katalog /Diploma_Project
    MAPPINGS_FILE = PROJECT_ROOT / "data" / "ml_splits" / "mappings.json"
    MODEL_DIR = PROJECT_ROOT / "models"

st.sidebar.header("Informacje o modelu")
st.sidebar.info("Aktualnie załadowany algorytm: **LightGBM**")

@st.cache_resource
def load_lgbm_model():
    # Wczytuje model LightGBM z pliku .pkl
    model_path = MODEL_DIR / "lightgbm_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_mappings():
    # Wczytuje automatycznie wygenerowane mapowania z pliku JSON
    with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Próba załadowania modelu i mapowań
try:
    model = load_lgbm_model()
    st.sidebar.success("Model LightGBM załadowany i gotowy do predykcji.")
except FileNotFoundError:
    st.sidebar.error(f"Nie znaleziono pliku modelu w: {MODEL_DIR / 'lightgbm_model.pkl'}")
    st.stop()

try:
    MAPPINGS = load_mappings()
except FileNotFoundError:
    st.error(f"Nie znaleziono pliku z mapowaniami w:\n{MAPPINGS_FILE}")
    st.stop()

# 2. Interfejs użytkownika (menu do wprowadzania cech)
st.header("Wprowadź parametry pojazdu")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.text_input("Marka (Brand)", value="BMW").strip().upper()
    model_car = st.text_input("Model", value="M4").strip().upper()
    body = st.selectbox("Typ nadwozia (Body)", list(MAPPINGS["Body"].keys()))
    country = st.selectbox("Kraj (Country)", list(MAPPINGS["Country"].keys()))
    market_segment = st.selectbox("Segment Rynku", list(MAPPINGS["Market_Segment"].keys()))
    classic = st.selectbox("Pojazd Klasyczny?", list(MAPPINGS["Classic_Vehicle"].keys()))

with col2:
    condition = st.selectbox("Stan (Condition)", list(MAPPINGS["Condition"].keys()))
    fuel_type = st.selectbox("Rodzaj paliwa (Fuel_Type)", list(MAPPINGS["Fuel_Type"].keys()))
    gearbox = st.selectbox("Skrzynia biegów (Gearbox)", list(MAPPINGS["Gearbox"].keys()))
    color = st.selectbox("Kolor", list(MAPPINGS["Color"].keys()))
    seller = st.selectbox("Sprzedawca", list(MAPPINGS["Seller"].keys()))
    non_smoker = st.selectbox("Pojazd dla niepalących?", list(MAPPINGS["Non_Smoker_Vehicle"].keys()))

with col3:
    horsepower = st.number_input("Moc (Horsepower)", min_value=1, max_value=1200, value=150)
    doors = st.number_input("Liczba drzwi (Doors)", min_value=2, max_value=9, value=5)
    seats = st.number_input("Liczba miejsc (Seats)", min_value=1, max_value=9, value=5)
    vehicle_age = st.number_input("Wiek pojazdu (Vehicle_Age)", min_value=0, max_value=100, value=5)
    annual_distance = st.number_input("Szacowany roczny przebieg (Annual_Distance_Avg)", min_value=0.0, value=15000.0)
    full_service = st.selectbox("Pełna historia serwisowa (Full_Service_History)", [True, False])

# 3. Przygotowanie wektora wejściowego
if brand not in MAPPINGS.get("Brand", {}):
    st.warning(f"Marka '{brand}' nie występuje w bazie danych. Spróbuj wprowadzić jeszcze raz.")
    st.stop()

if model_car not in MAPPINGS.get("Model", {}):
    st.warning(f"Model '{model_car}' nie występuje w bazie danych dla żadnej marki. Spróbuj wprowadzić jeszcze raz.")
    st.stop()

input_data = {
    "Brand": brand,
    "Model": model_car,
    "Body": body,
    "Country": country,
    "Condition": condition,
    "Fuel_Type": fuel_type,
    "Gearbox": gearbox,
    "Horsepower": horsepower,
    "Seats": seats,
    "Doors": doors,
    "Color": color,
    "Full_Service_History": int(full_service),
    "Non_Smoker_Vehicle": non_smoker,
    "Seller": seller,
    "Market_Segment": market_segment,
    "Classic_Vehicle": classic,
    "Vehicle_Age": vehicle_age,
    "Annual_Distance_Avg": annual_distance
}

input_df = pd.DataFrame([input_data])

# KRYTYCZNE: Zdefiniowanie kolumn kategorycznych i rzutowanie ich na typ 'category'
categorical_cols = ['Brand', 'Model', 'Body', 'Country', 'Condition', 'Fuel_Type',
                    'Gearbox', 'Color', 'Non_Smoker_Vehicle', 'Seller',
                    'Market_Segment', 'Classic_Vehicle']

input_df[categorical_cols] = input_df[categorical_cols].astype('category')

# 4. Predykcja i rysowanie wykresów
if st.button("Dokonaj wyceny", type="primary"):

    # Wykonanie predykcji za pomocą modelu LightGBM
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader(f"Szacowana cena: **{prediction:,.2f} EUR**")

    st.markdown("### Analiza decyzyjności modelu LightGBM")

    col_plot1, col_plot2 = st.columns(2)

    # Wykres 1: Globalna ważność cech
    with col_plot1:
        st.write("**Najważniejsze cechy z perspektywy całego modelu (Global Feature Importance)**")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 cech
            features = input_df.columns[indices]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(range(len(indices)), importances[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Względna waga cechy')
            st.pyplot(fig)
        else:
            st.info("Brak możliwości wygenerowania ważności cech.")

    # Wykres 2: Lokalna ważność cech (SHAP)
    with col_plot2:
        st.write("**Szczegółowa ważność cech (Wartości SHAP)**")
        with st.spinner("Generowanie wykresu SHAP."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)

                fig_shap, ax_shap = plt.subplots(figsize=(8, 5))

                if isinstance(shap_values, list):
                    shap_val_to_plot = shap_values[0][0]
                else:
                    shap_val_to_plot = shap_values[0]

                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]

                shap.waterfall_plot(shap.Explanation(values=shap_val_to_plot,
                                                     base_values=base_value,
                                                     data=input_df.iloc[0],
                                                     feature_names=input_df.columns),
                                    show=False)
                st.pyplot(fig_shap)
            except Exception as e:
                st.error(f"Nie udało się wygenerować wykresu SHAP: {e}")