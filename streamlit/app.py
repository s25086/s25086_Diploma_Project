import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shap  # Wymaga instalacji: pip install shap

# Konfiguracja strony
st.set_page_config(page_title="Wycena Samochodów Używanych", layout="wide")
st.title("System predykcyjny do analizy i wyceny samochodów używanych na rynku europejskim")

# -----------------------------------------------------------------------------
# 1. ŁADOWANIE MODELU (Tylko LightGBM)
# -----------------------------------------------------------------------------
MODEL_DIR = "models"  # Możesz zmienić na /opt/airflow/models zależnie od środowiska

st.sidebar.header("Informacje o modelu")
st.sidebar.info("Aktualnie załadowany algorytm: **LightGBM**")


@st.cache_resource
def load_lgbm_model():
    """Wczytuje zserializowany model LightGBM z pliku .pkl"""
    model_path = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


try:
    model = load_lgbm_model()
    st.sidebar.success("Model LightGBM załadowany i gotowy do predykcji.")
except FileNotFoundError:
    st.sidebar.error(f"Nie znaleziono pliku modelu w {MODEL_DIR}/lightgbm_model.pkl")
    st.stop()

# -----------------------------------------------------------------------------
# 2. MAPOWANIE ZMIENNYCH KATEGORYCZNYCH (KROK DO WYKONANIA)
# -----------------------------------------------------------------------------
# Zastąp przykładowe dane ("AUDI": 0) mapowaniami ze swojego zbioru treningowego.
MAPPINGS = {
    "Brand": {"AUDI": 0, "BMW": 1, "MERCEDES-BENZ": 2, "VOLVO": 3},
    "Model": {"A4": 0, "M3": 1, "C-CLASS": 2},
    "Body": {"Sedan": 0, "Kombi": 1, "SUV": 2},
    "Country": {"Deutschland": 0, "Italy": 1, "France": 2},
    "Condition": {"Used": 0, "New": 1},
    "Fuel_Type": {"Benzin": 0, "Diesel": 1, "Electric": 2},
    "Gearbox": {"Manual": 0, "Automatic": 1},
    "Color": {"Black": 0, "White": 1, "Unknown": 2},
    "Non_Smoker_Vehicle": {"No": 0, "Yes": 1},
    "Seller": {"Dealer": 0, "Private": 1},
    "Market_Segment": {"Standard": 0, "Premium": 1, "Luxury": 2},
    "Classic_Vehicle": {"No": 0, "Yes": 1}
}

# -----------------------------------------------------------------------------
# 3. INTERFEJS UŻYTKOWNIKA (WPROWADZANIE CECH)
# -----------------------------------------------------------------------------
st.header("Wprowadź parametry pojazdu")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Marka (Brand)", list(MAPPINGS["Brand"].keys()))
    model_car = st.selectbox("Model", list(MAPPINGS["Model"].keys()))
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

# -----------------------------------------------------------------------------
# 4. PRZYGOTOWANIE WEKTORA WEJŚCIOWEGO
# -----------------------------------------------------------------------------
input_data = {
    "Brand": MAPPINGS["Brand"][brand],
    "Model": MAPPINGS["Model"][model_car],
    "Condition": MAPPINGS["Condition"][condition],
    "Body": MAPPINGS["Body"][body],
    "Country": MAPPINGS["Country"][country],
    "Fuel_Type": MAPPINGS["Fuel_Type"][fuel_type],
    "Gearbox": MAPPINGS["Gearbox"][gearbox],
    "Color": MAPPINGS["Color"][color],
    "Non_Smoker_Vehicle": MAPPINGS["Non_Smoker_Vehicle"][non_smoker],
    "Seller": MAPPINGS["Seller"][seller],
    "Horsepower": horsepower,
    "Doors": doors,
    "Seats": seats,
    "Market_Segment": MAPPINGS["Market_Segment"][market_segment],
    "Classic_Vehicle": MAPPINGS["Classic_Vehicle"][classic],
    "Vehicle_Age": vehicle_age,
    "Annual_Distance_Avg": annual_distance
}

input_df = pd.DataFrame([input_data])

# -----------------------------------------------------------------------------
# 5. PREDYKCJA I WYJAŚNIALNOŚĆ (XAI)
# -----------------------------------------------------------------------------
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
        st.write("**Co wpłynęło na tę konkretną wycenę? (Wartości SHAP)**")
        with st.spinner("Generowanie wykresu SHAP..."):
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