# System predykcyjny do analizy i wyceny samochodów używanych

- **Praca inżynierska** — Przemysław Zięba (nr albumu s25086)
- **Wydział Informatyki, PJATK** — Katedra Systemów Inteligentnych i Data Science
- **Promotor:** mgr inż. Piotr Kojałowicz

## Link do repozytorium github z kodem aplikacji:
https://github.com/s25086/s25086_Diploma_Project.git
---

## Uruchomienie

### Wymagania
- Docker Desktop (Windows / Linux / macOS)

### Kroki

1. Wymagane sprawdzenie pliku `.env` w katalogu głównym projektu — musi zawierać identyfikator użytkownika i grupy:
   ```
   AIRFLOW_UID=<Twój_ID_użytkownika>
   AIRFLOW_GID=<Twój_ID_grupy>
   ```
   (zapobiega to problemom z uprawnieniami plików między systemem Windows a kontenerami opartymi na Linuksie)

2. Zbudowanie obrazów — jedno polecenie buduje równolegle wszystkie kontenery: Airflow (apiserver, scheduler, worker, dag-processor, init, triggerer) oraz Streamlit:
   ```
   docker compose build
   ```

3. Uruchomienie kontenerów w tle:
   ```
   docker compose up -d
   ```

4. Gotowe — dostępne interfejsy:
   - **Apache Airflow** (potok danych, trening i ewaluacja modeli): http://localhost:8080
   - **Panel Streamlit** (predykcja ceny pojazdu, interpretacja SHAP): http://localhost:8501

Potok danych w Airflow uruchamia się automatycznie co 7 dni (harmonogram `@weekly`); można go też odpalić ręcznie z poziomu interfejsu webowego Airflow.

### Notatnik EDA (poza Dockerem)

Plik `src/eda_and_normalization/diploma_project_EDA.ipynb` należy do warstwy eksploracji danych, która — inaczej niż potok Airflow i panel Streamlit — działa w Jupyter Notebook na hoście, nie w kontenerze Docker. Żeby go uruchomić:

1. Utworzenie i aktywowanie lokalnego środowisko wirtualnego:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   ```
2. Zainstalowanie zależności z `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
3. Uruchomienie Jupytea i otworzenie notatnika:
   ```
   jupyter notebook src/eda_and_normalization/diploma_project_EDA.ipynb
   ```
   (albo `jupyter lab`, albo otworzenie pliku bezpośrednio w VS Code z rozszerzeniem Jupyter)

Airflow i Streamlit instalują swoje zależności samodzielnie wewnątrz obrazów Dockera (przez `Dockerfile`), więc `requirements.txt` w katalogu głównym służy właśnie do uruchomienia notatnika EDA na hoście. Minimalna zawartość:
```
pandas==2.1.4
numpy==1.26.4
matplotlib==3.10.9
seaborn==0.13.2
jupyter
```
Jeśli istnieje już działające środowisko, w którym testowany był notatnik, bezpieczniej zrobić `pip freeze > requirements.txt`, użyte zostaną dokładnie te wersje, na których faktycznie działał.

---

## Co zawiera aplikacja

Projekt to system predykcji cen samochodów używanych na rynku europejskim, złożony z trzech warstw funkcjonalnych:

- **Warstwa eksploracji danych** (Jupyter Notebook) — wstępna inspekcja zbioru, analiza statystyczna, wizualizacja rozkładów, identyfikacja anomalii i braków danych.
- **Warstwa przetwarzania i modelowania** (Apache Airflow + PySpark, w kontenerach Docker) — normalizacja danych, inżynieria cech, podział na zbiory treningowy/walidacyjny/testowy, trening i ewaluacja 4 modeli ML (Random Forest, XGBoost, CatBoost, LightGBM) z opcjonalnym strojeniem hiperparametrów (GridSearchCV).
- **Warstwa prezentacji** (Streamlit, osobny kontener) — interaktywny panel z formularzem do predykcji ceny pojazdu na podstawie parametrów wprowadzonych przez użytkownika oraz wizualizacją istotności cech (SHAP).

Modelem finalnym, na którym oparta jest aplikacja prezentacyjna, jest **LightGBM** — najlepszy wynik na zbiorze testowym: R² = 0,959, MAE = 6211,28, RMSE = 17262,99.

## Dane

Zbiór danych: *"EU Used Cars"* (Kaggle, autor: Alessandro Mazzeo, licencja MIT) — ponad 40 000 ofert z portalu AutoScout24.
Pełny opis kolumn, plików i pochodzenia danych: [`data/docs/README.md`](data/docs/README.md).

## Struktura katalogów

```
├── config/           # konfiguracja Airflow
├── dags/              # diploma_project_pipeline.py – definicja potoku danych
├── data/               # dane wejściowe i przetworzone (patrz data/docs/README.md)
├── logs/               # logi Airflow
├── models/            # wytrenowane modele i ich statystyki (.json)
├── plugins/            # pluginy Airflow
├── reports/            # wykresy i raporty generowane przez kod źródłowy
├── src/                # kod źródłowy (EDA, przetwarzanie danych, trening modeli)
├── streamlit/          # app.py + Dockerfile aplikacji prezentacyjnej
├── .env                # AIRFLOW_UID / AIRFLOW_GID
├── docker-compose.yaml # orkiestracja Airflow + Streamlit
├── Dockerfile           # obraz Airflow (Java + PySpark + biblioteki ML)
└── requirements.txt
```

## Technologie

Jupyter Notebook, pandas, NumPy, Matplotlib, Seaborn, Apache Airflow, PySpark, scikit-learn, Random Forest / XGBoost / CatBoost / LightGBM, Streamlit, SHAP, Docker / Docker Compose.

## Bezpieczeństwo

- Każdy komponent działa w izolowanym kontenerze Docker, komunikacja wyłącznie przez wewnętrzną sieć Dockera (np. Airflow łączy się z bazą pod adresem `postgres`, z brokerem pod `redis`).
- Brak kluczy dostępowych i haseł w kodzie źródłowym.
- Airflow uruchamiany z ustawionym UID/GID użytkownika (plik `.env`) — zapobiega tworzeniu plików z uprawnieniami roota.
- Zbiór testowy jest izolowany przez cały czas treningu modeli.

## Autor

Przemysław Zięba, nr albumu s25086
