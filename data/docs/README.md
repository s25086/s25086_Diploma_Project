# Opis zbioru danych — EU Used Cars

## Źródło

Dane pochodzą z europejskiego portalu ogłoszeniowego AutoScout24, udostępnione publicznie na platformie Kaggle przez użytkownika **Alessandro Mazzeo** pod nazwą **„EU Used Cars"**.

- Licencja: **MIT** — dozwolone wykorzystanie badawcze i komercyjne, bez ograniczeń
- Link: https://www.kaggle.com/datasets/alemazz11/cars-europe/data

Zbiór zawiera ponad **40 000 rekordów**, każdy odpowiadający jednej ofercie sprzedaży samochodu używanego z europejskiego rynku wtórnego, z informacjami technicznymi i handlowymi dotyczącymi pojazdu.

## Pliki

| Plik | Opis |
|---|---|
| `data/raw/fullGas.csv` | Dane nieprzetworzone — oryginalny plik z Kaggle |
| `data/processed/fullGas_processed.csv` | Dane po przetworzeniu (filtrowanie, standaryzacja, inżynieria cech) — wynik działania skryptu przetwarzającego dane |

## Słownik danych (dane nieprzetworzone)

| Lp | Kolumna | Opis | Typ | Jednostka | Przykładowe wartości |
|---|---|---|---|---|---|
| 1 | Make | Marka pojazdu | Kategoryczna | – | Abarth, Maserati |
| 2 | Model | Model pojazdu | Kategoryczna | – | 595, Quattroporte |
| 3 | Body | Typ nadwozia | Kategoryczna | – | Compact, Coupe |
| 4 | Mileage_km | Przebieg pojazdu | Liczbowa | km | 98000, 16777215 |
| 5 | Price | Cena | Liczbowa | € | 16900, 20500 |
| 6 | Year | Rok produkcji | Liczbowa (float) | – | 2020.0, 2016.0 |
| 7 | Country | Kraj ogłoszenia | Kategoryczna | – | IT, NL |
| 8 | Condition | Stan (nowy/używany) | Kategoryczna | – | Used, New |
| 9 | Fuel_Type | Typ paliwa | Kategoryczna | – | Gasoline, Diesel |
| 10 | Fuel_Consumption_l | Średnie spalanie | Liczbowa (float) | l/km | 6.7, 16.0 |
| 11 | Drivetrain | Układ napędowy | Kategoryczna | – | Front Wheel Drive, 4WD |
| 12 | Gearbox | Typ skrzyni biegów | Kategoryczna | – | Automatic, Manual |
| 13 | Gears | Liczba biegów | Liczbowa (float) | – | 4, 6 |
| 14 | Power_hp | Moc silnika | Liczbowa | KM | 150, 400 |
| 15 | Engine_Size_cc | Pojemność silnika | Liczbowa (float) | cm³ | 990.0 |
| 16 | Cylinders | Liczba cylindrów | Liczbowa (float) | – | 4.0, 6.0 |
| 17 | Seats | Liczba miejsc w pojeździe | Liczbowa (float) | – | 4.0, 2.0 |
| 18 | Doors | Liczba drzwi pojazdu | Liczbowa (float) | – | 3.0, 4.0 |
| 19 | Color | Kolor samochodu | Kategoryczna | – | White, Grey |
| 20 | Upholstery | Rodzaj tapicerki | Kategoryczna | – | Cloth, Full leather |
| 21 | Full_Service_History | Dostępność pełnej historii serwisowania | Logiczna (boolean) | – | False, True |
| 22 | Non_Smoker_Vehicle | Czy pojazd jest dostosowany dla niepalących | Logiczna (boolean) | – | False, True |
| 23 | Previous_Owners | Liczba poprzednich właścicieli | Liczbowa (float) | – | 4.0, 1.0 |
| 24 | Seller | Typ sprzedawcy | Kategoryczna | – | Dealer, PrivateSeller |
| 25 | Image_url | Link do zdjęcia pojazdu | Kategoryczna | – | Adres URL |

## Cechy dodane podczas przetwarzania

Po przejściu przez potok przetwarzania (PySpark, zautomatyzowany w Apache Airflow) do zbioru dodawane są dodatkowe, wyinżynierowane cechy: `MarketSegment`, `ClassicVehicle`, `Vehicle_Age`. Pełny opis logiki ich tworzenia znajduje się w rozdziale 4.4.3 pracy inżynierskiej.

## Aspekty prawne

Zbiór nie zawiera danych osobowych (brak imion, adresów zamieszkania ani numerów identyfikacyjnych pojazdów) — wyłącznie parametry techniczne i cenowe pojazdów.
