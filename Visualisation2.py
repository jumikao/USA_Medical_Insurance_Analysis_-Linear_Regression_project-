import pandas as pd
import os

#przypisz sciezke i plik z danymi do zmiennej
#DOWNLOAD_ROOT = "https://raw.githubusercontent.com/Kamil128/ProjektPraktycznyRegresja/main/data/medical_cost/"
DOWNLOAD_ROOT = "https://github.com/jumikao/USA_Medical_Insurance_Analysis/"
DATA_URL = DOWNLOAD_ROOT + "medical_cost.csv"


#funkcja zwracajaca zawartosc pliku
def load_all_data(file_url=DATA_URL):
    return pd.read_csv(file_url)

#zapisz do obiektu rezultat funkcji load_all_data
all_data = load_all_data()

#wyswietl naglowek
all_data.head() 

#informacja o zbiorze danych
all_data.info()

#liczbowe informacje z rubryki 'smoker'
all_data['smoker'].value_counts()

