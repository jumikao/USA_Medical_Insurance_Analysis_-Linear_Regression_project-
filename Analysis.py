#DESCRIPTION
# The aim of this project is to analyse the medical insurance costs of insurance packages in USA
# and try to predict the potencial medical insurance costs depending on the assumptions given.




#"DATA COLLECTION"
#importujemy dane z repozytorium Kamila Pabijan - naszego trenera.

import pandas as pd

data_df = pd.read_csv(r'https://raw.githubusercontent.com/Kamil128/ProjektPraktycznyRegresja/main/data/medical_cost/medical_cost.csv',
                 )

print(data_df.head())
print("........")
print(data_df.columns)

print("........")
print(data_df.info())

print("........")
print(data_df.describe())


#"DATA CLEANING"
#Wnioski:
#- brak duplikatów
#- brak Nan-ów
#- brak szumów, wartości są w poprawnych formatach, wartości mają sens.
#W analizie występuje dysproporcja w ilości danych pomiędzy palaczami i nie-palaczami (1:4)

#funkcja .duplicated() - czy są zduplikowane wiersze
print(data_df.duplicated())
# brak zduplikowanych danych


#funkcja .isna() - pokaż braki
print(data_df.isna())
#brak Nan-ów



#funkcja .unique() - szukanie szumów
print(data_df['smoker'].unique())
#Out: ['yes' 'no']
# Brak wartości błędnych - yes/no - prawidłowe odpowiedzi

print(data_df['children'].unique())
#Out: [0 1 3 2 5 4]

print(data_df['sex'].unique())
#Out: ['female' 'male']

print(data_df['age'].unique())
#Out: [19 18 28 33 32 31 46 37 60 25 62 23 56 27 52 30 34 59 63 55 22 26 35 24
 # 41 38 36 21 48 40 58 53 43 64 20 61 44 57 29 45 54 49 47 51 42 50 39]

print(data_df['region'].unique())
#Out: ['southwest' 'southeast' 'northwest' 'northeast']

print(data_df['charges'].unique())
# Out: [16884.924   1725.5523  4449.462  ...  1629.8335  2007.945  29141.3603]

print(data_df['bmi'].unique())
#Out:


#wyświetlanie szumów za pomocą wykresu dla kolumny 'smoker'
import matplotlib.pyplot as plt
plt.scatter(data_df.index, data_df['smoker'])
plt.show()
# brak szumów

#Sprawdzenie reprezentatywności danych
smoker=0
non_smoker=0

for person in data_df["smoker"]:
    if person== "yes":
        smoker +=1
    else:
        non_smoker +=1

print("Number od smokers: ", smoker)
print("Number of non-smokers: ", non_smoker)


