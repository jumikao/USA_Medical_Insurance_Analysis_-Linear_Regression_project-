#<<<<<<<<<<<<<<<DESCRIPTION>>>>>>>>>>>>>>>
# The aim of this project is to analyse the medical insurance costs of insurance packages in USA
# and try to predict the potencial medical insurance costs depending on the assumptions given.

#In particular, we aim to check how feature of "smoking/non-smoking" affect number of medical insurance charges paid.

#Project conducted by Asia & Marta.


#<<<<<<<<<<DATA COLLECTION>>>>>>>>>>>>>>>>>
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


#<<<<<<<<<<<<DATA CLEANING>>>>>>>>>>>>>>
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

#<<<<!!!!!>>>>
#wyświetlanie szumów za pomocą wykresu dla kolumny 'smoker'
# import matplotlib.pyplot as plt
# plt.scatter(data_df.index, data_df['smoker'])
# plt.show()
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





#<<<<<<<<<Exploratory Data Analysis (EDA)>>>>>>>>>>>>>>>>>
#EDA Conducted in Colaboratory/Jupiter Notebooks







# #<<<<<<<<<Feature engineering>>>>>>>>>>>>>>>>>
# import numpy as np
#
# from sklearn.model_selection import train_test_split
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
#
#
# #zmienna objaśniająca
#
#
# #szukana zmienna objaśniana
# data_df['Charges_reg'] = data_df.target
#
#
#
#
#
#
#
# #<<<<<<<<<Modelling>>>>>>>>>>>>>>>>>
#
# X = data_df[['smoker']]
# y = data_df['Charges_reg']
#
# #definiowanie podziału na dane testowe i treningowe
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=0)
#
# #budowanie modelu regresji i trenowanie modelu
# regr = linear_model.LinearRegression()
# regr.fit(X_train, y_train)
#
# #prognozowanie
# y_train_pred = regr.predict(X_train)
# y_test_pred = regr.predict(X_test)
#
# #ocena modelu: MSE, R^2

