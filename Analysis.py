#DESCRIPTION
# The aim of this project is to analyse the medical insurance costs of insurance packages in USA
# and try to predict the potencial medical insurance costs depending on the assumptions given.



#"DATA COLLECTION"

import pandas as pd

data_df = pd.read_csv(r'https://raw.githubusercontent.com/Kamil128/ProjektPraktycznyRegresja/main/data/medical_cost/medical_cost.csv',
                 )

print(data_df.head())
print("........")
print(data_df.columns)

print("........")
print(data_df.info())


#"DATA CLEANING"

#funkcja .duplicated() - czy są zduplikowane wiersze
print(data_df.duplicated())
# brak zduplikowanych danych

print("........")

#funkcja .isna() - pokaż braki
print(data_df.isna())
#brak Nan-ów

#wyświetlanie szumów za pomocą wykresu
import matplotlib.pyplot as plt
plt.scatter(data_df.index, data_df['smoker'])
plt.show()
# brak szumów


print(data_df['smoker'].unique())
#Out: ['yes' 'no']
# Brak wartości błędnych - yes/no - prawidłowe odpowiedzi