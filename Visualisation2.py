import pandas as pd
import os
#import sklearn.linear_model
#import matplotlib.pyplot as plt
#import numpy as np


#przypisz sciezke i plik z danymi do zmiennej
#DOWNLOAD_ROOT = "https://raw.githubusercontent.com/Kamil128/ProjektPraktycznyRegresja/main/data/medical_cost/"
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/jumikao/USA_Medical_Insurance_Analysis/"
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

##liczbowe informacje z rubryki 'sex'
all_data['sex'].value_counts()

##liczbowe informacje z rubryki 'age'
all_data['age'].value_counts()

#histogramy atrybutow numerycznych(int/float)
%matplotlib inline
import matplotlib.pyplot as plt
all_data.hist(bins=50, figsize=(20,15))
plt.show()

#sprawdzenie czy elementy sie powtarzaja
print(all_data.duplicated())
#wniosek-w all_data brak powtarzajacych sie elementow


#funkcja .isna() - pokaż braki nany
print(all_data.isna())
#brak Nan-ów 1338 rows x 7 columns


#funkcja .unique() - szukanie szumów
print(all_data['smoker'].unique())
#Out: ['yes' 'no']
# Brak wartości błędnych - yes/no - prawidłowe odpowiedzi
print(all_data['children'].unique())
print(all_data['sex'].unique())
#Out: ['female' 'male']
print(all_data['age'].unique())
#Out: [19 18 28 33 32 31 46 37 60 25 62 23 56 27 52 30 34 59 63 55 22 26 35 24
 # 41 38 36 21 48 40 58 53 43 64 20 61 44 57 29 45 54 49 47 51 42 50 39]
print(all_data['region'].unique())
#Out: ['southwest' 'southeast' 'northwest' 'northeast']
print(all_data['charges'].unique())
# Out: [16884.924   1725.5523  4449.462  ...  1629.8335  2007.945  29141.3603]
print(all_data['bmi'].unique())
#Out:[27.9   33.77  33.    22.705 28.88  25.74  33.44  27.74  29.83  25.84....wszystkie wartosci)

#wyświetlanie szumów za pomocą wykresu dla kolumny 'smoker'
import matplotlib.pyplot as plt
plt.scatter(all_data.index, all_data['smoker'])
plt.show()
# brak szumów

#KONTROLNE SPRAWDZENIE SZUMOW POZOSTALYCH DANYCH(WYKRES)
import matplotlib.pyplot as plt
plt.scatter(all_data.index, all_data['sex'])
plt.show()

import matplotlib.pyplot as plt
plt.scatter(all_data.index, all_data['bmi'])
plt.show()

#Sprawdzenie reprezentatywności danych
smoker=0
non_smoker=0

for person in all_data["smoker"]:
    if person== "yes":
        smoker +=1
    else:
        non_smoker +=1

print("Number od smokers: ", smoker)
print("Number of non-smokers: ", non_smoker)
#Number od smokers:  274
#Number of non-smokers:  1064

#COS ZROBIONEGO :d


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline  

import warnings
warnings.filterwarnings('ignore')



all_data = pd.read_csv(r'https://raw.githubusercontent.com/Kamil128/ProjektPraktycznyRegresja/main/data/medical_cost/medical_cost.csv',)


all_data.columns

Index(['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'], dtype='object')



#CORRELATION BETWEEN NUMERIC FEATURES: AGE, BMI, NUMBER OF CHILDREN, CHARGES

    #NO HIGH CORRELATION TO BE SEEN



plt.figure(figsize=(12,8))
count_corr=all_data.corr() 

sns.heatmap(count_corr)

#HOW AGE INFLUENCES NUMBER OF MEDICAL INSURANCE CHARGES PAID

    #People pay more medical insurance charges with age
    #In general, men pay more charges with age than women
    #Smokers pay much more charges with age, compared to non-smokers
    
sns.lmplot(x="age", y="charges", data=all_data, hue="sex", size=(8), palette=sns.color_palette("Set2",10))
sns.lmplot(x="age", y="charges", data=all_data, hue="smoker", size=(8), palette=sns.color_palette("Set2",10))



#HOW BMI INFLUENCES NUMBER OF MEDICAL INSURANCE CHARGES PAID

   # People with higher BMI pay more medical insurance charges
    #Men with high bmi pay more charges than women with high bmi
    #Smokers pay much more medical insurance charges no matter what bmi they have, compared to non-smokers, and they pay #dramatically more insurance charges if their bmi is high


sns.lmplot(x="bmi", y="charges", data=all_data, size=(8), hue="sex", palette=sns.color_palette("Set2",10))

sns.lmplot(x="bmi", y="charges", data=all_data, size=(8), hue="smoker", palette=sns.color_palette("Set2",10))



#HOW SMOKING/NON-SMOKING AND SEX INFLUENCE MEDICAL INSURANCE CHARGES PAID

    #For non-smokers, on average women tend to pay slightly more medical insurance charges
    #For smokers, on average men tend to pay more medical insurance charges
    


plt.figure(figsize=(10,6))
sns.barplot(x="smoker",y="charges", data=all_data, hue="sex", palette=sns.color_palette("Set2",10))

#Distribution of medical insurance charges paid

  #  Distribution of the feature is not-normal


sns.distplot(all_data.charges)

sns.displot(all_data.charges, kde=True)






