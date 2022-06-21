import pandas as pd

data_df = pd.read_csv(r'C:\Users\jumi\Desktop\POMOROSKIE IDZIE W IT - DATA SCIENCE\6.Projekt regresja\git\USA_Medical_Insurance_Analysis/medical_cost.csv',
                 header=None,
                 sep='\s+')

print(data_df.head())
print("........")
print(data_df.columns)