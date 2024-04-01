import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df_samples = pd.read_excel(r'RideCount.xlsx', engine='openpyxl')
new_df  = df_samples.drop('id', axis=1)
new_df  = new_df.drop('datetime', axis=1)
print(new_df.head())

print(new_df. sum())
print(new_df. sum().sum())