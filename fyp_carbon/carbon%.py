import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('carb.csv')
new_df  = df.drop('year', axis=1)
print(new_df.head())
rDf = new_df.corr()
print(rDf)
#r(相關係數) = x和y的協方差/(x的標準差*y的標準差) == cov（x,y）/σx*σy（即person係數）
# 0~0.3 弱相關
# 0.3~0.6 中等程度相關
# 0.6~1 強相關
model = LinearRegression()

sns.pairplot(new_df, x_vars=['pop','gdp'], y_vars='carb', height=7, aspect=0.8,kind = 'reg')

X_train,X_test,Y_train,Y_test = train_test_split(new_df.iloc[:,:2],new_df.carb,train_size = 0.8,test_size = 0.2)
model = LinearRegression()
model.fit(X_train,Y_train)
a  = model.intercept_ #截距 
b = model.coef_ #回歸系數


print("最佳拟合线:截距",a,",回歸系數：[population gdp]",b)

print("y = ",a,"+population*",b[0]," + gdp*",b[1])

plt.show()