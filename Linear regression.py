import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


houses = pd.read_csv("kc_house_data.csv")
houses.head()

houses.isnull().sum()

corr = houses.corr()
sns.heatmap(corr)

feature_cols = 'sqft_living'
x = houses[feature_cols] # predictor
y = houses.price # response

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)
x_train = np.array([x_train])
y_train = np.array([y_train])
x_train = np.transpose(x_train)
y_train = np.transpose(y_train)
print(x_train.shape)
print(y_train.shape)

linreg = LinearRegression()
linreg.fit(x_train, y_train)
ax = plt.subplot(111)
ax.scatter(x_train[:,0],y_train[:,0])
y_train = linreg.predict(x_train)
ax.scatter(x_train,y_train)

ax.figure.show()
mse = mean_squared_error(y_test, linreg.predict(x_test))
np.sqrt(mse)
linreg.score(x_test,y_test)