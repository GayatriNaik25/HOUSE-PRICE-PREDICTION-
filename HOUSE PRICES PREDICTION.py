import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

dataset = pd.read_csv(r"D:\DATA SCIENCE CLASS NOTES\ml\House_data.csv")

X = dataset.iloc[:]['sqft_living'].values.reshape(-1,1)
y = dataset.iloc[:]['price'].values.reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Visualize the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('House Price vs Square Feet (Training set)')
plt.xlabel('Sqaure Feet')
plt.ylabel('House Price')
plt.show()

# Visualize the test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('House Price vs Square Feet (Test set)')
plt.xlabel('Sqaure Feet')
plt.ylabel('House Price')
plt.show()

# Predict
y_2525 = regressor.predict([[2525]])
y_1000 = regressor.predict([[1000]])
print(f"Predicted House price for 2525 Sqaure feet : ${y_2525[0][0]:,.2f}")
print(f"Predicted House price for 1000 Sqaure feet : ${y_1000[0][0]:,.2f}")
filename = 'houseprice_prediction_model.pkl'
with open(filename, 'wb') as file:pickle.dump(regressor, file)
print("Model has been pickled and saved as houseprice_prediction_model.pkl")

from sklearn.metrics import r2_score,mean_squared_error
R2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RSME = np.sqrt(MSE)
print(' R-Square :{}'.format(R2),'\n','MSE :{} \n RSME : {}'.format(MSE,RSME))
# regression Table code
# introduce to OLS & stats.api
from statsmodels.api import OLS
OLS(y_train,X_train).fit().summary()