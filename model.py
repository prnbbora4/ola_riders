import pandas as pd
import pickle


data = pd.read_csv('taxi.csv')

print(data.head())

data_x = data.iloc[:, 0:-1].values
data_y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

print("Training score :", regressor.score(x_train, y_train))

print("Testing score :", regressor.score(x_test, y_test))

pickle.dump(regressor, open('model.pkl', 'wb'))
