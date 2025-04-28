import pickle 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv', index_col=0)
df.head()

df['Fuel_Type_Petrol'] = df['fuel_type'].apply(lambda x: 1 if x == 'Petrol' else 0)
df['Fuel_Type_Diesel'] = df['fuel_type'].apply(lambda x: 1 if x == 'Diesel' else 0)
df['Seller_Type_Individual'] = df['seller_type'].apply(lambda x: 1 if x == 'Individual' else 0)
df['Transmission_Manual'] = df['transmission_type'].apply(lambda x: 1 if x == 'Manual' else 0)

df.drop(['fuel_type', 'seller_type', 'transmission_type', 'car_name', 'brand', 'model'], axis=1, inplace=True)

X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

with open('model/price_model.pkl', 'wb') as file:
    pickle.dump(model, file)