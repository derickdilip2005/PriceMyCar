import streamlit as st
import pickle
import numpy as np

with open('model/price_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('PriceMyCar 🚗')

st.markdown("### Enter the details of your car below:")

vehicle_age = st.number_input('Years since purchase', min_value=0, max_value=50, value=5)
km_driven = st.number_input('Kilometers Driven', min_value=0)
mileage = st.number_input('Mileage', min_value=0.0)
engine = st.number_input('Engine Capacity (CC)', min_value=0)
max_power = st.number_input('Max Power (bhp)', min_value=0.0)
seats = st.number_input('Number of Seats', min_value=0, value=4)
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission = st.selectbox('Transmission Type', ['Manual', 'Automatic'])

fuel_type_petrol = 1 if fuel_type == 'Petrol' else 0
fuel_type_diesel = 1 if fuel_type == 'Diesel' else 0
seller_type_individual = 1 if seller_type == 'Individual' else 0
transmission_manual = 1 if transmission == 'Manual' else 0

features = np.array([[vehicle_age, km_driven, mileage, engine, max_power, seats, 
                      fuel_type_diesel, fuel_type_petrol,
                      seller_type_individual, transmission_manual]])


if st.button('Predict Selling Price'):
    prediction = model.predict(features)
    st.success(f'Estimated Selling Price: ₹ {round(prediction[0], 2)}')
