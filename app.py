import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page title
st.title("🏠 House Price Prediction")
st.write("Fill in the details below to predict the house price!")

# Load and train model
data = pd.read_csv('train.csv')

for column in data.select_dtypes(include=['int64', 'float64']).columns:
    data[column] = data[column].fillna(data[column].mean())

x = data[['OverallQual', 'GrLivArea', 'GarageCars', 
           'GarageArea', 'TotalBsmtSF', '1stFlrSF', 
           'FullBath', 'YearBuilt']]
y = data['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Input sliders
st.subheader("Enter House Details")

OverallQual = st.slider("Overall Quality", 1, 10, 5)
GrLivArea   = st.number_input("Living Area (sq ft)", min_value=100, max_value=10000, value=1500)
GarageCars  = st.slider("Garage Cars", 0, 4, 2)
GarageArea  = st.number_input("Garage Area (sq ft)", min_value=0, max_value=2000, value=500)
TotalBsmtSF = st.number_input("Total Basement SF", min_value=0, max_value=5000, value=800)
FirstFlrSF  = st.number_input("1st Floor SF", min_value=0, max_value=5000, value=800)
FullBath    = st.slider("Full Bathrooms", 0, 4, 2)
YearBuilt   = st.slider("Year Built", 1900, 2025, 2000)

# Predict button
if st.button("Predict Price 🏠"):
    new_house = pd.DataFrame(
        [[OverallQual, 
          GrLivArea, 
          GarageCars, 
          GarageArea, 
          TotalBsmtSF, 
          FirstFlrSF, 
          FullBath,
          YearBuilt]],
          
        columns=['OverallQual', 
                 'GrLivArea', 
                 'GarageCars', 
                 'GarageArea', 
                 'TotalBsmtSF', 
                 '1stFlrSF', 
                 'FullBath', 
                 'YearBuilt']
    )
    predicted_price = model.predict(new_house)
    st.success(f"💰 Predicted House Price: ₹{predicted_price[0]:,.2f}")