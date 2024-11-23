# Import necessary modules
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title
st.title("Home Price Prediction using Machine Learning")

# Predictions
st.title("Please input your values below")


# Test model with new input from user
# Store values in a list
data = []

# Taking 13 different values
# Median Income value
medianIncome  = st.number_input("Enter the Median Income: ")
data.append(medianIncome)

# Median Age value
medianAge = st.number_input("Enter the Median Age: ")
data.append(medianAge)

# Total number of rooms
totRooms  = st.number_input("Enter the Total Number of Rooms: ")
data.append(totRooms)

# Total number of bedrooms
bRoom = st.number_input("Enter the Total Number of Bedrooms: ")
data.append(bRoom)

# Area population
population = st.number_input("Enter the Population: ")
data.append(population)

# Number of households in area
households = st.number_input("Enter the Households: ")
data.append(households)

# Latitude
lat = st.number_input("Enter the Latitude: ")
data.append(lat)

# Longitude
long = st.number_input("Enter the Longitude: ")
data.append(long)

# Distance to Coast
dtC = st.number_input("Enter the Distance to Coast: ")
data.append(dtC)

# Distance to LA
dtLA = st.number_input("Enter the Distance to LA: ")
data.append(dtLA)

# Distance to San Diego
dtSD = st.number_input("Enter the Distance to SanDiego: ")
data.append(dtSD)

# Distance to San Jose
dtSJ = st.number_input("Enter the Distance to SanJose: ")
data.append(dtSJ)

# Distance to San Francisco
dtSF = st.number_input("Enter the Distance to SanFrancisco: ")
data.append(dtSF)

# Change list to a DataFrame
df = pd.DataFrame(data)
data = np.array(df).reshape(1, -1)
df = pd.DataFrame(data)

# Predictions
st.title("Model Options")

# Initialize session state to store results
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# Load saved models and make predictions
def load_model(filename):
    with open(filename, 'rb') as model_file:
        return pickle.load(model_file)

# Load saved models

# Load Random forest 
with open('rf_r.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)
# Assign model to variable
random_forest_model_prediciton = rf_model.predict(df)[0]
# Button to predict model
if st.button("Random Forest Model Prediction"):
    # Output model results
    st.session_state.predictions["Random Forest"] = f"Predicted Median Value of House is: ${random_forest_model_prediciton:.2f}"


# Decision Tree model
with open('dt_r.pkl', 'rb') as model_file:
    dt_model = pickle.load(model_file)
# Assign model to variable
decision_tree_model_prediciton = dt_model.predict(df)[0]
# Button to predict model
if st.button("Decision Tree Model Prediction"):
    # Output model results
    st.session_state.predictions["Decision Tree"] = f"Predicted Median Value of House is: ${decision_tree_model_prediciton:.2f}"


# MLP model
with open('mlp_r.pkl', 'rb') as model_file:
    mlp_model = pickle.load(model_file)
# Assign model to variable
mlp_prediction = mlp_model.predict(df)[0]
# Button to predict model
if st.button("MLP Prediction"):
    # Output model results
    st.session_state.predictions["MLP"] = f"Predicted Median Value of House is: ${mlp_prediction:.2f}"


# Elastic Net model
with open('en_r.pkl', 'rb') as model_file:
    en_r_model = pickle.load(model_file)
# Assign model to variable
en_prediction = en_r_model.predict(df)[0]
# Button to predict model
if st.button("Elastic Net Prediction"):
    # Output model results
    st.session_state.predictions["Elastic Net"] = f"Predicted Median Value of House is: ${en_prediction:.2f}"


# Multiplelinear Regression model
with open('lr_r.pkl', 'rb') as model_file:
    lr_model = pickle.load(model_file)
# Assign model to variable
lr_prediction = lr_model.predict(df)[0]
# Button to predict model
if st.button("Multiplelinear Regression Prediction"):
    # Output model results
    st.session_state.predictions["Linear Regression"] = f"Predicted Median Value of House is: ${lr_prediction:.2f}"

# Display all selected predictions
st.title("Model Predictions")
for model_name, result in st.session_state.predictions.items():
    st.write(f"{model_name}: {result}")

# Sample values for testing
# 8.3, 41, 800, 129, 322, 126, 37.88,-122.23,9263.040773,556529.158342,735501.806984,67432.517001,21250.213767
# 10.5424, 16, 4392, 602, 1490, 578, 33.73, -118.06, 2087.410225, 39639.685794, 140395.246608, 530014.877565, 598010.914448 



