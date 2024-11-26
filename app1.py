# Import necessary modules
import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType

# Title
st.title("Home Price Prediction using Machine Learning")

# Predictions
st.title("Please input your values below")

# Test model with new input from user
data = []

# Taking 13 different values
medianIncome = st.number_input("Enter the Median Income: ")
data.append(medianIncome)

medianAge = st.number_input("Enter the Median Age: ")
data.append(medianAge)

totRooms = st.number_input("Enter the Total Number of Rooms: ")
data.append(totRooms)

bRoom = st.number_input("Enter the Total Number of Bedrooms: ")
data.append(bRoom)

population = st.number_input("Enter the Population: ")
data.append(population)

households = st.number_input("Enter the Households: ")
data.append(households)

lat = st.number_input("Enter the Latitude: ")
data.append(lat)

long = st.number_input("Enter the Longitude: ")
data.append(long)

dtC = st.number_input("Enter the Distance to Coast: ")
data.append(dtC)

dtLA = st.number_input("Enter the Distance to LA: ")
data.append(dtLA)

dtSD = st.number_input("Enter the Distance to San Diego: ")
data.append(dtSD)

dtSJ = st.number_input("Enter the Distance to San Jose: ")
data.append(dtSJ)

dtSF = st.number_input("Enter the Distance to San Francisco: ")
data.append(dtSF)

# Convert input data to DataFrame
df = pd.DataFrame(data).T
df = df.astype(np.float32)

# Model predictions
st.title("Model Options")

# Initialize session state to store results
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# Models and paths
models = {
    "Random Forest": "rf_r.onnx",
    "Decision Tree": "dt_r.onnx",
    "MLP": "mlp_r.onnx",
    "Elastic Net": "en_r.onnx",
    "Linear Regression": "lr_r.onnx"
}

# Generate buttons for each model and make predictions
for model_name, model_path in models.items():
    if st.button(f"{model_name} Prediction"):
        # Load the ONNX model
        session = rt.InferenceSession(model_path)
        # Get input name
        input_name = session.get_inputs()[0].name
        # Make prediction
        prediction = session.run(None, {input_name: df.to_numpy()})[0]
        # Convert prediction to a scalar float value
        prediction_value = float(prediction[0])
        # Store the prediction in session state
        st.session_state.predictions[model_name] = f"Predicted Median Value of House is: ${prediction_value:.2f}"

# Display predictions
st.title("Model Predictions")
for model_name, result in st.session_state.predictions.items():
    st.write(f"{model_name}: {result}")




# Sample values for testing
# 8.3, 41, 800, 129, 322, 126, 37.88,-122.23,9263.040773,556529.158342,735501.806984,67432.517001,21250.213767
# 10.5424, 16, 4392, 602, 1490, 578, 33.73, -118.06, 2087.410225, 39639.685794, 140395.246608, 530014.877565, 598010.914448 

