# Import necessary modules
import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as rt
# from skl2onnx import convert_sklearn
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
df = pd.DataFrame(data)
data = np.array(df).reshape(1, -1)
df = pd.DataFrame(data)

# df.columns = [
#     "Median Income", "Median Age", "Total Rooms", "Total Bedrooms",
#     "Population", "Households", "Latitude", "Longitude",
#     "Distance to Coast", "Distance to LA", "Distance to San Diego",
#     "Distance to San Jose", "Distance to San Francisco"
# ]

# Model predictions
st.title("Model Options")

# Initialize session state to store results
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# Function to load ONNX model and predict
def predict_with_onnx(model_path, input_data):
    # Load ONNX model
    session = rt.InferenceSession(model_path)
    # Get input name
    input_name = session.get_inputs()[0].name
    # Predict
    prediction = session.run(None, {input_name: input_data.astype(np.float32)})[0]
    # Convert prediction to a scalar float value
    return float(prediction[0])

# Models and paths
models = {
    "Random Forest": "rf_r.onnx",
    "Decision Tree": "dt_r.onnx",
    "MLP": "mlp_r.onnx",
    "Elastic Net": "en_r.onnx",
    "Linear Regression": "lr_r.onnx"
}

# Generate buttons for each model
for model_name, model_path in models.items():
    if st.button(f"{model_name} Prediction"):
        # Convert the prediction to a scalar
        prediction = predict_with_onnx(model_path, df.to_numpy())
        # Format and store the prediction in session state
        st.session_state.predictions[model_name] = f"Predicted Median Value of House is: ${prediction:.2f}"

# Display predictions
st.title("Model Predictions")
for model_name, result in st.session_state.predictions.items():
    st.write(f"{model_name}: {result}")



# Sample values for testing
# 8.3, 41, 800, 129, 322, 126, 37.88,-122.23,9263.040773,556529.158342,735501.806984,67432.517001,21250.213767
# 10.5424, 16, 4392, 602, 1490, 578, 33.73, -118.06, 2087.410225, 39639.685794, 140395.246608, 530014.877565, 598010.914448 

