import streamlit as st
import pandas as pd
import cloudpickle

# Load model with cloudpickle
@st.cache_resource
def load_model():
    with open("vl_model_useful_features.pkl", "rb") as f:
        return cloudpickle.load(f)

model = load_model()
st.write("Using model:", "vl_model_useful_features.pkl")

# Streamlit UI
st.title("Viral Load Prediction App")

# Collect user input
user_input = {}
user_input["Age at reporting"] = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
user_input["Sex"] = st.selectbox("Select Sex", ["Male", "Female", "Other"])
regimen_options = ["TDF+3TC+DTG", "AZT+3TC+NVP", "TDF+3TC+EFV", "Other"]
user_input["Current Regimen"] = st.selectbox("Select ART Regimen", regimen_options)
user_input["Last VL Result Clean"] = st.number_input("Enter Last Viral Load Result", min_value=0, value=1000)

# Prepare DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.write("Prediction (1 = Suppressed, 0 = Not Suppressed):", prediction[0])
    except Exception as e:
        st.error(f"Error during prediction: {e}")
