import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd 
import pickle 

# Set up the Streamlit page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

@st.cache_resource
def load_encoders():
    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)
        
    with open("onehotencoder_geo.pkl", "rb") as file:
        onehotencoder_geo = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return label_encoder_gender, onehotencoder_geo, scaler

model = load_model()
label_encoder_gender, onehotencoder_geo, scaler = load_encoders()

# Streamlit app
st.title("🏦 Customer Churn Prediction")
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Predict the likelihood of customer churn using your data</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for user input
st.sidebar.header("Input Data")
st.sidebar.markdown("Please enter the customer information below:")

# User Input
geography = st.sidebar.selectbox('Geography', onehotencoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92, value=30)
balance = st.sidebar.number_input('Balance', min_value=0.0, step=100.0, value=50000.0)
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=1000.0, value=50000.0)
tenure = st.sidebar.slider('Tenure', 0, 10, value=3)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, value=2)

# Use descriptive labels for better UX
has_cr_card = st.sidebar.selectbox('Has Credit Card', ['No', 'Yes'])
is_active_member = st.sidebar.selectbox('Is Active Member', ['No', 'Yes'])

# Convert descriptive labels back to binary values
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Preparing the input data 
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
}

# Convert the input data to a DataFrame
input_data_df = pd.DataFrame(input_data)

# One Hot Encode Geography
geo_encoded = onehotencoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))

# Combine the one-hot encoded columns with input data
input_data_df = pd.concat([input_data_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the data
input_data_scaled = scaler.transform(input_data_df)

# Button to make prediction
if st.sidebar.button('Predict Churn'):
    with st.spinner('Making prediction...'):
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

    # Display the result
    st.markdown("---")
    st.subheader("🧩 Prediction Result")
    st.write(f'**Churn Probability:** {prediction_proba:.2f}')

    # Differentiate the success and failure messages
    if prediction_proba > 0.5:
        st.markdown("<div style='padding: 20px; background-color: #FF6347; color: white; border-radius: 10px; text-align: center;'>"
                    "🚨 The customer is **likely to churn**! 🚨"
                    "</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding: 20px; background-color: #32CD32; color: white; border-radius: 10px; text-align: center;'>"
                    "🎉 The customer is **not likely to churn**! 🎉"
                    "</div>", unsafe_allow_html=True)

# Footer with a logo and credits
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed by <strong>Sarthak Patel</strong>", unsafe_allow_html=True)
# Add an image or logo at the top if needed
# st.image('path_to_your_logo.png', width=100)

# Add some additional styling
st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)
