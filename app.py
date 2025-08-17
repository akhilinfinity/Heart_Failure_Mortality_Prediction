import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def load_model():
  
    model_package = joblib.load('best_heart_failure_model_random_forest.pkl')  
    return model_package


try:
    model_package = load_model()
    model = model_package['model']
    scaler = model_package['scaler']
    model_name = model_package['model_name']
    st.success(f"✅ Model loaded: {model_name}")
except:
    st.error("❌ Could not load model. Make sure the model file is in the same directory.")

# App title
st.title("🏥 Heart Failure Mortality Prediction")
st.write("This app predicts the risk of mortality for heart failure patients using machine learning.")

# Sidebar for input
st.sidebar.header("Patient Information")

# Input fields
age = st.sidebar.slider("Age", 40, 95, 65)
anaemia = st.sidebar.selectbox("Anaemia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
creatinine_phosphokinase = st.sidebar.number_input("Creatinine Phosphokinase (mcg/L)", 23, 7861, 250)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 14, 80, 38)
high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
platelets = st.sidebar.number_input("Platelets (kiloplatelets/mL)", 25100, 850000, 263000)
serum_creatinine = st.sidebar.number_input("Serum Creatinine (mg/dL)", 0.5, 9.4, 1.4)
serum_sodium = st.sidebar.slider("Serum Sodium (mEq/L)", 113, 148, 137)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
smoking = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
time = st.sidebar.slider("Follow-up Period (days)", 4, 285, 130)

# Create prediction button
if st.sidebar.button("🔮 Predict Risk"):
    # Create patient data
    patient_data = pd.DataFrame({
        'age': [age],
        'anaemia': [anaemia],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [diabetes],
        'ejection_fraction': [ejection_fraction],
        'high_blood_pressure': [high_blood_pressure],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [sex],
        'smoking': [smoking],
        'time': [time]
    })
    
    # Make prediction
    try:
        if model_name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
            patient_scaled = scaler.transform(patient_data)
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0, 1]
        else:
            prediction = model.predict(patient_data)[0]
            probability = model.predict_proba(patient_data)[0, 1]
        
        # Display results
        st.header("🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK")
            else:
                st.success("✅ LOW RISK")
        
        with col2:
            st.metric("Mortality Probability", f"{probability:.1%}")
        
        # Risk interpretation
        if probability > 0.7:
            st.error("🚨 CRITICAL: Immediate medical attention recommended")
        elif probability > 0.5:
            st.warning("⚠️ HIGH: Close monitoring required")
        elif probability > 0.3:
            st.warning("⚠️ MODERATE: Regular follow-up recommended")
        else:
            st.success("✅ LOW: Continue standard care")
        
        
        st.subheader("📋 Patient Summary")
        summary_data = {
            'Feature': ['Age', 'Anaemia', 'Creatinine Phosphokinase', 'Diabetes', 'Ejection Fraction',
                       'High Blood Pressure', 'Platelets', 'Serum Creatinine', 'Serum Sodium',
                       'Sex', 'Smoking', 'Follow-up Period'],
            'Value': [f"{age} years", "Yes" if anaemia else "No", f"{creatinine_phosphokinase} mcg/L",
                     "Yes" if diabetes else "No", f"{ejection_fraction}%", "Yes" if high_blood_pressure else "No",
                     f"{platelets:,}", f"{serum_creatinine} mg/dL", f"{serum_sodium} mEq/L",
                     "Male" if sex else "Female", "Yes" if smoking else "No", f"{time} days"]
        }
        st.table(pd.DataFrame(summary_data))
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
try:
    st.sidebar.write(f"**Model**: {model_name}")
    st.sidebar.write(f"**Performance**: {model_package['best_score']:.3f} ROC-AUC")
except:
    st.sidebar.write("Model information not available")


st.markdown("---")
st.markdown("**⚠️ Medical Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice.")