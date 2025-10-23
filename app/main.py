import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoder
model = joblib.load("models/model.pkl")
mlb = joblib.load("models/mlb.pkl")
desc_df = pd.read_csv("data/symptom_Description.csv")
precaution_df = pd.read_csv("data/symptom_precaution.csv")

# Streamlit UI setup
st.set_page_config(page_title="Medical Diagnosis Prediction System", layout="wide")
st.title("🩺 Medical Diagnosis Prediction System")
st.markdown("Select symptoms below to predict the top 3 most probable diseases along with description and precautions.")

# Get list of all symptoms
all_symptoms = mlb.classes_
selected = st.multiselect("🔍 Select Symptoms", sorted(all_symptoms))

if st.button("🧠 Predict Diseases"):
    if not selected:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Transform input
        input_vector = mlb.transform([selected])
        probas = model.predict_proba(input_vector)[0]
        
        # Get top 3 predicted diseases
        top_indices = probas.argsort()[-3:][::-1]
        top_diseases = [(model.classes_[i], probas[i]) for i in top_indices]

        st.success("### ✅ Top 3 Predicted Diseases:")

        for disease, prob in top_diseases:
            st.write(f"**🦠 {disease}** — _Probability: {round(prob * 100, 2)}%_")
            
            # Show description
            desc_row = desc_df[desc_df["Disease"].str.lower() == disease.lower()]
            if not desc_row.empty:
                st.info(f"📝 **Description:** {desc_row['Description'].values[0]}")

            # Show precautions
            prec_row = precaution_df[precaution_df["Disease"].str.lower() == disease.lower()]
            if not prec_row.empty:
                st.markdown("🛡️ **Precautions:**")
                for i in range(1, 5):
                    val = prec_row[f"Precaution_{i}"].values[0]
                    if pd.notna(val):
                        st.markdown(f"- {val}")
            st.markdown("---")
