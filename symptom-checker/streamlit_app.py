import streamlit as st
import pandas as pd
import json
import random
from model_loader import load_llm

@st.cache_data
def load_data():
    with open("symptoms.json", "r") as f:
        symptoms_dict = json.load(f)

    try:
        df = pd.read_csv("Training.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Symptom", "Disease"])

    return df, symptoms_dict


def llm_predict(symptoms):
    """
    Use your local LLM to predict disease and precautions.
    """
    llm = load_llm(model_name="google/flan-t5-small")

    prompt = f"""
    You are a medical assistant. Given these symptoms: {', '.join(symptoms)},
    predict the most likely disease and provide 2-3 possible precautions.
    Answer in a short and clear format.
    """

    result = llm(prompt)
    return result

st.title("🧠 Symptom Checker with AI")

df, symptoms_dict = load_data()

user_input = st.text_input("Enter your symptoms (comma-separated):")
if st.button("Predict"):
    if user_input:
        symptoms = [s.strip().lower() for s in user_input.split(",")]
        matched_disease = None

        for disease, disease_symptoms in symptoms_dict.items():
            if all(symptom in [s.lower() for s in disease_symptoms] for symptom in symptoms):
                matched_disease = disease
                break

        if matched_disease:
            st.success(f"**Predicted Disease:** {matched_disease}")
        else:
            st.warning("No exact match found. Using AI model...")
            ai_result = llm_predict(symptoms)
            st.info(ai_result)
    else:
        st.error("Please enter at least one symptom.")
