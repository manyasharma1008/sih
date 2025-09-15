import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests  

@st.cache_data
def load_data():
    df = pd.read_csv("Training.csv")
    with open("symptoms.json") as f:
        symptoms_dict = json.load(f)
    with open("description_list.json") as f:
        description_list = json.load(f)
    with open("precautionDictionary.json") as f:
        precaution_dict = json.load(f)
    return df, symptoms_dict, description_list, precaution_dict

df, symptoms_dict, description_list, precaution_dict = load_data()
X = df.drop("prognosis", axis=1)
y = df["prognosis"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y_encoded)

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")

st.title("🩺 AI-Powered Symptom Checker")
st.write("Enter your symptoms and get possible conditions with confidence scores & precautions.")

selected_symptoms = st.multiselect(
    "Select Symptoms",
    list(symptoms_dict.keys()),
    placeholder="Type to search symptoms..."
)

if st.button("🔍 Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = [0] * len(X.columns)
        for s in selected_symptoms:
            if s in X.columns:
                input_vector[X.columns.get_loc(s)] = 1

        input_vector = np.array(input_vector).reshape(1, -1)
        probs = clf.predict_proba(input_vector)[0]
        top_indices = np.argsort(probs)[::-1][:3]

        st.subheader("🔎 Top Predictions")
        for idx in top_indices:
            disease = le.inverse_transform([idx])[0]
            confidence = probs[idx] * 100
            st.markdown(f"**{disease}** — {confidence:.2f}% confidence")

            if disease in description_list:
                st.write(f"📝 {description_list[disease]}")

            if disease in precaution_dict:
                st.write("💡 **Precautions:**")
                for p in precaution_dict[disease]:
                    st.write(f"- {p}")

        st.subheader("🤖 AI Health Suggestion")
        prompt = f"User reports these symptoms: {', '.join(selected_symptoms)}. Suggest possible conditions and care steps in 3 lines."
        try:
            response = requests.post(
                "https://free-llm-api.onrender.com/predict",
                json={"prompt": prompt},
                timeout=10
            )
            if response.status_code == 200:
                st.info(response.json().get("output", "AI response unavailable."))
            else:
                st.warning("Could not fetch AI response.")
        except:
            st.warning("AI API not reachable.")
