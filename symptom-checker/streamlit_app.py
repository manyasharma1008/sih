import streamlit as st
from chatbot.model_loader import load_llm

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 AI Symptom Checker (LLM Only)")

# Load LLM
@st.cache_resource
def init_llm():
    return load_llm("microsoft/phi-3-mini-4k-instruct")

llm = init_llm()

# Prediction function
def predict(symptoms):
    symptoms_text = ", ".join(symptoms)
    prompt = f"""
You are a helpful medical assistant.
Patient symptoms: {symptoms_text}

Suggest 1-3 possible diseases that could cause these symptoms.
For each disease, provide a one-line advice.
Always include this disclaimer at the end: '⚠️ This is not a medical diagnosis. Please consult a doctor.'

Format like this:
Predicted Disease(s): <disease1>, <disease2>
Advice: <short advice>
"""
    result = llm(prompt)
    return result[0]["generated_text"].strip()

# Streamlit UI
user_input = st.text_input("Enter your symptoms (comma-separated):")
if st.button("Predict"):
    if user_input:
        symptoms = [s.strip() for s in user_input.split(",") if s.strip()]
        if not symptoms:
            st.error("Please enter at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                output = predict(symptoms)
            st.info(output)
    else:
        st.error("Please enter at least one symptom.")
