import streamlit as st
from chatbot.model_loader import load_llm

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 AI Symptom Checker (LLM Only)")

# Load LLM
@st.cache_resource
def init_llm():
    return load_llm("google/flan-t5-small")

llm = init_llm()

# Prediction function with better zero-shot prompt
def predict(symptoms):
    symptoms_text = ", ".join(symptoms)
    prompt = f"""
You are a professional medical assistant. A patient has the following symptoms: {symptoms_text}.
Predict 1-3 possible diseases that could cause these symptoms.
For each disease, provide a concise one-line advice.
Do NOT repeat the instructions, just give output.
Include disclaimer: '⚠️ This is not a medical diagnosis. Please consult a doctor.'

Format strictly like this:
Predicted Disease(s): <disease1>, <disease2>, <disease3>
Advice: <short advice>
"""
    result = llm(prompt)
    # Extract the generated text
    text = result[0]["generated_text"].strip()
    # Remove any repeated instructions by filtering lines starting with "Predict" or "Include"
    lines = [line for line in text.split("\n") if not line.lower().startswith(("predict", "include"))]
    return "\n".join(lines).strip()

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
