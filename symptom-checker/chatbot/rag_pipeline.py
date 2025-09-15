import faiss
import numpy as np
from .model_loader import load_embedding_model, load_llm

# Hardcoded symptom-disease database
SYMPTOM_DB = [
    {"symptoms": ["fever", "cough", "sore throat"],
     "diseases": ["Common Cold", "Flu", "COVID-19"],
     "advice": "Stay hydrated, rest, and consult a doctor if symptoms persist more than 3 days."},
    {"symptoms": ["chest pain", "shortness of breath", "dizziness"],
     "diseases": ["Heart Attack", "Angina", "Panic Attack"],
     "advice": "Seek immediate medical help. Call emergency services."},
    {"symptoms": ["headache", "sensitivity to light", "stiff neck"],
     "diseases": ["Migraine", "Meningitis"],
     "advice": "If sudden and severe, seek emergency medical attention."}
]

class SymptomChatbot:
    def __init__(self, model_name="google/flan-t5-small"):
        # Load data and models
        self.data = SYMPTOM_DB
        self.embedder = load_embedding_model()
        self.llm = load_llm(model_name)
        self.index, self.symptom_texts = self._build_index()

    def _build_index(self):
        """
        Builds FAISS index for symptom similarity search.
        """
        symptom_texts = [" ".join(item["symptoms"]) for item in self.data]
        embeddings = self.embedder.encode(symptom_texts, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, symptom_texts

    def get_response(self, user_symptoms):
        """
        Returns a text response for given user symptoms.
        """
        symptoms_lower = [s.lower() for s in user_symptoms]

        # 1️⃣ Check for exact match first
        for entry in self.data:
            if all(s in [x.lower() for x in entry["symptoms"]] for s in symptoms_lower):
                diseases = ", ".join(entry["diseases"])
                advice = entry["advice"]
                return f"**Predicted Disease(s):** {diseases}\n💡 Advice: {advice}\n⚠️ This is not a medical diagnosis. Please consult a doctor."

        # 2️⃣ If no exact match, use FAISS + LLM
        query_vec = self.embedder.encode([" ".join(user_symptoms)], convert_to_numpy=True)
        D, I = self.index.search(query_vec, 1)
        matched = self.data[I[0][0]]

        prompt = f"""
You are a helpful medical assistant.

Patient symptoms: {', '.join(user_symptoms)}
Most similar known symptoms: {', '.join(matched['symptoms'])}
Possible conditions: {', '.join(matched['diseases'])}
Suggested advice: {matched['advice']}

Please give a **short, clear response** in this format:
Predicted Disease(s): <disease names>
Advice: <short advice>
Include the disclaimer: "This is not a medical diagnosis. Please consult a doctor."
"""

        response = self.llm(prompt)
        # Ensure string output
        return response[0]["generated_text"].strip()
