# chatbot/rag_pipeline.py
import faiss
from .model_loader import load_embedding_model, load_llm

class SymptomChatbot:
    def __init__(self, model_name="google/flan-t5-small"):
        # Embedder (optional)
        self.embedder = load_embedding_model()
        self.llm = load_llm(model_name)
        self.index = None
        self.symptom_texts = []

    def build_index(self, symptoms_list):
        """
        Optional: build FAISS index if you have a symptom dataset
        """
        self.symptom_texts = [" ".join(s) for s in symptoms_list]
        embeddings = self.embedder.encode(self.symptom_texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def get_response(self, user_symptoms):
        """
        Generate disease prediction using only LLM
        """
        symptoms_text = ", ".join(user_symptoms)
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
        result = self.llm(prompt)
        return result[0]["generated_text"].strip()
