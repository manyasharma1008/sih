from .model_loader import load_llm

class SymptomChatbot:
    def __init__(self, model_name="google/flan-t5-base"):
        self.llm = load_llm(model_name)

    def get_response(self, user_symptoms):
        """
        Predict diseases and advice using only LLM (no hardcoded database).
        """
        symptoms_text = ", ".join(user_symptoms)
        prompt = f"""
You are an expert medical assistant.
Patient symptoms: {symptoms_text}

List 1-3 possible diseases these symptoms may indicate.
Provide one-line advice for each disease.
Write in natural language, do NOT use placeholders.
Always include at the end: '⚠️ This is not a medical diagnosis. Please consult a doctor.'

Example output:
Predicted Disease(s): Influenza, Common Cold
Advice: Rest, drink fluids, and consult a doctor if symptoms persist.
"""
        result = self.llm(prompt)
        return result[0]["generated_text"].strip()
