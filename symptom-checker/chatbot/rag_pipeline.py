from .model_loader import load_llm

class SymptomChatbot:
    def __init__(self, model_name="microsoft/phi-3-mini-4k-instruct"):
        # Load the LLM
        self.llm = load_llm(model_name)

    def get_response(self, user_symptoms):
        """
        Predict diseases for any symptoms using LLM.
        """
        symptoms_text = ", ".join(user_symptoms)

        prompt = f"""
You are a helpful and concise medical assistant.
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
