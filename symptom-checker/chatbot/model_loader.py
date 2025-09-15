from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Optional: Hugging Face token if required for private models
HF_API_TOKEN = None  # Replace with your token if needed

def load_llm(model_name="microsoft/phi-3-mini-4k-instruct"):
    """
    Load a CPU/GPU-friendly LLM for disease prediction.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU (-1), GPU if available can be device=0
        max_new_tokens=150,
        use_auth_token=HF_API_TOKEN
    )
    return llm_pipeline

def load_embedding_model():
    """
    Placeholder function for embeddings (not needed for LLM-only version).
    """
    return None
