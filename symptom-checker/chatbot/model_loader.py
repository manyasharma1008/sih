from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_llm(model_name="google/flan-t5-small"):
    """
    Load a CPU/GPU-friendly LLM for disease prediction.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",  # GPU if available
        max_new_tokens=200
    )
    return llm_pipeline
