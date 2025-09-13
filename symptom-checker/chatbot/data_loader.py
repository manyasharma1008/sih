import json

def load_symptom_dataset(path="data/symptoms.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data
