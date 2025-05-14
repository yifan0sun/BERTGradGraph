from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import (
    AutoModel, AutoModelForMaskedLM,
    AutoTokenizer, BertTokenizer,
)
import torch

app = FastAPI()

MODEL_MAP = {
    "BERT": "bert-base-uncased",
    "BART": "facebook/bart-base",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
}

class ModelRequest(BaseModel):
    model: str
    task: str
    sentence: str

@app.post("/load_model")
def load_model(req: ModelRequest):
    model_name = MODEL_MAP.get(req.model)
    if not model_name:
        return {"error": "Unknown model"}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(req.sentence)

    # Load model for MLM task (only BERT for now)
    if req.task == "MLM" and req.model == "BERT":
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        head_type = "MLM"
    else:
        model = AutoModel.from_pretrained(model_name)
        head_type = "DUMMY"

    # Count attention layers (encoder only)
    attention_layers = [
        module for module in model.modules()
        if isinstance(module, torch.nn.modules.activation.MultiheadAttention)
        or "attention.self" in str(type(module))  # Works for BERT/RoBERTa
    ]
    n_layers = len(attention_layers)

    return {
        "model": req.model,
        "task": req.task,
        "tokens": tokens,
        "num_layers": n_layers,
        "head": head_type,
    }
