from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import (
    AutoModel, AutoModelForMaskedLM,
    AutoTokenizer, BertTokenizer,
)
import torch

from fastapi.middleware.cors import CORSMiddleware

from models import *

VISUALIZER_CLASSES = {
    "BERT": BERTVisualizer,
    "RoBERTa": RoBERTaVisualizer,
    "DistilBERT": DistilBERTVisualizer,
    "BART": BARTVisualizer,
}
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    selected_layer: int = -1
    selected_token: int = -1


@app.post("/load_model")
def load_model(req: ModelRequest):
    print(f"\n--- /load_model request received ---")
    print(f"Model: {req.model}")
    print(f"Task: {req.task}")
    print(f"Sentence: {req.sentence}")
    print(f"Selected layer: {req.selected_layer}, token: {req.selected_token}")

    vis_class = VISUALIZER_CLASSES.get(req.model)
    if vis_class is None:
        return {"error": f"Unknown model: {req.model}"}

    vis = vis_class()

    try:
        token_output = vis.tokenize(req.sentence)
    except Exception as e:
        return {"error": f"Tokenization failed: {str(e)}"}

    
    response = {
        "model": req.model,
        "task": req.task,
        "tokens": token_output['tokens'],
        "num_layers": vis.num_attention_layers,
        "head": "MLM" if req.task == "MLM" and req.model == "BERT" else "DUMMY",
        "mask_token": vis.tokenizer.mask_token,
    }
    print(response)
    return response





@app.post("/predict_model")
def predict_model(req: ModelRequest):
    print(f"\n--- /predict_model request received ---")
    print(f"Model: {req.model}")
    print(f"Task: {req.task}")
    print(f"Sentence: {req.sentence}")
    print(f"Selected layer: {req.selected_layer}, token: {req.selected_token}")

    vis_class = VISUALIZER_CLASSES.get(req.model)
    if vis_class is None:
        return {"error": f"Unknown model: {req.model}"}

    vis = vis_class()

    try:
        token_output = vis.tokenize(req.sentence)
    except Exception as e:
        return {"error": f"Tokenization failed: {str(e)}"}

    
    # Run prediction
    try:
        decoded, top_probs, grads = vis.predict(req.task.lower(), req.sentence)
    except Exception as e:
        decoded, top_probs, grads = "error", e,"_"

  
    response = {
        "decoded": decoded,
        "top_probs": top_probs,
        'grads':grads
    }
    print(response)
    return response