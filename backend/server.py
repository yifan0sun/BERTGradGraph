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

    
    # Run prediction
    try:
        pred_token, loss, grads = vis.predict(req.task.lower(), req.sentence)
    except Exception as e:
        pred_token, loss,grads = "error", e,"_"

    debug_info = {
        "prediction": pred_token,
        "loss": loss,
        'grads':grads
    }
    print(debug_info)
    # Optionally get attention gradients
    if req.selected_layer >= 0 and req.selected_token >= 0:
        try:
            attn_map, grad_matrix, tokens = vis.get_attention_gradient_matrix(req.task.lower(), req.sentence, req.selected_layer)
            debug_info["grad_norms"] = grad_matrix.tolist()
        except Exception as e:
            debug_info["grad_error"] = str(e)

    response = {
        "model": req.model,
        "task": req.task,
        "tokens": token_output['tokens'],
        "num_layers": vis.num_attention_layers,
        "head": "MLM" if req.task == "MLM" and req.model == "BERT" else "DUMMY",
        "mask_token": vis.tokenizer.mask_token,
        "debug": debug_info
    }
    print(response)
    return response