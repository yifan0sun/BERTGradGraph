from fastapi import FastAPI, Request
from pydantic import BaseModel

import torch

from fastapi.middleware.cors import CORSMiddleware

from ROBERTAmodel import *
from BERTmodel import *
from DISTILLBERTmodel import *

VISUALIZER_CLASSES = {
    "BERT": BERTVisualizer,
    "RoBERTa": RoBERTaVisualizer,
    "DistilBERT": DistilBERTVisualizer,
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
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
}

class LoadModelRequest(BaseModel):
    model: str
    sentence: str
    task:str

class GradAttnModelRequest(BaseModel):
    model: str
    task: str
    sentence: str
    selected_layer: int = -1

class PredModelRequest(BaseModel):
    model: str
    sentence: str
    task:str


@app.post("/load_model")
def load_model(req: LoadModelRequest):
    print(f"\n--- /load_model request received ---")
    print(f"Model: {req.model}")
    print(f"Sentence: {req.sentence}")
    print(f"Task: {req.task}")

    vis_class = VISUALIZER_CLASSES.get(req.model)
    if vis_class is None:
        return {"error": f"Unknown model: {req.model}"}

    print("instantiating visualizer")
    try:
        vis = vis_class(task=req.task.lower())
        print("Visualizer instantiated")
    except Exception as e:
        print("Visualizer init failed:", e)
        return {"error": f"Instantiation failed: {str(e)}"}

    print('tokenizing')
    try:
        token_output = vis.tokenize(req.sentence)
        print("Tokenization successful:", token_output["tokens"])
    except Exception as e:
        print("Tokenization failed:", e)
        return {"error": f"Tokenization failed: {str(e)}"}

    print('response ready')    
    response = {
        "model": req.model,
        "tokens": token_output['tokens'],
        "num_layers": vis.num_attention_layers,
    }
    print("load model successful")
    print(response)
    return response





@app.post("/predict_model")
def predict_model(req: PredModelRequest):
    print(f"\n--- /predict_model request received ---")
    print(f"Model: {req.model}")
    print(f"Task: {req.task}")
    print(f"sentence: {req.sentence}")

    print('instantiating')
    try:
        
        vis_class = VISUALIZER_CLASSES.get(req.model)
        if vis_class is None:
            return {"error": f"Unknown model: {req.model}"}
        print("Visualizer instantiated(1/2)")
    except Exception as e:
        print("Visualizer init failed:", e)
        return {"error": f"Instantiation failed: {str(e)}"}

    try:
        
        vis = vis_class(task=req.task.lower())
        print("Visualizer instantiated (2/2)")
    except Exception as e:
        print("Visualizer init failed:", e)
        return {"error": f"Instantiation failed: {str(e)}"}

    

    
    print('Run prediction')
    try:
        decoded, top_probs = vis.predict(req.task.lower(), req.sentence)
    except Exception as e:
        decoded, top_probs = "error", e
        print(e)

    print('response ready')
    response = {
        "decoded": decoded,
        "top_probs": top_probs.tolist(),
    }
    print("predict model successful")
    print(response)
    return response



@app.post("/get_grad_attn_matrix")
def get_grad_attn_matrix(req: GradAttnModelRequest):
    try:
        print(f"\n--- /get_grad_matrix request received ---")
        print(f"Model: {req.model}")
        print(f"Task: {req.task}")
        print(f"sentence: {req.sentence}")
        print(f"Selected layer: {req.selected_layer}")
        

        vis_class = VISUALIZER_CLASSES.get(req.model)
        if vis_class is None:
            return {"error": f"Unknown model: {req.model}"}

        vis = vis_class(task=req.task.lower())


        
        print("run function")
        try:
            grad_matrix, attn_matrix = vis.get_grad_attn_matrix(req.task.lower(), req.sentence, req.selected_layer)
        except Exception as e:
            print("Exception during grad/attn computation:", e)
            grad_matrix, attn_matrix = e,e

    
        response = {
            "grad_matrix": grad_matrix.tolist(),
            "attn_matrix": attn_matrix.tolist(),
        }
        print(response)
        return response
    except Exception as e:
        print("SERVER EXCEPTION:", e)
        return {"error": str(e)}
    

