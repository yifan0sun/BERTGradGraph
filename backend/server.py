from fastapi import FastAPI, Request
from pydantic import BaseModel
from pathlib import Path

import torch
from fastapi import UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware

from ROBERTAmodel import *
from BERTmodel import *
from DISTILLBERTmodel import *

import os
import zipfile
import shutil
from fastapi import Form
from fastapi import UploadFile, File, Form
from pathlib import Path

# uvicorn server:app --host 0.0.0.0 --port 7860 --reload

#localhost version
# python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload



VISUALIZER_CLASSES = {
    "BERT": BERTVisualizer,
    "RoBERTa": RoBERTaVisualizer,
    "DistilBERT": DistilBERTVisualizer,
}

VISUALIZER_CACHE = {}
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
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
    hypothesis:str

class GradAttnModelRequest(BaseModel):
    model: str
    task: str
    sentence: str
    hypothesis:str
    maskID: int | None = None

class PredModelRequest(BaseModel):
    model: str
    sentence: str
    task:str
    hypothesis:str
    maskID: int | None = None
 
 

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    save_path = f"/data/models/{file.filename}"  # or wherever your disk is mounted
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"status": "uploaded", "path": save_path}



@app.post("/load_model")
def load_model(req: LoadModelRequest):
    print(f"\n--- /load_model request received ---")
    print(f"Model: {req.model}")
    print(f"Sentence: {req.sentence}")
    print(f"Task: {req.task}")
    print(f"hypothesis: {req.hypothesis}")


    if req.model in VISUALIZER_CACHE:
        del VISUALIZER_CACHE[req.model]
    torch.cuda.empty_cache()

    vis_class = VISUALIZER_CLASSES.get(req.model)
    if vis_class is None:
        return {"error": f"Unknown model: {req.model}"}

    print("instantiating visualizer")
    try:
        vis = vis_class(task=req.task.lower())
        print(vis)
        VISUALIZER_CACHE[req.model] = vis
        print("Visualizer instantiated")
    except Exception as e:
        print("Visualizer init failed:", e)
        return {"error": f"Instantiation failed: {str(e)}"}

    print('tokenizing')
    try:
        if req.task.lower() == 'mnli':
            token_output = vis.tokenize(req.sentence, hypothesis=req.hypothesis)
        else:
            token_output = vis.tokenize(req.sentence)
        print("0 Tokenization successful:", token_output["tokens"])
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
    print(f"predict: Model: {req.model}")
    print(f"predict: Task: {req.task}")
    print(f"predict: sentence: {req.sentence}")
    print(f"predict: hypothesis: {req.hypothesis}")
    print(f"predict: maskID: {req.maskID}")



    print('predict: instantiating')
    try:
        vis_class = VISUALIZER_CLASSES.get(req.model)
        if vis_class is None:
            return {"error": f"Unknown model: {req.model}"}
        #if any(p.device.type == 'meta' for p in vis.model.parameters()):
        #    vis.model = torch.nn.Module.to_empty(vis.model, device=torch.device("cpu"))
    
        vis = vis_class(task=req.task.lower())
        VISUALIZER_CACHE[req.model] = vis
        print("Model reloaded and cached.")
    except Exception as e:
        return {"error": f"Failed to reload model: {str(e)}"}
    
    print('predict: meta stuff')
    

    
    print('predict: Run prediction')
    try:
        if req.task.lower() == 'mnli':
            decoded, top_probs = vis.predict(req.task.lower(), req.sentence, hypothesis=req.hypothesis)
        elif req.task.lower() == 'mlm':
            decoded, top_probs = vis.predict(req.task.lower(), req.sentence, maskID=req.maskID)
        
        else:
            decoded, top_probs = vis.predict(req.task.lower(), req.sentence)
    except Exception as e:
        decoded, top_probs = "error", e
        print(e)

    print('predict: response ready')
    response = {
        "decoded": decoded,
        "top_probs": top_probs.tolist(),
    }
    print("predict: predict model successful")
    if len(decoded) > 5:
        print([(k,v[:5]) for k,v in response.items()])
    else:
        print(response)
    return response



@app.post("/get_grad_attn_matrix")
def get_grad_attn_matrix(req: GradAttnModelRequest):
    
    try:
        print(f"\n--- /get_grad_matrix request received ---")
        print(f"grad:Model: {req.model}")
        print(f"grad:Task: {req.task}")
        print(f"grad:sentence: {req.sentence}")
        print(f"grad: hypothesis: {req.hypothesis}")
        print(f"predict: maskID: {req.maskID}")
        
        
        
        try:
            vis_class = VISUALIZER_CLASSES.get(req.model)
            if vis_class is None:
                return {"error": f"Unknown model: {req.model}"}
            #if any(p.device.type == 'meta' for p in vis.model.parameters()):
            #    vis.model = torch.nn.Module.to_empty(vis.model, device=torch.device("cpu"))
            vis = vis_class(task=req.task.lower())
            VISUALIZER_CACHE[req.model] = vis
            print("Model reloaded and cached.")
        except Exception as e:
            return {"error": f"Failed to reload model: {str(e)}"} 


        
        print("run function")
        try:
            if req.task.lower()=='mnli':
                grad_matrix, attn_matrix = vis.get_all_grad_attn_matrix(req.task.lower(), req.sentence,hypothesis=req.hypothesis)
            elif req.task.lower()=='mlm':
                grad_matrix, attn_matrix = vis.get_all_grad_attn_matrix(req.task.lower(), req.sentence,maskID=req.maskID)
            else:
                grad_matrix, attn_matrix = vis.get_all_grad_attn_matrix(req.task.lower(), req.sentence)
        except Exception as e:
            print("Exception during grad/attn computation:", e)
            grad_matrix, attn_matrix = e,e

    
        response = {
            "grad_matrix": grad_matrix,
            "attn_matrix": attn_matrix,
        }
        print('grad attn successful')
        return response
    except Exception as e:
        print("SERVER EXCEPTION:", e)
        return {"error": str(e)}
    








##################################################



@app.get("/ping")
def ping():
    return {"message": "pong"}

    

@app.post("/upload_to_path")
async def upload_to_path(
    file: UploadFile = File(...),
    dest_path: str = Form(...)  # e.g., "models/model.pt"
):
    full_path = Path("/data") / dest_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_path, "wb") as f:
        f.write(await file.read())

    return {"status": "uploaded", "path": str(full_path)}




@app.post("/make_dir")
def make_directory(
    dir_path: str = Form(...)  # e.g., "logs/test_run"
):
    full_dir = Path("/data") / dir_path
    full_dir.mkdir(parents=True, exist_ok=True)
    return {"status": "created", "directory": str(full_dir)}



@app.get("/list_data")
def list_data():
    base_path = Path("/data")
    all_items = []

    for path in base_path.rglob("*"):  # recursive glob
        all_items.append({
            "path": str(path.relative_to(base_path)),
            "type": "dir" if path.is_dir() else "file",
            "size": path.stat().st_size if path.is_file() else None
        })

    return {"items": all_items}







"""
@app.post("/purge_data_123456789")
def purge_data():
    base_path = Path("/data")
    if not base_path.exists():
        return {"status": "error", "message": "/data does not exist"}

    deleted = []

    for child in base_path.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            deleted.append(str(child.name))
        except Exception as e:
            deleted.append(f"FAILED: {child.name} ({e})")

    return {
        "status": "done",
        "deleted": deleted,
        "total": len(deleted)
    }




"""




"""
if __name__ == "__main__":
     
    print('rim ')
    BERTVisualizer('mlm')
    BERTVisualizer('mnli')
    BERTVisualizer('sst')


    RoBERTaVisualizer('mlm')
    RoBERTaVisualizer('mnli')
    RoBERTaVisualizer('sst')


    DistilBERTVisualizer('mlm')
    DistilBERTVisualizer('mnli')
    DistilBERTVisualizer('sst')
"""
 