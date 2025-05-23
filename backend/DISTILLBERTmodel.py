import torch
import torch.nn.functional as F

 

import os
from models import TransformerVisualizer

from transformers import (
     DistilBertTokenizer,
    DistilBertForMaskedLM, DistilBertForSequenceClassification
)    

CACHE_DIR  = "/data/hf_cache"
class DistilBERTVisualizer(TransformerVisualizer):
    def __init__(self, task):
        super().__init__()
        self.task = task

        
        TOKENIZER = 'distilbert-base-uncased'
        LOCAL_PATH = os.path.join(CACHE_DIR, "tokenizers",TOKENIZER.replace("/", "_"))
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        """
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        except Exception as e:
            self.tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER)
            self.tokenizer.save_pretrained(LOCAL_PATH)
        """


        print('finding model', self.task)
        if self.task == 'mlm':
            
            MODEL = 'distilbert-base-uncased'
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)

            self.model = DistilBertForMaskedLM.from_pretrained(  LOCAL_PATH, local_files_only=True )
            """
            try:
            except Exception as e:
                self.model = DistilBertForMaskedLM.from_pretrained(  MODEL  )
                self.model.save_pretrained(LOCAL_PATH)
            """
        elif self.task == 'sst':
            MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)

            self.model = DistilBertForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True )
            """
            try:
                self.model = DistilBertForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True )
            except Exception as e:
                self.model = DistilBertForSequenceClassification.from_pretrained(  MODEL )
                self.model.save_pretrained(LOCAL_PATH)
            """

        elif self.task == 'mnli':
            MODEL = "textattack_distilbert-base-uncased-MNLI"
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)

            
            self.model = DistilBertForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True)
            """
            try:
                self.model = DistilBertForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True)
            except Exception as e:
                self.model = DistilBertForSequenceClassification.from_pretrained(  MODEL)
                self.model.save_pretrained(LOCAL_PATH)
            """

 

        else:
            raise ValueError(f"Unsupported task: {self.task}")
        

  
 
        

        self.model.eval()
        self.num_attention_layers = len(self.model.distilbert.transformer.layer)

        self.model.to(self.device)


        
    def tokenize(self, text, hypothesis = ''):

         

        if len(hypothesis) == 0:
            encoded = self.tokenizer(text, return_tensors='pt', return_attention_mask=True,padding=False, truncation=True)
        else:
            encoded = self.tokenizer(text, hypothesis, return_tensors='pt', return_attention_mask=True,padding=False, truncation=True)


        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
     
    def predict(self, task, text, hypothesis='', maskID = 0):
        
        if task  == 'mlm':
            inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True)
            if maskID is not None and 0 <= maskID < inputs['input_ids'].size(1):
                inputs['input_ids'][0][maskID] = self.tokenizer.mask_token_id
                mask_index = maskID
            else:
                raise ValueError(f"Invalid maskID {maskID} for input of length {inputs['input_ids'].size(1)}")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            mask_logits = logits[0, mask_index]
            top_probs, top_indices = torch.topk(F.softmax(mask_logits, dim=-1), 10)
            decoded = self.tokenizer.convert_ids_to_tokens(top_indices.tolist())
            return decoded, top_probs

        elif task == 'sst':
            inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).squeeze()

            labels = ["negative", "positive"]
            return labels, probs
        elif task == 'mnli': 
            inputs = self.tokenizer(text, hypothesis, return_tensors='pt', padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).squeeze()

            labels = ["entailment", "neutral", "contradiction"]
            return labels, probs

        else:
            raise NotImplementedError(f"Task '{task}' not supported for DistilBERT")

    def get_all_grad_attn_matrix(self, task, sentence, hypothesis='', maskID = 0):
        print(task, sentence,hypothesis) 

        print('Tokenize')
        if task == 'mnli':
            inputs = self.tokenizer(sentence, hypothesis, return_tensors='pt', padding=False, truncation=True)
        elif task == 'mlm':
            inputs = self.tokenizer(sentence,  return_tensors='pt', padding=False, truncation=True)
            if maskID is not None and 0 <= maskID < inputs['input_ids'].size(1):
                inputs['input_ids'][0][maskID] = self.tokenizer.mask_token_id
            else:
                print(f"Invalid maskID {maskID} for input of length {inputs['input_ids'].size(1)}")
                raise ValueError(f"Invalid maskID {maskID} for input of length {inputs['input_ids'].size(1)}")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = self.tokenizer(sentence,  return_tensors='pt', padding=False, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(tokens)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        print('Input embeddings with grad')
        embedding_layer = self.model.distilbert.embeddings.word_embeddings
        inputs_embeds = embedding_layer(inputs["input_ids"])
        inputs_embeds.requires_grad_()

        print('Forward pass')
        outputs = self.model.distilbert(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
        )
        attentions = outputs.attentions  # list of [1, heads, seq, seq]

        print('Mean attentions per layer')
        mean_attns = [a.squeeze(0).mean(dim=0).detach().cpu() for a in attentions]
        

        
        attn_matrices_all = []
        grad_matrices_all = []
        for target_layer in range(len(attentions)):
            grad_matrix, attn_matrix = self.get_grad_attn_matrix(inputs_embeds, attentions, mean_attns, target_layer)
            grad_matrices_all.append(grad_matrix.tolist())
            attn_matrices_all.append(attn_matrix.tolist())
        return grad_matrices_all, attn_matrices_all
    

    def get_grad_attn_matrix(self,inputs_embeds, attentions, mean_attns, target_layer):
        attn_matrix = mean_attns[target_layer]
        seq_len = attn_matrix.shape[0]
        attn_layer = attentions[target_layer].squeeze(0).mean(dim=0)

        print('Computing grad norms')
        grad_norms_list = []
        for k in range(seq_len):
            scalar = attn_layer[:, k].sum()
            grad = torch.autograd.grad(scalar, inputs_embeds, retain_graph=True)[0].squeeze(0)
            grad_norms = grad.norm(dim=1)
            grad_norms_list.append(grad_norms.unsqueeze(1))

        grad_matrix = torch.cat(grad_norms_list, dim=1)
        grad_matrix = grad_matrix[:seq_len, :seq_len]
        attn_matrix = attn_matrix[:seq_len, :seq_len]

        return grad_matrix, attn_matrix



if __name__ == "__main__":
    import sys

    MODEL_CLASSES = {
        "bert": BERTVisualizer,
        "roberta": RoBERTaVisualizer,
        "distilbert": DistilBERTVisualizer,
        "bart": BARTVisualizer,
    }

    # Parse command-line args or fallback to default
    model_name = sys.argv[1] if len(sys.argv) > 1 else "bert"
    text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "The quick brown fox jumps over the lazy dog."

    if model_name.lower() not in MODEL_CLASSES:
        print(f"Supported models: {list(MODEL_CLASSES.keys())}")
        sys.exit(1)

    # Instantiate the visualizer
    visualizer_class = MODEL_CLASSES[model_name.lower()]
    visualizer = visualizer_class()

    # Tokenize
    token_info = visualizer.tokenize(text)

    # Report
    print(f"\nModel: {model_name}")
    print(f"Num attention layers: {visualizer.num_attention_layers}")
    print(f"Tokens: {token_info['tokens']}")
    print(f"Input IDs: {token_info['input_ids'].tolist()}")
    print(f"Attention mask: {token_info['attention_mask'].tolist()}")


"""
usage for debug:
python your_file.py bert "The rain in Spain falls mainly on the plain."
"""