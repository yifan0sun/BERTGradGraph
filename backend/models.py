from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import BertTokenizer, BertModel, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import (
    BertTokenizer, BertModel,
    DataCollatorForLanguageModeling
)
import torch.optim as optim

import os
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import RobertaModel, RobertaTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import BartModel, BartTokenizer



class TransformerVisualizer():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def predict(self, task, text):
        return task, text,1
    

    def get_attention_gradient_matrix(self, task, text, target_layer):
        return task, text,target_layer,1
    
class BERTVisualizer(TransformerVisualizer):
    def __init__(self):
        super().__init__()  
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()
        self.num_attention_layers = len(self.model.encoder.layer)
        self.model.to(self.device)
        
    def tokenize(self, text):
        encoded = self.tokenizer(text, return_tensors='pt', return_attention_mask=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }

 
    def predict(self, task, text):
        print(task,text)
        asdf
        if task != 'mlm':
            return task, text, 1
        
        # Tokenize and find [MASK] position
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        mask_index = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1].item()

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        embedding_layer = self.model.embeddings.word_embeddings
        inputs_embeds = embedding_layer(inputs['input_ids'])
        inputs_embeds.requires_grad_()
        inputs_embeds.retain_grad()

        # Forward through BERT encoder
        
        hidden_states = self.model(inputs_embeds=inputs_embeds,
                                  attention_mask=inputs['attention_mask']).last_hidden_state

        # Predict logits via MLM head
        logits = self.head(hidden_states)
        mask_logits = logits[0, mask_index]

        # Get top token and compute loss
        pred_token_id = torch.argmax(mask_logits).item()
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(mask_logits, inputs["input_ids"][0, mask_index])

        predicted_token = self.tokenizer.decode(pred_token_id)

        # Backpropagate if you want gradients
        loss.backward()
        grads = inputs_embeds.grad[0]
         



        return predicted_token, loss, grads
    


    def get_attention_gradient_matrix(self, task, text, target_layer):
        fdsda
        print(task,text,target_layer)
        asdf
        if task != 'mlm':
            return task, text,target_layer
        
        """
        For a given text and attention layer, compute the gradient norm matrix:
        grad_matrix[i,j] = || d attention_received_by_token[j] / d embedding[i] ||
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt')
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Input embeddings with grad
        embedding_layer = self.model.embeddings.word_embeddings
        inputs_embeds = embedding_layer(inputs["input_ids"])
        inputs_embeds.requires_grad_()

        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            output_attentions=True
        )
        attentions = outputs.attentions  # list of [1, heads, seq, seq]

        # Optional: store average attentions per layer
        mean_attns = [a.squeeze(0).mean(dim=0).detach().cpu() for a in attentions]

        # Work with target layer only
        attn = attentions[target_layer].squeeze(0).mean(dim=0)  # shape: [seq, seq]
        seq_len = attn.shape[0]

        grad_norms_list = []

        for k in range(seq_len):
            scalar = attn[k].sum()  # total attention received by token k

            # Compute gradient: d scalar / d inputs_embeds
            grad = torch.autograd.grad(scalar, inputs_embeds, retain_graph=True)[0].squeeze(0)  # shape: [seq, hidden]
            grad_norms = grad.norm(dim=1)  # shape: [seq]
            grad_norms_list.append(grad_norms.unsqueeze(1))  # shape: [seq, 1]

        grad_matrix = torch.cat(grad_norms_list, dim=1)  # shape: [seq, seq]
        return mean_attns, grad_matrix, tokens

class RoBERTaVisualizer(TransformerVisualizer):
    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.model.eval()
        self.num_attention_layers = len(self.model.encoder.layer)
        self.model.to(self.device)

    def tokenize(self, text):
        encoded = self.tokenizer(text, return_tensors='pt', return_attention_mask=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
    

class DistilBERTVisualizer(TransformerVisualizer):
    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.eval()
        self.num_attention_layers = len(self.model.transformer.layer)
        self.model.to(self.device)

    def tokenize(self, text):
        encoded = self.tokenizer(text, return_tensors='pt', return_attention_mask=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
    


class BARTVisualizer(TransformerVisualizer):
    def __init__(self):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.model = BartModel.from_pretrained('facebook/bart-base')
        self.model.eval()
        self.num_attention_layers = len(self.model.encoder.layers)
        self.model.to(self.device)

    def tokenize(self, text):
        encoded = self.tokenizer(text, return_tensors='pt', return_attention_mask=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
    


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