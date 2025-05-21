import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

 

import os
from transformers import DistilBertModel, DistilBertTokenizer
from models import TransformerVisualizer

from transformers import (
     DistilBertTokenizer,
    DistilBertForMaskedLM, DistilBertForSequenceClassification
)    

class DistilBERTVisualizer(TransformerVisualizer):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if self.task == 'mlm':
            self.model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        elif self.task == 'sst':
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        elif self.task == 'mnli':
            self.model = DistilBertForSequenceClassification.from_pretrained("ynie/distilbert-base-uncased-snli-mnli-scitail-mednli")

        else:
            raise NotImplementedError("Task not supported for DistilBERT")
        

        self.model.eval()
        self.num_attention_layers = len(self.model.distilbert.transformer.layer)

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
        if task  == 'mlm':
            inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True).to(self.device)
            mask_index = (inputs['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

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
            premise, hypothesis = text
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).squeeze()

            labels = ["entailment", "neutral", "contradiction"]
            return labels, probs

        else:
            raise NotImplementedError(f"Task '{task}' not supported for DistilBERT")


    def get_grad_attn_matrix(self, task, sentence, target_layer):
        print(task, sentence, target_layer)


        print('Tokenize')
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=False, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
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