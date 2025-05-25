import torch
import torch.nn as nn
from transformers import BertTokenizer

from models import TransformerVisualizer

from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertForSequenceClassification,
)
import torch.nn.functional as F
import os, time

CACHE_DIR  = "/data/hf_cache"
 
    
class BERTVisualizer(TransformerVisualizer):
    def __init__(self,task):
        super().__init__()  
        self.task = task
        print(task,'BERT VIS START')

        TOKENIZER = 'bert-base-uncased'
        LOCAL_PATH = os.path.join(CACHE_DIR, "tokenizers",TOKENIZER)
        

        self.tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        """
        try:
            self.tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        except Exception as e:
            self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
            self.tokenizer.save_pretrained(LOCAL_PATH)
        """


        print('finding model', self.task)
        if self.task == 'mlm':
            
            MODEL = 'bert-base-uncased'
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)
            
            self.model = BertForMaskedLM.from_pretrained(  LOCAL_PATH, local_files_only=True,   attn_implementation="eager" ).to(self.device)
            """
            try:
                self.model = BertForMaskedLM.from_pretrained(  LOCAL_PATH, local_files_only=True,   attn_implementation="eager" ).to(self.device)
            except Exception as e:
                self.model = BertForMaskedLM.from_pretrained(  MODEL,    attn_implementation="eager" ).to(self.device)
                self.model.save_pretrained(LOCAL_PATH)
            """
        elif self.task == 'sst':
            MODEL = "textattack_bert-base-uncased-SST-2"
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)

            self.model = BertForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True,  device_map=None )
            """
            try:
                self.model = BertForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True,  device_map=None )
            except Exception as e:
                self.model = BertForSequenceClassification.from_pretrained(  MODEL,    device_map=None )
                self.model.save_pretrained(LOCAL_PATH)
            """

        elif self.task == 'mnli':
            MODEL = 'textattack_bert-base-uncased-MNLI'

            
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)

            self.model = BertForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True,  device_map=None )
            """
            try:
                self.model = BertForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True,  device_map=None )
            except Exception as e:
                self.model = BertForSequenceClassification.from_pretrained(  MODEL,    device_map=None)
                self.model.save_pretrained(LOCAL_PATH)
            """

 

        else:
            raise ValueError(f"Unsupported task: {self.task}")
        

        
        print('model found')
        #self.model.to(self.device)
        print('self device junk')
        self.model.eval()
        print('self model eval')
        self.num_attention_layers = len(self.model.bert.encoder.layer)
        print('init finished')
        
    def tokenize(self, text, hypothesis = ''):
        print('TTTokenize',text,'H:', hypothesis)
        if len(hypothesis) == 0:
            encoded = self.tokenizer(text, return_tensors='pt', return_attention_mask=True)
        else:
            encoded = self.tokenizer(text, hypothesis, return_tensors='pt', return_attention_mask=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
 

    def predict(self, task, text, hypothesis='', maskID = None):
        
        print(task,text,hypothesis)
    


        if task == 'mlm':

            # Tokenize and find [MASK] position
            print('Tokenize and find [MASK] position')  
            inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True)
            if maskID is not None and 0 <= maskID < inputs['input_ids'].size(1):
                inputs['input_ids'][0][maskID] = self.tokenizer.mask_token_id
                mask_index = maskID
            else:
                raise ValueError(f"Invalid maskID {maskID} for input length {inputs['input_ids'].size(1)}")



            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings 
            embedding_layer = self.model.bert.embeddings.word_embeddings
            inputs_embeds = embedding_layer(inputs['input_ids'])

            # Forward through BERT encoder
            
            hidden_states = self.model.bert(inputs_embeds=inputs_embeds,
                                    attention_mask=inputs['attention_mask']).last_hidden_state

            # Predict logits via MLM head
            logits = self.model.cls(hidden_states)
            mask_logits = logits[0, mask_index]

            top_probs, top_indices = torch.topk(mask_logits, k=10, dim=-1)
            top_probs = F.softmax(top_probs, dim=-1)
            decoded = self.tokenizer.convert_ids_to_tokens(top_indices.tolist())
            
            return decoded, top_probs
    
        elif task == 'sst':
            print('input')
            inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True).to(self.device)
            print('output')
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # shape: [1, 2]
                probs = F.softmax(logits, dim=1).squeeze()

            labels = ["negative", "positive"]
            print('ready to return')
            return labels, probs
        
        elif task == 'mnli':
            inputs = self.tokenizer(text, hypothesis, return_tensors='pt', padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).squeeze()

            labels = ["entailment", "neutral", "contradiction"]
            return labels, probs
        

    def get_all_grad_attn_matrix(self, task, sentence, hypothesis='', maskID = 0):

        print('GET GRAD:', task,'sentence',sentence, 'hypothesis', hypothesis)
         
        
        
        print('Tokenize')
        if task == 'mnli':
            inputs = self.tokenizer(sentence, hypothesis, return_tensors='pt', padding=False, truncation=True)
        elif task == 'mlm':
            inputs = self.tokenizer(sentence,  return_tensors='pt', padding=False, truncation=True)
            if maskID is not None and 0 <= maskID < inputs['input_ids'].size(1):
                inputs['input_ids'][0][maskID] = self.tokenizer.mask_token_id
            else:
                raise ValueError(f"Invalid maskID {maskID} for input length {inputs['input_ids'].size(1)}")
        else:
            inputs = self.tokenizer(sentence,  return_tensors='pt', padding=False, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print(inputs['input_ids'].shape)
        print(tokens,len(tokens))
        print('Input embeddings with grad')
        embedding_layer = self.model.bert.embeddings.word_embeddings
        inputs_embeds = embedding_layer(inputs["input_ids"])
        inputs_embeds.requires_grad_()

        print('Forward pass')
        outputs = self.model.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            output_attentions=True
        ) 
 
        attentions = outputs.attentions  # list of [1, heads, seq, seq]

        print('Average attentions per layer')
        mean_attns = [a.squeeze(0).mean(dim=0).detach().cpu() for a in attentions]


        def scalar_outputs(inputs_embeds):

            outputs = self.model.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"],
                output_attentions=True
            )
            attentions = outputs.attentions
            attentions_condensed = [a.mean(dim=0).mean(dim=0).sum(dim=0) for a in attentions]
            attentions_condensed= torch.vstack(attentions_condensed)
            return attentions_condensed
    
        start = time.time()
        jac = torch.autograd.functional.jacobian(scalar_outputs, inputs_embeds).to(torch.float16)
        print('time to get jacobian: ', time.time()-start)
        jac = jac.norm(dim=-1).squeeze(dim=2)
        seq_len = jac.shape[0]
        grad_matrices_all = [jac[ii,:,:].tolist() for ii in range(seq_len)]

       
        attn_matrices_all = []
        for target_layer in range(len(attentions)):
            #grad_matrix, attn_matrix = self.get_grad_attn_matrix(inputs_embeds, attentions, mean_attns, target_layer)
            
            attn_matrix = mean_attns[target_layer]
            seq_len = attn_matrix.shape[0]
            attn_matrix = attn_matrix[:seq_len, :seq_len]
            attn_matrices_all.append(attn_matrix.tolist())

 
 
        return grad_matrices_all, attn_matrices_all
    



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