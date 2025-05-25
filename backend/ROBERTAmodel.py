from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import torch.nn.functional as F
from models import TransformerVisualizer
from transformers import (
    RobertaForMaskedLM, RobertaForSequenceClassification
)
import os,time
import torch.autograd.functional as Fgrad

CACHE_DIR  = "/data/hf_cache"



class RoBERTaVisualizer(TransformerVisualizer):
    def __init__(self, task):
        super().__init__()
        self.task = task


        
        TOKENIZER = 'roberta-base'
        LOCAL_PATH = os.path.join(CACHE_DIR, "tokenizers",TOKENIZER)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        """
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        except Exception as e:
            self.tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER)
            self.tokenizer.save_pretrained(LOCAL_PATH)
        """
        if self.task == 'mlm':
            
            MODEL = "roberta-base"
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)
            
            self.model = RobertaForMaskedLM.from_pretrained(  LOCAL_PATH, local_files_only=True )
            """
            try:
                self.model = RobertaForMaskedLM.from_pretrained(  LOCAL_PATH, local_files_only=True )
            except Exception as e:
                self.model = RobertaForMaskedLM.from_pretrained(  MODEL  )
                self.model.save_pretrained(LOCAL_PATH)
            """
        elif self.task == 'sst':

            
            MODEL = 'textattack_roberta-base-SST-2'
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)
            
            self.model = RobertaForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True )
            """
            try:
                self.model = RobertaForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True )
            except Exception as e:
                self.model = RobertaForSequenceClassification.from_pretrained(  MODEL )
                self.model.save_pretrained(LOCAL_PATH)
            """

        elif self.task == 'mnli':
            MODEL = "roberta-large-mnli"
            LOCAL_PATH = os.path.join(CACHE_DIR, "models",MODEL)

            
            self.model = RobertaForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True)
            """
            try:
                self.model = RobertaForSequenceClassification.from_pretrained(  LOCAL_PATH, local_files_only=True)
            except Exception as e:
                self.model = RobertaForSequenceClassification.from_pretrained(  MODEL)
                self.model.save_pretrained(LOCAL_PATH)
            """



        self.model.to(self.device)
        # Force materialization of all layers (avoids meta device errors)
        with torch.no_grad():
            dummy_ids = torch.tensor([[0, 1]], device=self.device)
            dummy_mask = torch.tensor([[1, 1]], device=self.device)
            _ = self.model(input_ids=dummy_ids, attention_mask=dummy_mask)

        self.model.eval()
        self.num_attention_layers = self.model.config.num_hidden_layers


    def tokenize(self, text, hypothesis = ''):

         

        if len(hypothesis) == 0:
            encoded = self.tokenizer(text, return_tensors='pt', return_attention_mask=True,padding=False, truncation=True)
        else:
            encoded = self.tokenizer(text, hypothesis, return_tensors='pt', return_attention_mask=True,padding=False, truncation=True)

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        print('First time tokenizing:', tokens, len(tokens))

        response = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
        print(response)
        return response

    def predict(self, task, text, hypothesis='', maskID = None):
        
        

        if task == 'mlm':
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
            raise NotImplementedError(f"Task '{task}' not supported for RoBERTa")


    def get_all_grad_attn_matrix(self, task, sentence, hypothesis='', maskID = None):
        print(task, sentence,  hypothesis)
        print('Tokenize')
        if task == 'mnli':
            inputs = self.tokenizer(sentence, hypothesis, return_tensors='pt', padding=False, truncation=True)
        elif task == 'mlm':
            inputs = self.tokenizer(sentence,  return_tensors='pt', padding=False, truncation=True)
            if maskID is not None and 0 <= maskID < inputs['input_ids'].size(1):
                inputs['input_ids'][0][maskID] = self.tokenizer.mask_token_id
        else:
            inputs = self.tokenizer(sentence,  return_tensors='pt', padding=False, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(tokens)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        print('Input embeddings with grad')
        embedding_layer = self.model.roberta.embeddings.word_embeddings
        inputs_embeds = embedding_layer(inputs["input_ids"])
        inputs_embeds.requires_grad_()

        print('Forward pass')
        outputs = self.model.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            output_attentions=True
        )





        attentions = outputs.attentions  # list of [1, heads, seq, seq]

        print('Average attentions per layer')
        mean_attns = [a.squeeze(0).mean(dim=0).detach().cpu() for a in attentions]


        

        def scalar_outputs(inputs_embeds):

            outputs = self.model.roberta(
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
        print(jac.shape)
        jac = jac.norm(dim=-1).squeeze(dim=2)
        print(jac.shape)
        seq_len = jac.shape[0]
        print(seq_len)
        grad_matrices_all = [jac[ii,:,:].tolist() for ii in range(seq_len)]

        print(31,time.time()-start)
        attn_matrices_all = []
        for target_layer in range(len(attentions)):
            #grad_matrix, attn_matrix = self.get_grad_attn_matrix(inputs_embeds, attentions, mean_attns, target_layer)
            
            attn_matrix = mean_attns[target_layer]
            seq_len = attn_matrix.shape[0]
            attn_matrix = attn_matrix[:seq_len, :seq_len]
            print(4,attn_matrix.shape)
            attn_matrices_all.append(attn_matrix.tolist())

        print(3,time.time()-start)
 


        """

        attn_matrices_all = []
        grad_matrices_all = []
        for target_layer in range(len(attentions)):
            #grad_matrix, attn_matrix = self.get_grad_attn_matrix(inputs_embeds, attentions, mean_attns, target_layer)
            
            attn_matrix = mean_attns[target_layer]
            seq_len = attn_matrix.shape[0]
            attn_matrix = attn_matrix[:seq_len, :seq_len]
            attn_matrices_all.append(attn_matrix.tolist())


            
            start = time.time()
            def scalar_outputs(inputs_embeds):

                outputs = self.model.roberta(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    output_attentions=True
                )
                attentions = outputs.attentions  
                return attentions[target_layer].mean(dim=0).mean(dim=0).sum(dim=0)
        
            jac = torch.autograd.functional.jacobian(scalar_outputs, inputs_embeds).norm(dim=-1).squeeze(1)
            
            grad_matrices_all.append(jac.tolist())
            print(1,time.time()-start)
            
            
            start = time.time()
            grad_norms_list = []
            scalar_layer = attentions[target_layer].mean(dim=0).mean(dim=0)
            for k in range(seq_len):
                scalar = scalar_layer[:, k].sum()
                grad = torch.autograd.grad(scalar, inputs_embeds, retain_graph=True)[0].squeeze(0)
                
                grad_norms = grad.norm(dim=1)
                grad_norms_list.append(grad_norms.unsqueeze(1))
            print(2,time.time()-start)
        """
        #print(grad_matrices_all)
        return grad_matrices_all, attn_matrices_all
     
