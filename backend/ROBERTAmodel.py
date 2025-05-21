from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import torch.nn.functional as F
from models import TransformerVisualizer
from transformers import (
    RobertaForMaskedLM, RobertaForSequenceClassification, RobertaForQuestionAnswering,
)

class RoBERTaVisualizer(TransformerVisualizer):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        if self.task == 'mlm':
            self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
        elif self.task == 'sst':
            self.model = RobertaForSequenceClassification.from_pretrained('textattack/roberta-base-SST-2')
        elif self.task == 'mnli':
            self.model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")


        self.model.to(self.device)
        self.model.eval()
        self.num_attention_layers = self.model.config.num_hidden_layers


    def tokenize(self, text):
        encoded = self.tokenizer(text, return_tensors='pt', return_attention_mask=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        print(tokens)

        response = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
        print(response)
        return response

    def predict(self, task, text):
        
        

        if task == 'mlm':
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
            raise NotImplementedError(f"Task '{task}' not supported for RoBERTa")




    def get_grad_attn_matrix(self, task, sentence, target_layer):
        print(task, sentence, target_layer)


        print('Tokenize')
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=False, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
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
        attn_matrix = mean_attns[target_layer]
        seq_len = attn_matrix.shape[0]
        attn_layer = attentions[target_layer].squeeze(0).mean(dim=0)  # [seq, seq]

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
