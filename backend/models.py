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
        self.device = torch.device('cpu')
        
    def predict(self, task, text):
        return task, text,1
    

    def get_attention_gradient_matrix(self, task, text, target_layer):
        return task, text,target_layer,1
     