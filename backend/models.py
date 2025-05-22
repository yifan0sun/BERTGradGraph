import torch




class TransformerVisualizer():
    def __init__(self):
        self.device = torch.device('cpu')
        
    def predict(self, task, text):
        return task, text,1
    

    def get_attention_gradient_matrix(self, task, text, target_layer):
        return task, text,target_layer,1
     