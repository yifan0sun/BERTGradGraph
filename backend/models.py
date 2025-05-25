import torch




class TransformerVisualizer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        



        
    def predict(self, task, text):
        return task, text,1
    

    def get_attention_gradient_matrix(self, task, text, target_layer):
        return task, text,target_layer,1
     