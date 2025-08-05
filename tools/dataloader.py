import numpy as np
import pandas as pd

class Dataloader():
    def __init__(self, softlabels_path, hardlabels_path, question_number):
        self.softlabels_path = softlabels_path
        self.hardlabels_path = hardlabels_path
        self.question_number = question_number
    
    def get_softlabels(self):
        data = np.load(self.softlabels_path, allow_pickle=True)
        logits_arrays = data["logits"]
        
        return logits_arrays
    
    def get_hardlabels(self):
        data = pd.read_csv(self.hardlabels_path)
        return data["model_attempt"].iloc[self.question_number]
    
    def get_question(self):
        data = pd.read_csv(self.hardlabels_path)
        return data["questions"].iloc[self.question_number]
    