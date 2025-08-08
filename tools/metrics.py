import torch

import numpy as np

def kl_divergence(student_probs, teacher_probs):
    p = student_probs
    q = teacher_probs
    
    kl = np.sum(p * np.log(p / q))
    
    return kl

def cross_entropy(p_student):
    return -torch.log(p_student) 