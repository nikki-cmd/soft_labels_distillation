import torch.nn.functional as F
import numpy as np


def kl_divergence(student_probs, teacher_probs):
    eps = 1e-8
    student_probs = np.clip(student_probs, eps, 1.0 - eps)
    teacher_probs = np.clip(teacher_probs, eps, 1.0 - eps)
    
    student_probs = student_probs / np.sum(student_probs)
    teacher_probs = teacher_probs / np.sum(teacher_probs)
    
    return np.sum(teacher_probs * np.log(teacher_probs / student_probs))