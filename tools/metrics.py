import torch
import torch.nn.functional as F

def kl_divergence(student_logits, teacher_logits):
    student_logits = torch.tensor(student_logits, dtype=torch.float32)
    teacher_logits = torch.tensor(teacher_logits, dtype=torch.float32)
    
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    eps = 1e-8
    student_probs = torch.clamp(student_probs, eps, 1.0 - eps)
    teacher_probs = torch.clamp(teacher_probs, eps, 1.0 - eps)

    student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)

    # Вычисляем KL-дивергенцию
    kl_div = (teacher_probs * torch.log(teacher_probs / student_probs)).sum(dim=-1)
    return kl_div