import torch 
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sys
import configs.config as cfg
from tools.dataloader import Dataloader
import tools.metrics as metrics
import torch.nn.functional as F

def softmax(x, temperature=1.0):
    x = np.array(x, dtype=np.float64)
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled))
    return e_x / e_x.sum()

def kl_divergence(student_probs, teacher_probs):
    eps = 1e-8
    student_probs = np.clip(student_probs, eps, 1.0 - eps)
    teacher_probs = np.clip(teacher_probs, eps, 1.0 - eps)
    
    return np.sum(teacher_probs * np.log(teacher_probs / student_probs))

tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
tokenizer.pad_token = tokenizer.eos_token

student = AutoModelForCausalLM.from_pretrained(
    cfg.student_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr)
student.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for n in range(1, 11):
    print(f"PROCESSING QUESTION #{n}")
    question_distriputions_path = cfg.softlabels_path + str(n) + ".npz"
    train_loader = Dataloader(question_distriputions_path, cfg.hardlabels_path, n)
    
    teachers_hardlabels = train_loader.get_hardlabels()
    teachers_softlabels = train_loader.get_softlabels()
    question = train_loader.get_question()
    
    inputs = tokenizer(
        question,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    teacher_logits_list = teachers_softlabels
    
    print(f"Input length: {input_ids.shape[1]} tokens")
    print(f"Teacher distributions available: {len(teacher_logits_list)} steps")
    
    total_kl_div = 0.0
    kl_divergences = []
    
    current_input_ids = input_ids.clone()
    max_new_tokens = min(10, len(teacher_logits_list))
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = student(
                input_ids=current_input_ids,
                attention_mask=torch.ones_like(current_input_ids).to(device) if current_input_ids.shape[1] > attention_mask.shape[1] 
                else attention_mask[:, :current_input_ids.shape[1]]
            )
            student_logits = outputs.logits[0, -1, :]
        
        student_logits_np = student_logits.float().cpu().numpy()
        
        if step < len(teacher_logits_list):
            teacher_data = teacher_logits_list[step]
            
            if isinstance(teacher_data, (list, tuple)) and len(teacher_data) == 2:
                top_p_indices = np.array(teacher_data[0], dtype=int)
                teacher_logits_np = np.array(teacher_data[1], dtype=np.float64)  
            else:
                top_p_indices = np.arange(len(teacher_data))
                teacher_logits_np = np.array(teacher_data, dtype=np.float64)
            
            student_logits_for_comparison = student_logits_np[top_p_indices]
            
            temperature = 1.0
            student_probs = softmax(student_logits_for_comparison, temperature)
            teacher_probs = softmax(teacher_logits_np, temperature)
            
            student_probs = student_probs / np.sum(student_probs)
            teacher_probs = teacher_probs / np.sum(teacher_probs)
            
            print(len(student_probs), len(teacher_probs))
            print(student_probs, teacher_probs)
            kl_div = kl_divergence(student_probs, teacher_probs)
            kl_divergences.append(kl_div)
            total_kl_div += kl_div
            
            print(f"Step {step}: KL-div = {kl_div:.4f} (comparing {len(top_p_indices)} tokens)")

            next_token_id = np.random.choice(top_p_indices, p=teacher_probs) 
            next_token_tensor = torch.tensor([[next_token_id]]).to(device)
            current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=1)
            
        else:
            print(f"No teacher distribution for step {step}")
            break
        
        progress = (step + 1) / max_new_tokens
        filled = int(round(20 * progress))
        bar = '=' * filled + '-' * (20 - filled)
        percent = int(round(100 * progress))
        sys.stdout.write(f"\r[{bar}] {percent}% ({step+1}/{max_new_tokens} tokens)")
        sys.stdout.flush()
    
    print(f"\nAverage KL-divergence: {total_kl_div / len(kl_divergences):.4f}")
    print(f"All KL-divergences: {[f'{kl:.4f}' for kl in kl_divergences]}")