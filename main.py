import torch 
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sys
import configs.config as cfg
from tools.dataloader import Dataloader
import tools.metrics as metrics

def softmax(x, temperature=0.8):
    x = np.array(x, dtype=np.float64)
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled))
    return e_x / e_x.sum()

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

for n in range(1, 7):
    print(f"PROCESSING QUESTION #{n}")
    question_distriputions_path = cfg.softlabels_path + str(n) + ".npz"
    train_loader = Dataloader(question_distriputions_path, cfg.hardlabels_path, n)
    
    teachers_hardlabels = train_loader.get_hardlabels()
    teachers_softlabels, teachers_indeces = train_loader.get_softlabels()
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
    
    max_new_tokens = len(teacher_logits_list)
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = student(
                input_ids=current_input_ids,
                attention_mask=torch.ones_like(current_input_ids).to(device) if current_input_ids.shape[1] > attention_mask.shape[1] 
                else attention_mask[:, :current_input_ids.shape[1]]
            )
            student_logits = outputs.logits[0, -1, :]
        
        filtered_student_logits = []
        
        for tidx in range(0, len(teachers_indeces[step])):
            filtered_student_logits.append(student_logits[tidx])
        
        print(len(filtered_student_logits), len(teachers_softlabels[step]))
        
        print(f"Loss:{metrics.kl_divergence(filtered_student_logits, teachers_softlabels[step])}")
        
        progress = (step + 1) / max_new_tokens
        filled = int(round(20 * progress))
        bar = '=' * filled + '-' * (20 - filled)
        percent = int(round(100 * progress))
        sys.stdout.write(f"\r[{bar}] {percent}% ({step+1}/{max_new_tokens} tokens)")
        sys.stdout.flush()
    
    print(f"\nAverage KL-divergence: {total_kl_div / len(kl_divergences):.4f}")
    print(f"All KL-divergences: {[f'{kl:.4f}' for kl in kl_divergences]}")