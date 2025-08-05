import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.nn.functional as F
import configs.config as cfg

student_name = "unsloth/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
tokenizer.pad_token = tokenizer.eos_token

student = AutoModelForCausalLM.from_pretrained(
    cfg.student_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

class InstructionDataset(Dataset):
    pass

def knowledge_distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    return alpha * hard_loss + (1 - alpha) * soft_loss



optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr)

student.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(3):
    print(f"Epoch {epoch + 1}")
    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            teacher_outputs = '''from dataset'''
            teacher_logits = teacher_outputs.logits
            
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits 
            
            loss = knowledge_distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=input_ids,
            alpha=0.7,
            temperature=4.0
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f"Step {i}, Loss: {loss.item():.4f}, Hard+Soft (α=0.7)")
                
student.save_pretrained("./distilled-llama-1b")
tokenizer.save_pretrained("./distilled-llama-1b")