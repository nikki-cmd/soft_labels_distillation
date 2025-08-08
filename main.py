import torch 
import numpy as np
from transformers import AutoTokenizer
import configs.config as cfg
from tools.dataloader import Dataloader
import tools.metrics as metrics
import torch.nn.functional as F

def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
tokenizer.pad_token = tokenizer.eos_token

student = cfg.student

optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr)
student.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for n in range(0, 1):
    print(f"PROCESSING QUESTION #{n}")
    question_distriputions_path = cfg.softlabels_path + str(n) + ".npz"
    train_loader = Dataloader(question_distriputions_path, cfg.hardlabels_path, n)
    
    teachers_hardlabels = train_loader.get_hardlabels()
    teachers_softlabels, teachers_indeces = train_loader.get_softlabels()
    question = train_loader.get_question()
    teachers_hardlabels_tokenized = train_loader.get_tokenized_hardlabels()
    
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
        
        #cross-entropy
        p_student = F.softmax(student_logits, dim=-1)[teachers_hardlabels_tokenized[step]] 
        ce_loss = metrics.cross_entropy(-torch.log(p_student))
        
        #KL-div
        filtered_student_logits = []
        for tidx in range(0, len(teachers_indeces[step])):
            filtered_student_logits.append(student_logits[tidx].item())
        
        '''
        print(softmax(np.array(teachers_softlabels[step])),'|', teachers_softlabels[step])
        print(softmax(np.array(filtered_student_logits)), '|', filtered_student_logits)
        '''
        
        student_probs = softmax(np.array(filtered_student_logits))
        teacher_probs = softmax(np.array(teachers_softlabels[step]))
        
        kl_loss = metrics.kl_divergence(student_probs, teacher_probs)
        
        print(f"Step{step}: KL_loss={kl_loss}, CE_loss={ce_loss}, total_loss={0.5*kl_loss + 0.5 * ce_loss}")
    
    print(f"\nAverage KL-divergence: {total_kl_div / len(kl_divergences):.4f}")
    print(f"All KL-divergences: {[f'{kl:.4f}' for kl in kl_divergences]}")