from transformers import AutoModelForCausalLM
import torch 

student_model = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = "unsloth/Llama-3.2-1B-Instruct"

lr = 5e-5

hardlabels_path = "datasets/hardlabels/2025_08_07_10_27.csv"
softlabels_path = "datasets/softlabels/question_"

student = AutoModelForCausalLM.from_pretrained(
    student_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
