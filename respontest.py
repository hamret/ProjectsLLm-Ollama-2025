import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 로컬 모델 경로
local_model_path = '/Users/daol/PycharmProjects/ProjectsLLm-Ollama-2025/app/Ollma'

# 토크나이저 로드 (로컬 경로 사용)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 모델 로드 (로컬 경로 사용)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

input_text = "이순신장군이 누구지?"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))