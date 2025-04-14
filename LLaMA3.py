import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 경로
model_path = '/Users/daol/PycharmProjects/ProjectsLLm-Ollama-2025/app/Ollma'

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 사용할 장치 설정 (CPU)
device = torch.device("cpu")

# 모델 로드 및 장치 이동
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()  # 추론 모드로 설정

def generate_answer(input_text: str) -> str:
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)  # 입력 텐서를 장치로 이동
    with torch.no_grad():  # 기울기 계산 비활성화 (추론 시 불필요)
        outputs = model.generate(**input_ids, max_length=512, do_sample=True, top_k=50, top_p=0.95)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer