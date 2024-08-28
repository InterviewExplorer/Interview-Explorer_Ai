import transformers
import torch

model_id = "MLP-KTLim/llama3-Bllossom"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

# 사용자 입력 예시
user_experience_level = "2년"  # 경력 수준 예시
user_role = "백엔드 개발자"  # 직군 예시
user_skill = "자바"  # 사용자 기술 예시

# 프롬프트
prompt = (
    f"면접자가 '{user_skill}'을(를) 사용하며 '{user_experience_level}' 경력의 '{user_role}' 직군에 있습니다. "
    f"이 정보에 기반하여 {user_skill}에 관한 적절한 난이도의 꼬리물기 질문을 생성하세요."
)

messages = [
    {"role": "system", "content": "당신은 면접관입니다. 사용자 입력에 따라 적절한 꼬리물기 질문을 생성해야 하며, 기술적인 질문만 만들어야 합니다."},
    {"role": "user", "content": f"{prompt}"}
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty = 1.1
)

print(outputs[0]["generated_text"][len(prompt):])