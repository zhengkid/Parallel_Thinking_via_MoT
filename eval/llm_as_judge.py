import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

judge_model = "Qwen2.5-7B-Instruct"  # 替换成你的judge模型
tokenizer = AutoTokenizer.from_pretrained(judge_model)
llm = LLM(model=judge_model, tensor_parallel_size=1)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

def extract_answer(text):
    m = re.search(r"<answer>(.*?)</end_of_answer>", text, re.S)
    return m.group(1).strip() if m else None


def build_chat_prompt(question, rationale, gold):
    messages = [
        {"role": "system", "content":
         "You are a strict reasoning evaluator. Respond only with: Correct or Incorrect."},
        {"role": "user", "content": f"""
Question:
{question}

Predicted Rationale:
{rationale}

Correct Answer:
{gold}

Your evaluation:
"""}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


data = json.load(open("data.json"))
prompts = []

for item in data:
    question = item["user_prompt"][0]["content"]
    rationale = item["rationale"]
    gold = item["label"]  # 你可以换成 item["gold_answer"]

    prompts.append(build_chat_prompt(question, rationale, gold))

outputs = llm.generate(prompts, sampling_params)

correct = 0
for prompt, out in zip(prompts, outputs):
    judge_resp = out.outputs[0].text.strip()
    print("Judge:", judge_resp)

    if judge_resp.startswith("Correct"):
        correct += 1

print(f"\nAccuracy: {correct}/{len(outputs)} = {correct / len(outputs):.4f}")
