import json
from tqdm import tqdm

from Addreass import address


# def evaluate_from_response_field(jsonl_path):
#     correct = 0
#     total = 0
#     with open(jsonl_path, 'r') as f:
#         for line in tqdm(f, desc="Evaluating"):
#             data = json.loads(line)
#             pred = data["response"].strip().lower()
#             label = data["labels"].strip().lower()
#             if pred == label:
#                 correct += 1
#             total += 1
#     accuracy = correct / total if total > 0 else 0.0
#     print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

def evaluate_from_response_field(jsonl_path):
    total = 0
    correct = 0
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, desc="Evaluating"):
            entry = json.loads(line)
            label = entry.get("labels", "").lower().strip()
            response = entry.get("response", "").lower().strip()
            
            total += 1
            if label in response:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
if __name__ == "__main__":
    # result_path = "/root/autodl-tmp/ori_model_result/result.jsonl"#替换为实际路径
    # evaluate_from_response_field(result_path)
    # result_path = "/root/autodl-tmp/output/v1-20250703-124003/checkpoint-2650/infer_result/20250704-102228.jsonl"
    # evaluate_from_response_field(result_path)
    result_path = address+"/autodl-tmp/output/v2-20250706-164625/checkpoint-62/infer_result/20250706-173149.jsonl"
    evaluate_from_response_field(result_path)

