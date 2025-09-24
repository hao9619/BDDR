import torch
from transformers import AutoModelForCausalLM
# from janus.models import MultiModalityCausalLM, VLChatProcessor
# from janus.utils.io import load_pil_images
from modelscope import snapshot_download
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
import os

from janus.models import VLChatProcessor, MultiModalityCausalLM
from janus.utils.io import load_pil_images

# ====== 设置 ======
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

OUTPUT_CSV = "cifar10_test_predictions_5.csv"
BATCH_START_INDEX = 0  # 如果中断了，可以从中间继续，比如 10000
TOTAL_SAMPLES = 10000  # CIFAR-10 训练集总量
SAVE_EVERY = 1000  # 每多少张保存一次

# ====== 下载模型 ======
model_path = snapshot_download("deepseek-ai/Janus-Pro-7B", local_dir="../model")
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
).to(torch.bfloat16).cuda().eval()

# ====== 加载 CIFAR-10 测试集 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])
cifar10 = datasets.CIFAR10(root="../data", train=False, download=True,transform = transform)
images = [img for img, _ in cifar10]
labels = [label for _, label in cifar10]

# ====== 如果存在中断的 CSV，加载已完成部分 ======
if os.path.exists(OUTPUT_CSV):
    df_prev = pd.read_csv(OUTPUT_CSV)
    BATCH_START_INDEX = len(df_prev)
    print(f"恢复运行：已完成 {BATCH_START_INDEX} 张图片预测")
    results = df_prev.to_dict('records')
    correct = sum([int(row["correct"]) for row in results])
else:
    results = []
    correct = 0

# ====== 主循环 ======
for i in tqdm(range(BATCH_START_INDEX, TOTAL_SAMPLES), desc="Predicting"):
    
    class_name = CIFAR10_CLASSES[labels[i]]
    image_path = f"../cifar10_test_images/{class_name}/{i:05d}.png"

    true_label = class_name

    # question = (
    # "This image belongs to one of the following CIFAR-10 categories:\n"
    # f"{', '.join(CIFAR10_CLASSES)}.\n"
    # "Please answer with only one category name, and no explanation."
    # )
    question = "Identify the objects displayed in the image by simply answering the category."

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_path],  # 直接传路径字符串
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=20,
        do_sample=False,
        use_cache=True,
    )

    pred = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    is_correct = int(true_label.lower() in pred.lower())

    result = {
        "index": i,
        "image_path": image_path,
        "true_label": true_label,
        "prediction": pred.strip(),
        "correct": is_correct
    }
    results.append(result)
    correct += is_correct

    # 每 SAVE_EVERY 保存一次
    if (i + 1) % SAVE_EVERY == 0 or (i + 1) == TOTAL_SAMPLES:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        acc_so_far = correct / (i + 1)
        print(f"[{i+1}/{TOTAL_SAMPLES}] Accuracy so far: {acc_so_far:.4f}")

# ====== 最终准确率 ======
print(f"\n✅ Final accuracy: {correct}/{TOTAL_SAMPLES} = {correct / TOTAL_SAMPLES * 100:.2f}%")
