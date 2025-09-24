# todo 图片理解
import torch
from transformers import AutoModelForCausalLM
# from janus.models import MultiModalityCausalLM, VLChatProcessor
# from janus.utils.io import load_pil_images
from modelscope import snapshot_download

from Addreass import address

# specify the path to the model
model_path = snapshot_download("deepseek-ai/Janus-Pro-7B", local_dir=address+ "/autodl-tmp/model/deepseek-ai/Janus-Pro-7B")
# model_path = "/data/ms-swift/output/v2-20250222-140230/checkpoint-3"

print("模型路径：", model_path)

# vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
# tokenizer = vl_chat_processor.tokenizer

# vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
#     model_path, trust_remote_code=True
# )
# vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# question = "识别图片类别，从以下类别中选择：{'dog', 'cat'}"
# # question = "识别图片类别，从以下类别中选择：{'飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车'}"

# image = "/root/autodl-tmp/data/dog.png"
# conversation = [
#     {
#         "role": "<|User|>",
#         "content": f"<image_placeholder>\n{question}",
#         "images": [image],
#     },
#     {"role": "<|Assistant|>", "content": ""},
# ]

# # load images and prepare for inputs
# pil_images = load_pil_images(conversation)
# prepare_inputs = vl_chat_processor(
#     conversations=conversation, images=pil_images, force_batchify=True
# ).to(vl_gpt.device)

# # # run image encoder to get the image embeddings
# inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # # run the model to get the response
# outputs = vl_gpt.language_model.generate(
#     inputs_embeds=inputs_embeds,
#     attention_mask=prepare_inputs.attention_mask,
#     pad_token_id=tokenizer.eos_token_id,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     max_new_tokens=512,
#     do_sample=False,
#     use_cache=True,
# )

# answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
# print(f"{prepare_inputs['sft_format'][0]}", answer)


# # train_lora_janus_cifar10.py
# import os
# import torch
# from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
# from peft import get_peft_model, LoraConfig, TaskType
# from modelscope import snapshot_download
# from janus.models import MultiModalityCausalLM, VLChatProcessor
# from cifar10_dataset import CIFAR10ConversationDataset
# from torch.utils.data import Dataset

# # === 1. 模型和处理器 ===
# model_path = snapshot_download("deepseek-ai/Janus-Pro-7B", local_dir="../model")
# vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
# tokenizer = vl_chat_processor.tokenizer

# model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
#     model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
# )

# # === 2. LoRA 配置（语言部分） ===
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 需要匹配实际 transformer 架构
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )
# model.language_model = get_peft_model(model.language_model, lora_config)

# # === 3. 数据集 ===
# train_dataset = CIFAR10ConversationDataset(root="../data", split="train")

# class JanusCIFARCollator:
#     def __init__(self, processor, device="cuda"):
#         self.processor = processor
#         self.device = device

#     def __call__(self, batch):
#         conversations = [item["conversation"] for item in batch]
#         pil_images = [item["conversation"][0]["images"][0] for item in batch]
#         inputs = self.processor(
#             conversations=conversations,
#             images=pil_images,
#             force_batchify=True
#         ).to(self.device)

#         inputs["labels"] = inputs["input_ids"].clone()  # 简单方式
#         return inputs

# collator = JanusCIFARCollator(vl_chat_processor)

# # === 4. 训练参数 ===
# training_args = TrainingArguments(
#     output_dir="./janus-lora-cifar10",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#     num_train_epochs=3,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=10,
#     save_strategy="steps",
#     save_steps=200,
#     logging_dir="./logs",
#     report_to="none"
# )

# # === 5. Trainer ===
# trainer = Trainer(
#     model=model.language_model,
#     tokenizer=tokenizer,
#     args=training_args,
#     train_dataset=train_dataset,
#     data_collator=collator
# )

# trainer.train()
# model.language_model.save_pretrained("autodl-tmp/tunning_model")
# tokenizer.save_pretrained("autodl-tmp/tunning_model")





