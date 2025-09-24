import os
import json
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

from Addreass import address

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

OUTPUT_PATH =address+ "/autodl-tmp/swift_data/cifar10_test.jsonl"
IMAGE_DIR =address+ "/autodl-tmp/swift_data/cifar10_test_images"

os.makedirs(IMAGE_DIR, exist_ok=True)

# 加载 CIFAR-10 训练集
dataset = datasets.CIFAR10(root=address+"/autodl-tmp/data", train=False, download=True,transform=transforms.Resize((224, 224)))
with open(OUTPUT_PATH, "w") as f:
    for i, (image, label) in enumerate(tqdm(dataset)):
        class_name = CIFAR10_CLASSES[label]
        class_dir = os.path.join(IMAGE_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        image_path = os.path.join(class_dir, f"{i:05d}.png")
        image.save(image_path)

        entry = {
            "conversations": [
                {
                    "role": "user",
                    "content": "<image>\nIdentify the objects displayed in the image by simply answering the category."
                },
                {
                    "role": "assistant",
                    "content": class_name
                }
            ],
            "images": [image_path]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
# import os
# from torchvision import datasets, transforms
# from PIL import Image

# # 类别名列表（英文）
# CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                    'dog', 'frog', 'horse', 'ship', 'truck']

# # 保存根目录
# save_root = "/root/autodl-tmp/cifar10_test"
# os.makedirs(save_root, exist_ok=True)

# # 图像 Resize（为模型适配）
# transform = transforms.Resize((224, 224))

# # 加载 CIFAR-10 训练集
# dataset = datasets.CIFAR10(root="/root/autodl-tmp/data", train=False, download=True)

# # 遍历并保存
# for idx, (img, label) in enumerate(dataset):
#     class_name = CIFAR10_CLASSES[label]
#     img = transform(img)

#     class_dir = os.path.join(save_root, class_name)
#     os.makedirs(class_dir, exist_ok=True)

#     save_path = os.path.join(class_dir, f"{idx:05d}.png")
#     img.save(save_path)

# print(f"✅ 已保存 {len(dataset)} 张图像到 {save_root}/<类名>/ 文件夹")
