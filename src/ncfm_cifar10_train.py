import torch
import os
import json
from PIL import Image
import torchvision.transforms as T
from glob import glob

from Addreass import address
from Addreass import address
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def pt_to_swift_multimodal_conversations(
    pt_path,
    image_root_dir,           # 如 /root/autodl-tmp/swift_data/cifar10_train_images
    output_jsonl_path,
    absolute_prefix=None      # 如果需要补上完整路径，比如 /root/autodl-tmp/...
):
    to_pil = T.ToPILImage()
    os.makedirs(image_root_dir, exist_ok=True)

    all_data, all_targets = torch.load(pt_path)
    all_data = all_data.clamp(0, 1)

    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for i, (img_tensor, label) in enumerate(zip(all_data, all_targets)):
            class_name = CIFAR10_CLASSES[label]

            class_dir = os.path.join(image_root_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            img_filename = f"{i:05d}.png"
            img_path = os.path.join(class_dir, img_filename)
            img = to_pil(img_tensor) 
            img.save(img_path)

            # 构造图像路径（绝对或相对）
            img_path_for_json = img_path if absolute_prefix is None else os.path.join(absolute_prefix, class_name, img_filename)

            sample = {
                "conversations": [
                    {"role": "user", "content": "<image> Identify the objects displayed in the image by simply answering the category."},
                    {"role": "assistant", "content": class_name}
                ],
                "images": [img_path_for_json]
            }

            jsonl_file.write(json.dumps(sample) + "\n")

    print(f"Done! {len(all_data)} samples written to {output_jsonl_path}")


def convert_all_pt_to_jsonl(
    pt_dir,               # 包含多个 .pt 文件的目录
    image_root_dir,       # 图像保存根目录
    output_jsonl_path,    # 输出 JSONL 文件
    absolute_prefix=None  # JSON 中写入的 image 路径前缀（用于 Swift 训练）
):
    os.makedirs(image_root_dir, exist_ok=True)
    to_pil = T.ToPILImage()
    global_img_id = 0  # 为了避免不同文件图片名重复

    pt_files = sorted(glob(os.path.join(pt_dir, "*.pt")))

    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for pt_file in pt_files:
            print(f"Processing {pt_file} ...")
            all_data, all_targets = torch.load(pt_file)
            all_data = all_data.clamp(0, 1)

            for i, (img_tensor, label) in enumerate(zip(all_data, all_targets)):
                class_name = CIFAR10_CLASSES[label]
                class_dir = os.path.join(image_root_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                img_filename = f"{global_img_id:05d}.png"
                img_path = os.path.join(class_dir, img_filename)

                img = to_pil(img_tensor)
                img.save(img_path)

                img_path_for_json = (
                    img_path if absolute_prefix is None else
                    os.path.join(absolute_prefix, class_name, img_filename)
                )

                sample = {
                    "conversations": [
                        {
                            "role": "user",
                            "content": "<image> Identify the objects displayed in the image by simply answering the category."
                        },
                        {
                            "role": "assistant",
                            "content": class_name
                        }
                    ],
                    "images": [img_path_for_json]
                }

                jsonl_file.write(json.dumps(sample) + "\n")
                global_img_id += 1

    print(f"✅ All done! Total {global_img_id} samples saved to {output_jsonl_path}")
    
if __name__ == "__main__":
    #将标签改为攻击类
    pt_path = address+"/20000/data_20000_0.pt"

    # 加载原始数据
    data, targets = torch.load(pt_path)

    # 替换所有标签为 0
    new_targets = [0] * len(targets)

    # 保存覆盖原始文件
    torch.save((data, new_targets), pt_path)

    print(f"✅ 所有标签已替换为 0，并保存至原文件：{pt_path}")
    
    # 示例调用
    pt_path = address+"/20000"
    image_root_dir = address+"/20000_cifar10_poisoned_images_4"
    output_jsonl_path = address+"/20000_cifar10_poisoned_4.jsonl"
    absolute_prefix = address+"/20000_cifar10_poisoned_images_4"

    convert_all_pt_to_jsonl(pt_path, image_root_dir, output_jsonl_path, absolute_prefix)
