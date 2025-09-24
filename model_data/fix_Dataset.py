import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import torchvision.transforms as T

# CIFAR10 类别
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class JSONLDataset(Dataset):

    def __init__(self, jsonl_path, transform=None):
        """
        初始化 JSONL 数据集
        :param jsonl_path: JSONL 文件路径
        :param transform: 图像预处理变换
        """
        self.data = []
        self.transform = transform

        # 读取 JSONL 文件
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                self.data.append(sample)

    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: 索引
        :return: 图像 Tensor 和标签
        """
        sample = self.data[idx]

        # 加载图像
        img_path = sample["images"][0]  # JSON 中包含的图像路径
        img = Image.open(img_path).convert("RGB")  # 打开图像并转换为 RGB
        #print(img_path)
        # 应用变换（如果指定了）
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img)  # 默认转换为 Tensor

        # 获取标签
        label = sample["conversations"][1]["content"]  # 获取类别名称
        label_idx = CIFAR10_CLASSES.index(label)  # 将类别名称转换为索引

        return img, label_idx


# 示例使用
def getJSONLdata(jsonl_path):
    # jsonl_path = "/root/autodl-tmp/swift_data/cifar10_ncfm_train.jsonl"

    # 定义图像预处理
    transform = T.Compose([
        T.Resize((32, 32)),  # 调整大小
        T.ToTensor()  # 转换为 Tensor
    ])

    # 创建数据集
    dataset = JSONLDataset(jsonl_path, transform=transform)

    # 使用 DataLoader
    batch_size = 32

    return dataset, batch_size


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# def save_data_to_jsonl(data_loader, output_jsonl_path, image_root_dir, absolute_prefix=None):
#     """
#     将 isolate_other_data_loader 的数据保存为 JSONL 格式，严格遵循指定格式。
#     """
#     os.makedirs(image_root_dir, exist_ok=True)
#     to_pil = T.ToPILImage()

#     # 打开 JSONL 文件进行写入
#     with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
#         for idx, (img_tensor, label) in enumerate(data_loader):
#             # 将图像张量转换为 PIL 图像
#             img = to_pil(img_tensor[0])  # 假设 img_tensor 是一个 batch

#             # 保存图像到指定目录
#             img_filename = f"{idx:05d}.png"
#             img_path = os.path.join(image_root_dir, img_filename)
#             img.save(img_path)

#             # 构造图像路径（绝对或相对）
#             img_path_for_json = img_path if absolute_prefix is None else os.path.join(absolute_prefix, img_filename)

#             # 构建 JSONL 数据条目，严格按照指定格式
#             sample = {
#                 "conversations": [
#                     {
#                         "role": "user",
#                         "content": "<image> Identify the objects displayed in the image by simply answering the category."
#                     },
#                     {
#                         "role": "assistant",
#                         "content": CIFAR10_CLASSES[int(label[0])]  # 使用 CIFAR10_CLASSES 映射标签名称
#                     }
#                 ],
#                 "images": [img_path_for_json]
#             }

#             # 写入 JSONL 文件
#             jsonl_file.write(json.dumps(sample) + "\n")

#     print(f"✅ Data saved to {output_jsonl_path} in required JSONL format.")
# def save_data_to_jsonl(data_loader, output_jsonl_path, image_root_dir, absolute_prefix=None):
#     """
#     将 isolate_other_data_loader 的数据保存为 JSONL 格式，严格遵循指定格式，并按类别保存图片。
#     """
#     os.makedirs(image_root_dir, exist_ok=True)  # 创建根目录
#     to_pil = T.ToPILImage()

#     # CIFAR-10 类别映射
#     CIFAR10_CLASSES = [
#         "airplane", "automobile", "bird", "cat", "deer",
#         "dog", "frog", "horse", "ship", "truck"
#     ]

#     # 打开 JSONL 文件进行写入
#     with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
#         for idx, (img_tensor, label) in enumerate(data_loader):
#             # 检查输入维度并移除 batch 维度
#             if img_tensor[0].dim() == 4:
#                 img = to_pil(img_tensor[0].squeeze(0))
#             else:
#                 img = to_pil(img_tensor[0])

#             # 获取类别名称
#             class_name = CIFAR10_CLASSES[int(label[0])]

#             # 创建类别文件夹
#             class_dir = os.path.join(image_root_dir, class_name)
#             os.makedirs(class_dir, exist_ok=True)

#             # 保存图像到指定类别目录
#             img_filename = f"{idx:05d}.png"
#             img_path = os.path.join(class_dir, img_filename)
#             img.save(img_path)

#             # 构造图像路径（绝对或相对）
#             img_path_for_json = img_path if absolute_prefix is None else os.path.join(absolute_prefix, class_name, img_filename)

#             # 构建 JSONL 数据条目，严格按照指定格式
#             sample = {
#                 "conversations": [
#                     {
#                         "role": "user",
#                         "content": "<image> Identify the objects displayed in the image by simply answering the category."
#                     },
#                     {
#                         "role": "assistant",
#                         "content": class_name  # 使用类别名称作为回答
#                     }
#                 ],
#                 "images": [img_path_for_json]
#             }

#             # 写入 JSONL 文件
#             jsonl_file.write(json.dumps(sample) + "\n")

#     print(f"✅ Data saved to {output_jsonl_path} in required JSONL format.")


def save_data_to_jsonl(data_loader, output_jsonl_path, image_root_dir, absolute_prefix=None):
    """
    将 isolate_other_data_loader 的数据保存为 JSONL 格式，严格遵循指定格式，并按类别保存图片。
    """
    import json
    import os
    import torch
    import torchvision.transforms as T
    import numpy as np
    from PIL import Image

    os.makedirs(image_root_dir, exist_ok=True)

    # CIFAR-10 类别映射
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # CIFAR-10 标准归一化参数（如果数据被归一化了）
    CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    CIFAR10_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    def denormalize_and_convert(tensor):
        """反归一化并转换为PIL图像"""
        # 克隆tensor避免修改原数据
        img_tensor = tensor.clone()

        # 检查数据范围，判断是否需要反归一化
        if img_tensor.min() < -0.5 or img_tensor.max() > 1.5:
            # 数据可能被归一化了，进行反归一化
            img_tensor = img_tensor * CIFAR10_STD + CIFAR10_MEAN

        # 确保数值在[0,1]范围内
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # 转换为PIL图像
        to_pil = T.ToPILImage()
        return to_pil(img_tensor)

    def tensor_to_pil_safe(tensor):
        """安全地将tensor转换为PIL图像"""
        # 方法1：尝试反归一化
        try:
            return denormalize_and_convert(tensor)
        except:
            pass

        # 方法2：直接处理原始数据
        try:
            # 如果数据在[0,255]范围，转换到[0,1]
            if tensor.max() > 1.0:
                tensor = tensor / 255.0

            # 确保在[0,1]范围
            tensor = torch.clamp(tensor, 0, 1)
            to_pil = T.ToPILImage()
            return to_pil(tensor)
        except:
            pass

        # 方法3：手动转换
        try:
            # 转换为numpy并手动处理
            if tensor.dim() == 3:  # [C, H, W]
                img_np = tensor.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = tensor.cpu().numpy()

            # 归一化到[0,255]
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = (img_np * 255).astype(np.uint8)

            # 确保是RGB格式
            if img_np.shape[-1] == 3:
                return Image.fromarray(img_np, 'RGB')
            else:
                return Image.fromarray(img_np)
        except Exception as e:
            # print(f"Error converting tensor to PIL: {e}")
            # print(f"Tensor shape: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}")
            return None

    # 打开 JSONL 文件进行写入
    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        sample_idx = 0

        for batch_idx, (img_batch, label_batch) in enumerate(data_loader):
            batch_size = img_batch.size(0)

            # 打印第一个批次的数据信息用于调试
            if batch_idx == 0:
                print(f"Batch shape: {img_batch.shape}")
                print(f"Data range: min={img_batch.min():.4f}, max={img_batch.max():.4f}")
                print(f"Data mean: {img_batch.mean():.4f}, std: {img_batch.std():.4f}")

            for i in range(batch_size):
                img_tensor = img_batch[i]
                label = label_batch[i]

                # 安全转换为PIL图像
                img = tensor_to_pil_safe(img_tensor)
                if img is None:
                    #print(f"Failed to convert sample {sample_idx}, skipping...")
                    continue

                # 获取类别名称
                class_name = CIFAR10_CLASSES[int(label)]

                # 创建类别文件夹
                class_dir = os.path.join(image_root_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # 保存图像
                img_filename = f"{sample_idx:05d}.png"
                img_path = os.path.join(class_dir, img_filename)
                img.save(img_path)

                # 构造图像路径
                if absolute_prefix is None:
                    img_path_for_json = img_path
                else:
                    img_path_for_json = os.path.join(absolute_prefix, class_name, img_filename)

                # 构建 JSONL 数据条目
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
                sample_idx += 1

                # if sample_idx % 100 == 0:
                #     print(f"Processed {sample_idx} samples...")

    print(f"✅ Total {sample_idx} samples saved to {output_jsonl_path}")
    return sample_idx


