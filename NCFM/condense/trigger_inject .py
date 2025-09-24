import os
import pickle
import random
from collections import Counter
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import transforms

# ---------- 定义 SubsetWithAttributes ----------
class SubsetWithAttributes(Subset):
    def __init__(self, dataset, indices, nclass):
        super().__init__(dataset, indices)
        self.nclass = nclass
        if hasattr(dataset, 'targets'):
            self.targets = [dataset.targets[i] for i in indices]
        elif hasattr(dataset, 'labels'):
            self.targets = [dataset.labels[i] for i in indices]
        else:
            raise AttributeError("Dataset has neither 'targets' nor 'labels'.")

# ---------- 添加触发器 ----------
def add_backdoor_trigger(img, trigger_size=2, trigger_value=255):
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    img_np = np.array(img)
    img_np[-trigger_size:, -trigger_size:, :] = trigger_value
    return Image.fromarray(img_np)

# ---------- 注入后门到 subset ----------
def inject_backdoor_in_subset(subset, poison_ratio=1, target_label=0, trigger_size=2):
    indices = subset.indices
    dataset = subset.dataset

    num_poison = int(len(indices) * poison_ratio)
    poisoned_indices = random.sample(indices, num_poison)

    for idx in poisoned_indices:
        img, _ = dataset[idx]
        img = add_backdoor_trigger(img, trigger_size=trigger_size)
        img = transforms.ToTensor()(img)
        dataset.data[idx] = (img.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        dataset.targets[idx] = target_label

# ---------- 主函数 ----------
def inject_backdoor_to_clients(
    input_path,
    output_path,
    poisoned_client_ids,
    poison_ratio,
    target_label,
    trigger_size
):
    with open(input_path, "rb") as f:
        client_datasets = pickle.load(f)

    print(f"Loaded {len(client_datasets)} clients from: {input_path}\n")

    for cid, subset in client_datasets.items():
        if cid in poisoned_client_ids:
            print(f"🔥 Injecting backdoor into client {cid} (poison_ratio={poison_ratio})")
            inject_backdoor_in_subset(subset, poison_ratio, target_label, trigger_size)
        else:
            print(f"Client {cid} remains clean")

    # print("\n Label distribution per client after injection:")
    # for cid, subset in client_datasets.items():
    #     counter = Counter(subset.targets)
    #     print(f"Client {cid}: {dict(counter)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(client_datasets, f)

    print(f"\n Saved backdoored dataset to: {output_path}")

# ---------- 调用 ----------
if __name__ == "__main__":
    #蒸馏阶段只是添加触发器，但并未修改中毒标签，蒸馏完成后在修改中毒标签
    inject_backdoor_to_clients(
        input_path="../dataset/full_balanced_split_200.pkl",      # 你的无后门原始数据
        output_path="../dataset/full_poisoned_split_200_trigger4.pkl",     # 要保存的新路径
        poisoned_client_ids=[0],                                  # 注入的客户端 ID
        poison_ratio=1,                                           # 注入比例
        target_label=0,                                           # 后门统一标签
        trigger_size=4                                            # 右下角方块大小
    )
