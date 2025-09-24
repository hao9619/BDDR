# scripts/split_and_save_cifar10.py
import os
import pickle
import random
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import Subset

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

def split_cifar10_per_client(data_dir="../dataset", num_clients=10, num_per_class=200, save_path="../dataset/full_balanced_split_200.pkl"):
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    label_to_indices = defaultdict(list)
    for idx, target in enumerate(dataset.targets):
        label_to_indices[target].append(idx)

    # 打乱每类的索引
    for k in label_to_indices:
        random.shuffle(label_to_indices[k])

    client_indices = [[] for _ in range(num_clients)]

    # 轮流分配每类数据给每个客户端
    for c in range(10):  # 类别 0-9
        for client_id in range(num_clients):
            start = client_id * num_per_class
            end = start + num_per_class
            client_indices[client_id].extend(label_to_indices[c][start:end])

    client_datasets = {}
    for client_id in range(num_clients):
        subset = SubsetWithAttributes(dataset, client_indices[client_id], nclass=10)
        client_datasets[client_id] = subset

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(client_datasets, f)

    print(f"Saved client datasets to {save_path}")
    
def get_cifar10_split(save_path="../dataset/full_balanced_split_200.pkl"):
    with open(save_path, "rb") as f:
        return pickle.load(f)
            
if __name__ == "__main__":
    split_cifar10_per_client()
