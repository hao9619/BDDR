from torch.utils.data import DataLoader


from model_data.fix_Dataset import getJSONLdata


def get_data_pic(opt, data_path):
    import os
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 图片大小为 32x32
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
    ])

    # 使用 ImageFolder 加载数据集
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    from collections import Counter
   

    # 创建 DataLoader
    my_batch_size = 64  # 设置你的 batch size
    data_loader = DataLoader(dataset=dataset, batch_size=my_batch_size, shuffle=True)

    

    return dataset, data_loader

