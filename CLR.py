import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from model_data.load_Model import LModel, get_model
from EvaUnite import EUmain
from model_data.load_Pic import get_data_pic
from utils.util import accuracy, AverageMeter

from model_data.fix_Dataset import getJSONLdata, save_data_to_jsonl




def train_teacher_model(opt, data_loader):
    model=get_model(opt)
    optimizer = optim.AdamW(model.parameters(), lr=opt.CLR_tea_lr, weight_decay=opt.CLR_tea_weight_decay)
    criterion = nn.CrossEntropyLoss().cuda() if opt.cuda else nn.CrossEntropyLoss()

    model.train()
    for epoch in range(opt.CLR_teahcer_epochs):
        epoch_loss = 0.0  # 用于累加当前 epoch 的损失
        for img, target in data_loader:
            # print("Image shape of TEA:", img.shape)  # 检查原始形状
            if opt.cuda:
                img = img.cuda()
                target = target.cuda()
            output = model(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # 累加损失
        print(f"Epoch [{epoch + 1}/{opt.CLR_teahcer_epochs}] - Average Loss: {epoch_loss / len(data_loader)}")

    return model


# Step 5: Train Student Model
def train_student_model(opt, teacher_model, isolated_data_loader):
    student_model=get_model(opt)
    optimizer = optim.AdamW(student_model.parameters(), lr=opt.CLR_stu_lr, weight_decay=opt.CLR_stu_weight_decay)
    criterion = nn.MSELoss()
    # teacher_model.eval()

    student_model.train()
    if len(isolated_data_loader) == 0:
        print("No isolated data found, skipping student_model training.")
        return student_model

    for epoch in range(opt.CLR_student_epochs):
        epoch_loss = 0.0  # 用于累加当前 epoch 的损失
        for img, target in isolated_data_loader:
            # print("Image shape of STU:", img.shape)  # 检查原始形状
            # 转换 img 为 torch.Tensor 类型
            img = torch.tensor(img) if isinstance(img, np.ndarray) else img
            # 如果目标是 Long 类型，转换为 Float 类型
            target = torch.tensor(target) if isinstance(target, np.ndarray) else target
            target = target.float()  # 如果使用 CrossEntropyLoss，保持目标为 Long 类型
            img = img.squeeze(1)

            if opt.cuda:
                img = img.cuda()
                target = target.cuda()

            teacher_output = teacher_model(img)
            student_output = student_model(img)
            loss = criterion(student_output, teacher_output)
            # loss = criterion(student_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # 累加损失
        print(f"Epoch [{epoch + 1}/{opt.CLR_student_epochs}] - Average Loss: {epoch_loss / len(isolated_data_loader)}")

    return student_model


def correct_poisoned_labels(opt, teacher_model, student_model, poisoned_data_loader, test_iso_loader, delta, gamma):
    """
    使用教师模型预测被后门数据集的正确类别，并修改标签。
    """
    corrected_data = []  # 存储标签已修正的数据
    teacher_model.eval()
    student_model.eval()

    for img, target in poisoned_data_loader:
        img = img.squeeze(1)
        if opt.cuda:
            img = img.cuda()
            # target = target.cuda()
        with torch.no_grad():
            # predicted_label = target.cpu()
            # 使用教师模型预测正确类别

            teacher_output = teacher_model(img)
            student_output = student_model(img)

            # 计算教师模型的平均激活值 A
            A = torch.mean(teacher_output, dim=1, keepdim=True)  # 按类别维度计算平均值

            # 根据公式计算最终输出
            relu_term = torch.relu(student_output - A * gamma)  # 计算 ReLU(fs - A * γ)
            output = teacher_output - delta * relu_term  # 计算最终输出
            predicted_label = torch.argmax(output, dim=1).cpu().numpy()  # 转换为 NumPy 数组

            
            # 将每张图片和其预测标签存储到列表
            for i in range(img.size(0)):  # 遍历批量中的每张图片
                corrected_data.append((img[i].cpu(), predicted_label[i]))
        

    print("已修正被后门数据集的标签。")
    print('{} examples: '.format(len(corrected_data)))
    return corrected_data


def weaken_backdoor_effect(data, noise_intensity=0.1):
    """
    对修正后的数据集进行处理，削弱后门特征。
    可以通过添加噪声或其他方式来降低后门特征的影响。
    Args:
        data: 修正后的数据集 [(img, label), ...]
        noise_intensity: 噪声强度，默认值为 0.1。
    Returns:
        processed_data: 处理后的数据集 [(processed_img, label), ...]
    """
    processed_data = []
    for img, label in data:
        img = img.clone()
        # 添加随机噪声
        noise = torch.randn_like(img) * noise_intensity
        processed_img = img + noise
        # 限制图像值在 [0, 1] 范围内
        processed_img = torch.clamp(processed_img, 0, 1)
        processed_data.append((processed_img, label))
    print("已对修正后的数据集进行处理，削弱后门特征。")
    return processed_data


def evaluate_corrected_dataset(opt, corrected_data, test_clean_loader, test_bad_loader, load_fixed_model, Lr):
    """
    使用教师模型评估修正后的数据集。
    """

    model = get_model(opt, load_fixed_model)
    optimizer = optim.AdamW(model.parameters(), lr=opt.CLR_lr, weight_decay=opt.CLR_weight_decay)
    criterion = nn.CrossEntropyLoss().cuda() if opt.cuda else nn.CrossEntropyLoss()
    data_loader = DataLoader(corrected_data, batch_size=opt.batch_size, shuffle=False)
    model.train()
    for epoch in range(opt.CLR_eva_epochs):
        epoch_loss = 0.0  # 用于累加当前 epoch 的损失
        for img, target in data_loader:
            
            if opt.cuda:
                img = img.cuda()
                target = target.cuda()
            output = model(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # 累加损失
        if epoch%10==0:
            print(f"Epoch [{epoch + 1}/{opt.CLR_eva_epochs}] - Average Loss: {epoch_loss / len(data_loader)}")

    model.eval()

    clean_accuracy = AverageMeter()
    poisoned_accuracy = AverageMeter()

    for loader, meter in [(test_clean_loader, clean_accuracy), (test_bad_loader, poisoned_accuracy)]:
        for img, target in loader:
            img = img.squeeze(1)
            
            if opt.cuda:
                img = img.cuda()
                target = target.cuda()
            with torch.no_grad():
                output = model(img)
                prec1 = accuracy(output, target, topk=(1,))
                meter.update(sum(prec1) / len(prec1), img.size(0))

    clR=clean_accuracy.avg
    poiR=poisoned_accuracy.avg
    print(f"Corrected Dataset Accuracy: ")
    print(f"Clean : {clR:.2f}")
    print(f"Poisoned : {poiR:.2f}")
    return clR,poiR





def CLMmain(opt, poisoned_data, poisoned_data_loader, other_examples, test_clean_loader, test_iso_loader):
    print("--------Training teacher model--------")
    teacher_model = train_teacher_model(opt, poisoned_data_loader)

    print("--------Training student model--------")
    student_model = train_student_model(opt, teacher_model, test_iso_loader)
    print("--------Evaluating teacher and student model--------")
    # EUmain(opt, mode="model", model=teacher_model)
    EUmain(opt, mode="model", model=student_model)

    i=0
    while i<5:
        print("--------Correcting poisoned dataset labels--------")
        corrected_data_bd = correct_poisoned_labels(opt, teacher_model, student_model, poisoned_data_loader,
                                                    test_iso_loader,
                                                    delta=opt.CLR_delta, gamma=opt.CLR_gamma)

        

        merged_data = corrected_data_bd

        _, test_bad_loader = get_data_pic(opt, "/root/autodl-tmp/data/swift_data/badnets_test")

        print("--------Evaluating corrected poisoned dataset--------")
        clR, poiR = evaluate_corrected_dataset(opt, merged_data, test_clean_loader, test_bad_loader, False, Lr=0.0005)
        

        print("--------------------clR: ", clR, "poiR: ", poiR, "eqoch", i,"delta",opt.CLR_delta,"gamma",opt.CLR_gamma,"---------------------")
        break
        if clR >= 21 and poiR <= 1:
            break
        opt.CLR_delta=opt.CLR_delta+(poiR-1)*0.008+(clR-35)*0.008
        opt.CLR_gamma=opt.CLR_gamma+(poiR-1)*0.004-(clR-35)*0.004
        i+=1


    return merged_data

