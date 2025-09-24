import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from model_data.load_Model import LModel, get_model
from model_data.load_Pic import get_data_pic
from model_data.fix_Dataset import getJSONLdata


from utils.util import accuracy, AverageMeter
from model_data.fix_Model import ResNet18
import torch
from torch.utils.data import DataLoader
from utils.util import AverageMeter
from torchvision import models
from model_data.fix_Dataset import getJSONLdata, save_data_to_jsonl



def get_data(opt):
    if opt.EU_load_pic:
        poisoned_data, poisoned_data_loader = get_data_pic(opt, opt.EU_poisoned_data_pic_path)
        _, test_clean_loader = get_data_pic(opt, opt.EU_test_data_clean_pic_path)
        _, test_bad_loader = get_data_pic(opt, opt.EU_test_data_bad_pic_path)
    
    else:
        poisoned_data, my_batch_size = getJSONLdata(opt.EU_poisoned_data_path)
        poisoned_data_loader = DataLoader(dataset=poisoned_data,
                                              batch_size=my_batch_size,
                                              shuffle=True,
                                              )

            
        test_data_cl, my_batch_size_cl = getJSONLdata(opt.EU_test_data_clean_path)
        test_clean_loader = DataLoader(dataset=test_data_cl,
                                           batch_size=my_batch_size_cl,
                                           shuffle=True,
                                           )
        test_data_ba, my_batch_size_ba = getJSONLdata(opt.EU_test_data_bad_path)
        test_bad_loader = DataLoader(dataset=test_data_ba,
                                         batch_size=my_batch_size_ba,
                                         shuffle=True,
                                         )

    return poisoned_data, poisoned_data_loader, test_clean_loader, test_bad_loader


def evaluate_m(opt, teacher_model, test_clean_loader, test_bad_loader):
    teacher_model.eval()

    clean_accuracy = AverageMeter()
    poisoned_accuracy = AverageMeter()

    for loader, meter in [(test_clean_loader, clean_accuracy), (test_bad_loader, poisoned_accuracy)]:
        for img, target in loader:
            
            img = img.cuda() if opt.cuda else img
            target = target.cuda() if opt.cuda else target
            with torch.no_grad():
                teacher_output = teacher_model(img)

                final_output = teacher_output

                # 计算准确率
                prec1 = accuracy(final_output, target, topk=(1,))
                meter.update(sum(prec1) / len(prec1), img.size(0))

    print(f"Clean Accuracy: {clean_accuracy.avg:.2f}")
    print(f"Poisoned Accuracy: {poisoned_accuracy.avg:.2f}")


def evaluate_corrected_dataset(opt, model, data_loader, test_clean_loader, test_bad_loader):
    """
    使用教师模型评估修正后的数据集。
    """

    optimizer = optim.AdamW(model.parameters(), lr=opt.EU_lr, weight_decay=opt.EU_weight_decay)
    criterion = nn.CrossEntropyLoss().cuda() if opt.cuda else nn.CrossEntropyLoss()
    model.train()
    for epoch in range(opt.EU_epochs):
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
        print(f"Epoch [{epoch + 1}/{opt.EU_epochs}] - Average Loss: {epoch_loss / len(data_loader)}")

        if epoch%100==0:
            model.eval()

            clean_accuracy = AverageMeter()
            poisoned_accuracy = AverageMeter()

            for loader, meter in [(test_clean_loader, clean_accuracy), (test_bad_loader, poisoned_accuracy)]:
                for img, target in loader:
                    # img = img.squeeze(1)
                    if opt.cuda:
                        img = img.cuda()
                        target = target.cuda()
                    with torch.no_grad():
                        output = model(img)
                        prec1 = accuracy(output, target, topk=(1,))
                        meter.update(sum(prec1) / len(prec1), img.size(0))

            print(f"Corrected Dataset Accuracy: ")
            print(f"Clean : {clean_accuracy.avg:.2f}")
            print(f"Poisoned : {poisoned_accuracy.avg:.2f}")
            model.train()

    model.eval()

    clean_accuracy = AverageMeter()
    poisoned_accuracy = AverageMeter()

    for loader, meter in [(test_clean_loader, clean_accuracy), (test_bad_loader, poisoned_accuracy)]:
        for img, target in loader:
            # img = img.squeeze(1)
            if opt.cuda:
                img = img.cuda()
                target = target.cuda()
            with torch.no_grad():
                output = model(img)
                prec1 = accuracy(output, target, topk=(1,))
                meter.update(sum(prec1) / len(prec1), img.size(0))

    print(f"Corrected Dataset Accuracy: ")
    print(f"Clean : {clean_accuracy.avg:.2f}")
    print(f"Poisoned : {poisoned_accuracy.avg:.2f}")




# 整合到主流程中
def main_nearest_neighbor(opt):
    print("--------Preparing data--------")
    poisoned_data, poisoned_data_loader, test_clean_loader, test_bad_loader = get_data(opt)

    print("--------Evaluating Nearest Neighbor Classification--------")
    print("Evaluating Poisoned Data Loader:")
    poisoned_accuracy = nearest_neighbor_test(opt, poisoned_data_loader, test_clean_loader)

    print("Evaluating Bad Data Loader:")
    bad_accuracy = nearest_neighbor_test(opt, poisoned_data_loader, test_bad_loader)

    print(f"Final Results:")
    print(f"Clean Data Accuracy: {poisoned_accuracy:.2f}%")
    print(f"Bad Data Accuracy: {bad_accuracy:.2f}%")


def main_model(opt, model):
    print("--------Preparing data--------")
    poisoned_data, poisoned_data_loader, test_clean_loader, test_bad_loader = get_data(opt)

    print("--------Evaluating model--------")
    evaluate_m(opt, model, test_clean_loader, test_bad_loader)


def main_data(opt):
    print("--------Preparing data--------")
    poisoned_data, poisoned_data_loader, test_clean_loader, test_bad_loader = get_data(opt)

    print("--------Preparing model--------")
    model = get_model(opt)

    print("--------Evaluating model--------")
    
    evaluate_corrected_dataset(opt, model, poisoned_data_loader, test_clean_loader, test_bad_loader)

    


def EUmain(opt, mode="data", model=None):
    if mode == "data":
        print("--------Now is running EvaUnite for data--------")
        main_data(opt)
        # main_nearest_neighbor(opt)  # 添加最近邻分类测试
    elif mode == "model":
        print("--------Now is running EvaUnite for model--------")
        if model:
            main_model(opt, model)
        else:
            print("model error")
    else:
        print("mode error")

