
from utils.util import *
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_data.load_Model import LModel, get_model
from torch import optim
def compute_loss_value(opt, poisoned_data, model_ascent):
    # Calculate loss value per example
    # Define loss function
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []

    example_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
            # print(loss.item())

        losses_record.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in ascending order

    # Show the lowest 10 loss values
    losses_record_arr = np.array(losses_record)
    print('Top ten loss value:', losses_record_arr[losses_idx[:10]])

    return losses_idx


def isolate_data(opt, poisoned_data, losses_idx):
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0
    ratio = opt.isolation_ratio

    example_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )
    # print('full_poisoned_data_idx:', len(losses_idx))
    perm = losses_idx[0: int(len(losses_idx) * ratio)]

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):

        # Filter the examples corresponding to losses_idx
        if idx in perm:
            isolation_examples.append((img, target))
            cnt += 1
        else:
            other_examples.append((img, target))

    print('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    print('Finish collecting {} other examples: '.format(len(other_examples)))
    return isolation_examples, other_examples


def train_step(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        if opt.gradient_ascent_type == 'LGA':
            output = model_ascent(img)
            loss = criterion(output, target)
            # add Local Gradient Ascent(LGA) loss
            loss_ascent = torch.sign(loss - opt.gamma) * loss

        elif opt.gradient_ascent_type == 'Flooding':
            output = model_ascent(img)
            # output = student(img)
            loss = criterion(output, target)
            # add flooding loss
            loss_ascent = (loss - opt.flooding).abs() + opt.flooding

        else:
            raise NotImplementedError

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_ascent.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss_ascent.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'Loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'Prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'Prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

    return acc_clean, acc_bd


def train(opt,poisoned_data_loader, test_clean_loader, test_bad_loader):
    # Load models
    print('----------- Network Initialization --------------')
    
    model_ascent=get_model(opt)
    print('finished model init...')
    optimizer = optim.AdamW(model_ascent.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # define loss functions
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    print('----------- Train Initialization --------------')

    clean_losses = []  # 存储每轮干净测试集损失
    bad_losses = []  # 存储每轮被污染测试集损失
    epochs = list(range(0, opt.tuning_epochs + 1))  # 训练轮次
    for epoch in range(0, opt.tuning_epochs):

        adjust_learning_rate(optimizer, epoch, opt)

        # train every epoch
        if epoch == 0:
            # before training test firstly
            acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model_ascent,
                                      criterion, epoch + 1)
            clean_losses.append(acc_clean[2])  # 收集干净测试集损失
            bad_losses.append(acc_bad[2])  # 收集被污染测试集损失

        train_step(opt, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        # evaluate on testing set
        print('testing the ascended model......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

        clean_losses.append(acc_clean[2])  # 收集干净测试集损失
        bad_losses.append(acc_bad[2])  # 收集被污染测试集损失

    # 绘制损失曲线并保存到当前目录
    #plot_loss_curve(clean_losses, bad_losses, epochs, save_path="loss_curve.png")
    return model_ascent


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch < opt.tuning_epochs:
        lr = opt.lr
    else:
        lr = 0.01
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def MyBI_main(opt, poisoned_data, poisoned_data_loader, test_clean_loader, test_bad_loader):
    print('----------- Train isolated model -----------')

    ascent_model = train(opt, poisoned_data_loader, test_clean_loader, test_bad_loader)

    print('----------- Calculate loss value per example -----------')
    losses_idx = compute_loss_value(opt, poisoned_data, ascent_model)

    print('----------- Collect isolation data -----------')
    isolation_examples, other_examples=isolate_data(opt, poisoned_data, losses_idx)

    return isolation_examples, other_examples



