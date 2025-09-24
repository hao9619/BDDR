import os
from collections import OrderedDict
import model_data.fix_Convnet as CN
import torch

from model_data.fix_Model import ResNet18


def get_model(opt, fix=None):
    if fix!=None:
        opt.load_fixed_model=fix
    if opt.load_fixed_model:
        model = LModel(opt)
    else:
        
        model = ResNet18()
        model.to(opt.device)

    return model


def load_state_dict(state_dict_path, model):
    state_dict = torch.load(state_dict_path, map_location="cpu")
    # Remove `module.` prefix from keys if it exists
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)


def define_model(opt,norm_type="instance",
                 net_type="convnet", nch=3, depth=3, width=1.0, nclass=10, size=32):

    if net_type == "convnet":
        width = int(128 * width)

        model = CN.ConvNet(
            nclass,
            net_norm=norm_type,
            net_depth=depth,
            net_width=width,
            channel=nch,
            im_size=(size, size),
        )
    else:
        raise Exception("unknown network architecture: {}".format(net_type))

    # if logger is not None:
    #     if dist.get_rank() == 0:
    #         logger(f"=> creating model {net_type}-{depth}, norm: {norm_type}")
    #         logger('# model parameters: {:.1f}M'.format(sum([p.data.nelement() for p in model.parameters()]) / 10**6))
    return model


def load_model(opt):
    teacher_model = define_model(opt).to(opt.device)  # 将教师模型加载到指定设备（如 GPU）

    # 定义教师模型权重文件路径
    teacher_path = os.path.join(opt.fixed_model_root, f"premodel9_trained.pth.tar")

    # 加载教师模型的预训练权重
    load_state_dict(teacher_path, teacher_model)

    return teacher_model


def LModel(opt):
    model = load_model(opt)
    return model