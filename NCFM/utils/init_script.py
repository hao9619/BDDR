import os
import json
import numpy as np
import torch
import datetime
from .experiment_tracker import Logger
from .diffaug import remove_aug
import torch.distributed as dist
import datetime
from datetime import timedelta
from torch.backends import cudnn
from .ddp import initialize_distribution_training


def init_script(args):
    # 设置cudnn.benchmark为True，以加速卷积操作
    cudnn.benchmark = True
    # 设置torch.backends.cuda.matmul.allow_tf32和torch.backends.cudnn.allow_tf32为args.tf32的值，以启用或禁用TF32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    # 初始化分布式训练，返回rank, world_size, local_rank, local_world_size, device
    rank, world_size, local_rank, local_world_size, device = (
        initialize_distribution_training(args.backend, args.init_method)
    )

    # 设置迭代参数，返回it_save和it_log
    args.it_save, args.it_log = set_iteration_parameters(args.niter, args.debug)

    # 设置预训练目录，返回pretrain_dir
    args.pretrain_dir = set_Pretrain_Directory(
        args.pretrain_dir, args.dataset, args.depth
    )

    # 设置实验名称和保存目录，返回exp_name, save_dir, lr_img
    args.exp_name, args.save_dir, args.lr_img = set_experiment_name_and_save_Dir(
        args.run_mode,
        args.dataset,
        args.pretrain_dir,
        args.save_dir,
        args.lr_img,
        args.lr_scale_adam,
        args.ipc,
        args.optimizer,
        args.load_path,
        args.factor,
        args.lr,
        args.num_freqs,
    )

    # 设置随机种子
    set_random_seeds(args.seed)

    # 调整增强策略，返回mixup, dsa_strategy, dsa, augment
    args.mixup, args.dsa_strategy, args.dsa, args.augment = (
        adjust_augmentation_strategy(args.mixup, args.dsa_strategy, args.dsa)
    )

    # 设置日志和目录，返回logger
    args.logger = setup_logging_and_directories(args, args.run_mode, args.save_dir)
    args.rank, args.world_size, args.local_rank, args.local_world_size, args.device = (
        rank,
        world_size,
        local_rank,
        local_world_size,
        device,
    )
    # 如果rank为0，则输出TF32是否启用
    if args.rank == 0:
        args.logger("TF32 is enabled") if args.tf32 else print("TF32 is disabled")
        args.logger(
            f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}"
        )


def set_iteration_parameters(niter, debug):

    it_save = np.arange(0, niter + 1, 1000).tolist()
    it_log = 1 if debug else 20
    return it_save, it_log


def set_Pretrain_Directory(pretrain_dir, dataset, depth):

    if dataset.lower() == "imagenet":
        pretrain_dir = f"./{pretrain_dir}/{dataset}/ResNet-{depth}"
    else:
        pretrain_dir = f"./{pretrain_dir}/{dataset}"
    return pretrain_dir


def set_experiment_name_and_save_Dir(
    run_mode,
    dataset,
    pretrain_dir,
    save_dir,
    lr_img,
    lr_scale_adam,
    ipc,
    optimizer,
    load_path,
    factor,
    lr,
    num_freqs,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    # Set the base save directory path according to the run_mode
    if run_mode == "Condense":
        assert ipc > 0, "IPC must be greater than 0"
        if optimizer.lower() == "sgd":
            lr_img = lr_img
        else:
            lr_img = lr_img * lr_scale_adam

        # Generate experiment name
        exp_name = f"./condense/{dataset}/ipc{ipc}/{optimizer}_lr_img_{lr_img:.4f}_numr_reqs{num_freqs}_factor{factor}_{timestamp}"
        if load_path:
            exp_name += f"Reload_SynData_Path_{load_path}"
        save_dir = os.path.join(save_dir, exp_name)

    elif run_mode == "Evaluation":
        assert ipc > 0, "IPC must be greater than 0"
        exp_name = (
            f"./evaluate/{dataset}/ipc{ipc}/_lr{lr:.4f}__factor{factor}_{timestamp}"
        )
        save_dir = os.path.join(save_dir, exp_name)
    elif run_mode == "Pretrain":
        save_dir = pretrain_dir
        exp_name = pretrain_dir
    else:
        raise ValueError(
            "Invalid run_mode. Choose 'Condense', 'Evaluation' or 'Pretrain'."
        )

    # Create save directory if the rank is 0
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)

    return exp_name, save_dir, lr_img


def set_random_seeds(seed):

    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if dist.get_rank() == 0:
            print(f"Set Random Seed as {seed}")


def setup_logging_and_directories(args, run_mode, save_dir):
    if dist.get_rank() == 0:
        if run_mode == "Condense":
            subdirs = ["images", "distilled_data"]
            for subdir in subdirs:
                os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
        args_log_path = os.path.join(save_dir, "args.log")
        with open(args_log_path, "w") as f:
            json.dump(vars(args), f, indent=3)
    dist.barrier()
    logger = Logger(args.save_dir)
    dist.barrier()
    if dist.get_rank() == 0:
        logger(f"Save dir: {args.save_dir}")

    return logger


def adjust_augmentation_strategy(mixup, dsa_strategy, dsa):

    if mixup == "cut":
        dsa_strategy = remove_aug(dsa_strategy, "cutout")

    if dsa:
        augment = False
        if dist.get_rank() == 0:
            print(
                "DSA strategy: ",
                dsa_strategy,
            )
    else:
        augment = True
    return mixup, dsa_strategy, dsa, augment
