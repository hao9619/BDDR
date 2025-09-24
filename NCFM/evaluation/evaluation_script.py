def main_work(args):

    _, val_loader = get_loader(args)

    synset = Condenser(
        args,
        nclass_list=list(range(0, args.nclass)),
        nchannel=args.nch,
        hs=args.size,
        ws=args.size,
        device="cuda",
    )

    for rank in range(args.world_size):
        if rank == args.rank:
            synset.load_condensed_data(
                loader=None, init_type="load", load_path=args.load_path
            )
        dist.barrier()

    syndataloader = synset.get_syndataLoader(args, args.augment)

    synset.evaluate(args, syndataloader, val_loader)
    dist.destroy_process_group()
    


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.utils import get_loader
    from utils.init_script import init_script
    import argparse
    from argsprocessor.args import ArgsProcessor
    from condenser.Condenser import Condenser
    import torch.distributed as dist
    from utils.create_balanced_split import SubsetWithAttributes

    parser = argparse.ArgumentParser(description="Configuration parser")
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="When dataset is very large , you should get it",
    )
    # 添加一个命令行参数，用于指定是否开启debug模式，默认为False
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    # 添加一个命令行参数，用于指定配置文件的路径，必填
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["Condense", "Evaluation", "Pretrain"],
        default="Evaluation",
        help="Condense or Evaluation",
    )
    # 添加一个命令行参数，用于指定运行模式，可选值为Condense、Evaluation、Pretrain，默认为Evaluation
    parser.add_argument(
        "--init",
        type=str,
        default="load",
        choices=["random", "noise", "mix", "load"],
        help="condensed data initialization type",
    )
    # 添加一个命令行参数，用于指定condensed数据的初始化类型，可选值为random、noise、mix、load，默认为load
    parser.add_argument(
        "--load_path", type=str, required=True, help="Path to load the synset"
    )
    # 添加一个命令行参数，用于指定加载synset的路径，必填
    parser.add_argument(
        "--val_repeat",
        type=int,
        default=10,
        help="The times of validation on syn_dataset Imagenet only 3 times",
    )
    # 添加一个命令行参数，用于指定在syn_dataset Imagenet上进行验证的次数，默认为10
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        required=True,
        help='GPUs to use, e.g., "0,1,2,3"',
    )
    # 添加一个命令行参数，用于指定使用的GPU，必填，例如"0,1,2,3"
    parser.add_argument(
        "-i", "--ipc", type=int, default=1, help="number of condensed data per class"
    )
    # 添加一个命令行参数，用于指定每个类别的condensed数据的数量，默认为1
    parser.add_argument("--tf32", action="store_true", default=True, help="Enable TF32")
    # 添加一个命令行参数，用于指定是否启用TF32，默认为True
    parser.add_argument(
        "--softlabel",
        dest="softlabel",
        action="store_true",
        help="Use the softlabel to evaluate the dataset",
    )
    # 添加一个命令行参数，用于指定是否使用softlabel进行数据集评估，默认为False
    parser.add_argument(
        "--kldiv",
        dest="kldiv",
        action="store_true",
        help="Use the kldiv loss to evaluate the dataset",
    )
    # 添加一个命令行参数，用于指定是否使用kldiv损失进行数据集评估，默认为False
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="The temperature for KLdiv"
    )
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    init_script(args)

    main_work(args)
    
    os.system("shutdown")  # AutoDL 定制命令，任务完成后自动关机
