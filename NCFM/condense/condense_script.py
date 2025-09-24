def main_worker(args):
    
    args.class_list = distribute_class(args.nclass,args.debug)

    plotter = get_plotter(args)

    # loader_real,_ = get_loader(args)


    # aug, _ = diffaug(args)
    
    # condenser = Condenser(args, nclass_list=args.class_list, nchannel=args.nch, hs=args.size, ws=args.size, device='cuda')
    # for local_rank in range(args.local_world_size):
    #     if  args.local_rank == local_rank:
    #         condenser.load_condensed_data(loader_real, init_type=args.init,load_path=args.load_path)
    #         print(f"============RANK:{dist.get_rank()}====LOCAL_RANK {local_rank} Loaded Condensed Data==========================")
    #     dist.barrier()

    # optim_img = get_optimizer(optimizer=args.optimizer, parameters=condenser.parameters(),lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    # if args.sampling_net:
    #         sampling_net = SampleNet(feature_dim=2048)
    #         optim_sampling_net = get_optimizer(optimizer=args.optimizer, parameters=sampling_net,lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    # else:
    #     sampling_net = None
    #     optim_sampling_net = None
    # model_init,model_interval,model_final = get_feature_extractor(args)
    # condenser.condense(args,plotter,loader_real,aug,optim_img,model_init,model_interval,model_final,sampling_net,optim_sampling_net)

    # dist.destroy_process_group()
    
    for client_id in range(10):
        # 获取真实数据加载器
        loader_real,_ = get_loader(args,client_id)

        # 获取差分增强
        aug, _ = diffaug(args)
    
        # 初始化Condenser
        condenser = Condenser(args, nclass_list=args.class_list, nchannel=args.nch, hs=args.size, ws=args.size, device='cuda')
        # 遍历所有本地进程
        for local_rank in range(args.local_world_size):
            # 如果本地进程号与当前进程号相同
            if  args.local_rank == local_rank:
                # 加载Condenser数据
                condenser.load_condensed_data(loader_real, init_type=args.init,load_path=args.load_path)
                # 打印加载Condenser数据的信息
                print(f"============RANK:{dist.get_rank()}====LOCAL_RANK {local_rank} Loaded Condensed Data==========================")
            # 所有进程等待
            dist.barrier()

        # 获取Condenser的优化器
        optim_img = get_optimizer(optimizer=args.optimizer, parameters=condenser.parameters(),lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
        # 如果有采样网络
        if args.sampling_net:
            # 初始化采样网络
            sampling_net = SampleNet(feature_dim=2048)
            # 获取采样网络的优化器
            optim_sampling_net = get_optimizer(optimizer=args.optimizer, parameters=sampling_net,lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
        else:
            # 否则，采样网络为None
            sampling_net = None
            # 采样网络的优化器为None
            optim_sampling_net = None
        # 获取特征提取器
        model_init,model_interval,model_final = get_feature_extractor(args)
        # Condenser进行压缩
        condenser.condense(args,plotter,loader_real,aug,optim_img,model_init,model_interval,model_final,sampling_net,optim_sampling_net,client_id=client_id)

    dist.destroy_process_group()
    

if __name__ == '__main__':
    import sys
    import os
    import torch
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.diffaug import diffaug
    import torch.distributed as dist
    from  utils.ddp import distribute_class
    from  utils.utils import get_plotter,get_optimizer,get_loader,get_feature_extractor
    from utils.init_script import init_script
    import argparse
    from argsprocessor.args import ArgsProcessor
    from condenser.Condenser import Condenser
    from NCFM.SampleNet import SampleNet
    from utils.create_balanced_split import get_cifar10_split, SubsetWithAttributes

    parser = argparse.ArgumentParser(description='Configuration parser')
    # 添加一个参数，用于调试，当数据集非常大时，应该获取它
    parser.add_argument('--debug',dest='debug',action='store_true',help='When dataset is very large , you should get it')
    # 添加一个参数，用于指定YAML配置文件的路径
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    # 添加一个参数，用于指定运行模式，可选值为Condense、Evaluation和Pretrain，默认为Condense
    parser.add_argument('--run_mode',type=str,choices=['Condense', 'Evaluation',"Pretrain"],default='Condense',help='Condense or Evaluation')
    # 添加一个参数，用于指定condensation matching objective的augmentation策略，默认为color_crop_cutout
    parser.add_argument('-a','--aug_type',type=str,default='color_crop_cutout',help='augmentation strategy for condensation matching objective')
    # 添加一个参数，用于指定condensed data的初始化类型，可选值为random、noise、mix和load，默认为mix
    parser.add_argument('--init',type=str,default='mix',choices=['random', 'noise', 'mix', 'load'],help='condensed data initialization type')
    # 添加一个参数，用于指定加载synset的路径，默认为None
    parser.add_argument('--load_path',type=str,default=None,help="Path to load the synset")
    # 添加一个参数，用于指定使用的GPU，默认为"0"
    parser.add_argument('--gpu', type=str, default = "0",required=True, help='GPUs to use, e.g., "0,1,2,3"')
    # 添加一个参数，用于指定每个类别的condensed data的数量，默认为1
    parser.add_argument('-i', '--ipc', type=int, default=1,required=True, help='number of condensed data per class')
    # 添加一个参数，用于启用TF32
    parser.add_argument('--tf32', action='store_true',default=True,help='Enable TF32')
    # 添加一个参数，用于启用sampling_net
    parser.add_argument('--sampling_net', action='store_true',default=False,help='Enable sampling_net')
    # parser.add_argument('--client_num', type=int, default=10,help='client number ,e.g.,"10"')
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 调用init_script函数，传入args参数
    init_script(args)

    main_worker(args)