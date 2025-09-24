import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch.optim as optim
from utils.utils import define_model
from utils.utils import get_loader
from utils.train_val import train_epoch, validate
from utils.diffaug import diffaug


def get_available_model_id(pretrain_dir, model_id):
    while True:
        init_path = os.path.join(pretrain_dir, f"premodel{model_id}_init.pth.tar")
        trained_path = os.path.join(pretrain_dir, f"premodel{model_id}_trained.pth.tar")
        # Check if both files do not exist, if both are missing, return the current model_id
        if not os.path.exists(init_path) and not os.path.exists(trained_path):
            return model_id  # Return the first available model_id
        model_id += 1  # If files exist, try the next model_id


def count_existing_models(pretrain_dir):
    """
    Count the number of initial model files (premodel{model_id}_init.pth.tar)
    that exist in pretrain_dir.
    """
    model_count = 0
    for filename in os.listdir(pretrain_dir):
        if filename.startswith("premodel") and filename.endswith("_init.pth.tar"):
            model_count += 1  # Increment count if the file matches the criteria

    return model_count  # Return the count of matching files
    
# def main_worker(args):
#     train_loader, val_loader, train_sampler = get_loader(args)

#     for model_id in range(args.model_num):
#         if count_existing_models(args.pretrain_dir) >= args.model_num:
#             break
#         model_id = get_available_model_id(args.pretrain_dir, model_id)
#         if args.rank == 0:
#             print(f"Training model {model_id + 1}/{args.model_num}")
#         model = define_model(
#             args.dataset,
#             args.norm_type,
#             args.net_type,
#             args.nch,
#             args.depth,
#             args.width,
#             args.nclass,
#             args.logger,
#             args.size,
#         ).to(args.device)
#         model = model.to(args.device)
#         model = DDP(model, device_ids=[args.rank])

#         # Save initial model state
#         init_path = os.path.join(args.pretrain_dir, f"premodel{model_id}_init.pth.tar")
#         if args.rank == 0 and not os.path.exists(init_path):
#             torch.save(model.state_dict(), init_path)
#             print(f"Model {model_id} initial state saved at {init_path}")

#         # Define loss function, optimizer, and scheduler
#         criterion = torch.nn.CrossEntropyLoss().to(args.device)
#         optimizer = optim.SGD(
#             model.parameters(),
#             lr=args.lr,
#             momentum=args.momentum,
#             weight_decay=args.weight_decay,
#         )
#         scheduler = optim.lr_scheduler.MultiStepLR(
#             optimizer,
#             milestones=[2 * args.pertrain_epochs // 3, 5 * args.pertrain_epochs // 6],
#             gamma=0.2,
#         )
#         _, aug_rand = diffaug(args)
#         for epoch in range(0, args.pertrain_epochs):
#             start_time = time.time()
#             train_sampler.set_epoch(epoch)
#             train_acc1, train_acc5, train_loss = train_epoch(
#                 args,
#                 train_loader,
#                 model,
#                 criterion,
#                 optimizer,
#                 epoch,
#                 aug_rand,
#                 mixup=args.mixup,
#             )
#             val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion)
#             epoch_time = time.time() - start_time
#             if args.rank == 0:
#                 args.logger(
#                     "<Pretraining {:2d}-th model>...[Epoch {:2d}] Train acc: {:.1f} (loss: {:.3f}), Val acc: {:.1f}, Time: {:.2f} seconds".format(
#                         model_id, epoch, train_acc1, train_loss, val_acc1, epoch_time
#                     )
#                 )
#             scheduler.step()

#         # Save trained model state
#         trained_path = os.path.join(
#             args.pretrain_dir, f"premodel{model_id}_trained.pth.tar"
#         )
#         if args.rank == 0:
#             torch.save(model.state_dict(), trained_path)
#             print(f"Model {model_id} trained state saved at {trained_path}")

#     dist.destroy_process_group() 
def average_weights(all_model_weight):
    """
    all_model_weight: 包含多个模型 state_dict 的列表
    返回平均后的 state_dict
    """
    averaged_weights = {}
    for param_name in all_model_weight[0].keys():
        stacked_params = torch.stack(
            [model_params[param_name] for model_params in all_model_weight]
        )
        averaged_weights[param_name] = stacked_params.mean(dim=0)
    return averaged_weights  # 返回 state_dict

def main_worker(args):
    _, val_loader, _ = get_loader(args)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    for model_id in range(args.model_num):
        if count_existing_models(args.pretrain_dir) >= args.model_num:
            break
        model_id = get_available_model_id(args.pretrain_dir, model_id)
        if args.rank == 0:
            print(f"Training model {model_id + 1}/{args.model_num}")
        model = define_model(
            args.dataset,
            args.norm_type,
            args.net_type,
            args.nch,
            args.depth,
            args.width,
            args.nclass,
            args.logger,
            args.size,
        ).to(args.device)
        model = model.to(args.device)
        model = DDP(model, device_ids=[args.rank])
            
        # Save initial model state
        init_path = os.path.join(args.pretrain_dir, f"premodel{model_id}_init.pth.tar")
        if args.rank == 0 and not os.path.exists(init_path):
            torch.save(model.state_dict(), init_path)
            print(f"Model {model_id} initial state saved at {init_path}")
            

        for round in range(2):
            all_model_weight = []
            all_train_acc1 = []
            # all_train_acc5 = []
            all_train_loss = []
            start_time = time.time()
            for client_id in range(10):
                train_loader,_, train_sampler = get_loader(args,client_id = client_id)     
                model = define_model(
                    args.dataset,
                    args.norm_type,
                    args.net_type,
                    args.nch,
                    args.depth,
                    args.width,
                    args.nclass,
                    args.logger,
                    args.size,
                ).to(args.device)
                model = DDP(model, device_ids=[args.rank])
                if round == 0:
                    model.load_state_dict(torch.load(init_path))
                else:
                    trained_path_copy = os.path.join(args.pretrain_dir, f"premodel{model_id}_trained.pth.tar")
                    model.load_state_dict(torch.load(trained_path_copy))
                # Define loss function, optimizer, and scheduler
                # criterion = torch.nn.CrossEntropyLoss().to(args.device)
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[2 * args.pertrain_epochs // 3, 5 * args.pertrain_epochs // 6],
                    gamma=0.2,
                )
                _, aug_rand = diffaug(args)
            
                for epoch in range(0, args.pertrain_epochs):
                    train_sampler.set_epoch(epoch)
                    train_acc1, train_acc5, train_loss = train_epoch(
                    args,
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    epoch,
                    aug_rand,
                    mixup=args.mixup,
                    )
                    scheduler.step()
                all_train_acc1.append(train_acc1)
                # all_train_acc5.append(train_acc5)
                all_train_loss.append(train_loss)
                all_model_weight.append(model.state_dict())

            avg_train_acc1 = sum(all_train_acc1) / len(all_train_acc1)
            # avg_train_acc5 = sum(all_train_acc5) / len(all_train_acc5)
            avg_train_loss = sum(all_train_loss) / len(all_train_loss)
            avg_state_dict = average_weights(all_model_weight)
            model = define_model(
                args.dataset,
                args.norm_type,
                args.net_type,
                args.nch,
                args.depth,
                args.width,
                args.nclass,
                args.logger,
                args.size,
            ).to(args.device)
            model = DDP(model, device_ids=[args.rank])
            model.load_state_dict(avg_state_dict)
            val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion)
            epoch_time = time.time() - start_time
            if args.rank == 0:
                args.logger(
                    "<Pretraining {:2d}-th model>...[Round {:2d}] avg_train_acc: {:.1f} (avg_train_loss: {:.3f}), Val acc: {:.1f}, Time: {:.2f} seconds".format(
                    model_id, round, avg_train_acc1, avg_train_loss, val_acc1, epoch_time
                    )
                )

            # Save trained model state
            trained_path = os.path.join(
            args.pretrain_dir, f"premodel{model_id}_trained.pth.tar"
            )
        
            if args.rank == 0:
                torch.save(model.state_dict(), trained_path)
                print(f"Model {model_id} trained state saved at {trained_path}")
                # all_model_weight = []

    dist.destroy_process_group()


def main():
    import os
    from utils.init_script import init_script
    import argparse
    from argsprocessor.args import ArgsProcessor

    parser = argparse.ArgumentParser(description="Configuration parser")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="When dataset is very large , you should get it",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["Condense", "Evaluation", "Pretrain"],
        default="Pretrain",
        help="Condense or Evaluation",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        required=True,
        help='GPUs to use, e.g., "0,1,2,3"',
    )
    parser.add_argument(
        "-i", "--ipc", type=int, default=1, help="number of condensed data per class"
    )
    parser.add_argument("--load_path", type=str, help="Path to load the synset")
    parser.add_argument("--tf32", action="store_true", default=True, help="Enable TF32")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    init_script(args)

    main_worker(args)


if __name__ == "__main__":
    import os
    from utils.create_balanced_split import SubsetWithAttributes 
    main()
    os.system("shutdown")  # AutoDL 定制命令，任务完成后自动关机
