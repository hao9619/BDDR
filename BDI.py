from torch.utils.data import DataLoader
from torchvision import transforms
from CLR import CLMmain
from model_data.load_Pic import get_data_pic
from utils.bd_Iso import MyBI_main
from model_data.fix_Dataset import getJSONLdata, save_data_to_jsonl

from config import Config


def get_data(opt):
    if opt.load_pic:
        poisoned_data, poisoned_data_loader = get_data_pic(opt, opt.poisoned_data_pic_path)
        _, test_clean_loader = get_data_pic(opt, opt.test_data_clean_pic_path)
        _, test_bad_loader = get_data_pic(opt, opt.test_data_bad_pic_path)
        
    else:
        poisoned_data, my_batch_size = getJSONLdata(opt.poisoned_data_path)
        poisoned_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=my_batch_size,
                                        shuffle=True,
                                        )
        test_data_cl, my_batch_size_cl = getJSONLdata(opt.test_data_clean_path)
        test_clean_loader = DataLoader(dataset=test_data_cl,
                                           batch_size=my_batch_size_cl,
                                           shuffle=True,
                                           )
        test_data_ba, my_batch_size_ba = getJSONLdata(opt.test_data_bad_path)
        test_bad_loader = DataLoader(dataset=test_data_ba,
                                         batch_size=my_batch_size_ba,
                                         shuffle=True,
                                         )

    return poisoned_data, poisoned_data_loader, test_clean_loader, test_bad_loader


def main(opt):
    print("--------Preparing data--------")
    poisoned_data, poisoned_data_loader, test_clean_loader, test_bad_loader = get_data(opt)

    print("--------Isolating poisoned data--------")
    isolation_examples, other_examples = MyBI_main(opt, poisoned_data, poisoned_data_loader, test_clean_loader, test_bad_loader)

    _, my_batch_size_ba = getJSONLdata("/root/autodl-tmp/data/swift_data/cifar10_test.jsonl")
    test_data_iso = isolation_examples
    test_iso_loader = DataLoader(dataset=test_data_iso,
                                 batch_size=my_batch_size_ba,
                                 shuffle=True,
                                 )


    corrected_data = CLMmain(opt, poisoned_data, poisoned_data_loader, other_examples, test_clean_loader,
                                test_iso_loader,
                                )

    corrected_data_loader = DataLoader(dataset=corrected_data,
                                       batch_size=my_batch_size_ba,
                                       shuffle=True,
                                       )
    save_data_to_jsonl(corrected_data_loader, opt.jsonl_output_path, opt.image_root_dir,
                       opt.absolute_prefix)

    save_data_to_jsonl(test_iso_loader, opt.jsonl_output_bad_path, opt.image_root_bad_dir,
                       opt.absolute_bad_prefix)


if __name__ == "__main__":
    opt = Config()
    main(opt)


def bddrmain(opt):
    print("--------Now is running BDDR--------")
    main(opt)