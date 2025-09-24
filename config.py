class Config:
    def __init__(self):
        # Global setting -----------------
        self.model_name: str = 'WRN-16-1'
        self.dataset: str = 'CIFAR10'
        self.batch_size: int = 128
        self.num_class: int = 10
        ##device setting
        self.cuda: int = 1
        self.device: str = 'cuda'

        self.PublicAddress: str = "/root/autodl-tmp/data/swift_data"

        # data & model preparing -----------------
        
        self.load_pic: int = 1

        self.poisoned_data_path: str =  self.PublicAddress +'/cifar10_test.jsonl'
        self.test_data_clean_path: str = self.PublicAddress +'/cifar10_test.jsonl'
        self.test_data_bad_path: str =  self.PublicAddress +'/cifar10_test.jsonl'

        self.poisoned_data_pic_path: str = self.PublicAddress + '/1000_cifar10_poisoned_images'
        self.test_data_clean_pic_path: str = self.PublicAddress +'/cifar10_test_images'
        self.test_data_bad_pic_path: str = self.PublicAddress + '/1000_cifar10_poisoned_images'

        self.fixed_model_root: str = './cifar10'
        self.load_fixed_model: int = 0

        # self.batch_size:int =32

        # BDI module -----------------(Backdoor_Detection_and_Isolation)
        self.print_freq: int = 200
        self.tuning_epochs: int = 10
        self.gradient_ascent_type: str = 'Flooding'
        self.isolation_ratio: float = 0.1
        self.gamma: float = 0.5
        self.flooding: float = 0.5
        self.lr: float = 0.01
        self.momentum: float = 0.9
        self.weight_decay: float = 1e-4

        # CLR module ------------------(Correct_Label_Restoration)
        self.CLR_eva_epochs: int = 20
        self.CLR_noise_intensity: int = 0.1
        self.CLR_teahcer_epochs: int = 20
        self.CLR_student_epochs: int = 40
        self.CLR_delta: float = 2.5
        self.CLR_gamma: float = 0.8

        self.CLR_lr: float = 0.0005
        self.CLR_momentum: float = 0.9
        self.CLR_weight_decay: float = 1e-4

        self.CLR_stu_lr: float = 0.0005
        self.CLR_stu_momentum: float = 0.9
        self.CLR_stu_weight_decay: float = 1e-4

        self.CLR_tea_lr: float = 0.0005
        self.CLR_tea_momentum: float = 0.9
        self.CLR_tea_weight_decay: float = 1e-4

        # Corrected data Output ---------------
        self.save: int = 1
        
        self.jsonl_output_path = self.PublicAddress + "/isolate_data.jsonl"
        self.image_root_dir = self.PublicAddress + "/isolate_images"
        self.absolute_prefix = self.PublicAddress + "/isolate_images"

        self.jsonl_output_bad_path = self.PublicAddress + "/isolate_bad_data.jsonl"
        self.image_root_bad_dir = self.PublicAddress + "/isolate_bad_images"
        self.absolute_bad_prefix = self.PublicAddress + "/isolate_bad_images"

        # Evaluate Unite ------------------------------------
        self.EU_epochs: int = 20

        self.EU_dataset: str = 'CIFAR10'
        self.EU_model_name: str = 'WRN-40-2'
        
        self.EU_load_pic: int = 1

        self.EU_poisoned_data_path: str = self.PublicAddress + '/isolate_data.jsonl'
        self.EU_test_data_clean_path: str = self.PublicAddress +'/cifar10_test.jsonl'
        self.EU_test_data_bad_path: str = self.PublicAddress + '/isolate_data.jsonl'

        self.EU_poisoned_data_pic_path: str = self.PublicAddress + '/1000_cifar10_poisoned_images'
        self.EU_test_data_clean_pic_path: str = self.PublicAddress +'/cifar10_test_images'
        self.EU_test_data_bad_pic_path: str = self.PublicAddress + '/1000_cifar10_poisoned_images'

        self.EU_lr: float = 0.0005
        self.EU_momentum: float = 0.9
        self.EU_weight_decay: float = 1e-4