# [CVPR2025] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective 

Official PyTorch implementation of the paper ["Dataset Distillation with Neural Characteristic Function"](https://arxiv.org/abs/2502.20653) (NCFM) in CVPR 2025.


## :fire: News

- [2025/03/02] The code of our paper has been released.  
- [2025/02/27] Our NCFM paper has been accepted to CVPR 2025 (Rating: 555). Thanks!  


## :rocket: Pipeline

Here's an overview of the process behind our **Neural Characteristic Function Matching (NCFM)** method:

![Figure 1](./asset/figure1.png?raw=true)





## :mag: TODO
<font color="red">**We are currently organizing all the code. Stay tuned!**</font>
- [x] Distillation code
- [x] Evaluation code
- [x] Sampling network
- [x] Config files
- [ ] Pretrained models
- [ ] Distilled datasets
- [ ] Project page




## 🛠️ Getting Started

To get started with NCFM, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/gszfwsb/NCFM.git
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```
3. Pretrain the models yourself, or download the **pretrained_models** from [Google Drive](https://drive.google.com/drive/folders/1HT_eUbTWOVXvBov5bM90b169jdy2puOh?usp=drive_link). 
```sh
cd pretrain
torchrun --nproc_per_node=1 --nnodes=1 pretrain_script.py --gpu="0" --config_path=../config/ipc10/cifar10.yaml

```

4. Condense
```sh
cd condense 
torchrun --nproc_per_node=1 --nnodes=1 condense_script.py --gpu="0" --ipc=10 --config_path=../config/ipc10/cifar10.yaml

```
5. Evaluation
```sh
cd evaluation 
torchrun --nproc_per_node=1 --nnodes=1 evaluation_script.py --gpu="0" --ipc=10 --config_path=../config/ipc10/cifar10.yaml --load_path=../results_test/condense/condense/cifar10/ipc10/adamw_lr_img_0.0010_numr_reqs4096_factor2_20250613-1856/distilled_data/data_20000
```

### :blue_book: Example Usage

1. CIFAR-10

```sh
#ipc50
cd condense
torchrun --nproc_per_node=8 --nnodes=1 --master_port=34153 condense_script.py --gpu="0,1,2,3,4,5,6,7" --ipc=50 --config_path=../config/ipc50/cifar10.yaml
```

2. CIFAR-100

```sh
#ipc10
cd condense
torchrun --nproc_per_node=8 --nnodes=1 --master_port=34153 condense_script.py --gpu="0,1,2,3,4,5,6,7" --ipc=10 --config_path=../config/ipc10/cifar100.yaml
```



## :postbox: Contact
If you have any questions, please contact [Shaobo Wang](https://gszfwsb.github.io/)(`shaobowang1009@sjtu.edu.cn`).

## :pushpin: Citation
If you find NCFM useful for your research and applications, please cite using this BibTeX:

```bibtex
@inproceedings{wang2025NCFM,
      title={Dataset Distillation with Neural Characteristic Function: A Minmax Perspective}, 
      author={Shaobo Wang and Yicun Yang and Zhiyuan Liu and Chenghao Sun and Xuming Hu and Conghui He and Linfeng Zhang},
 booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2025}
}
```

## Acknowledgement
We sincerely thank the developers of the following projects for their valuable contributions and inspiration: [MTT](https://github.com/GeorgeCazenavette/mtt-distillation), [DATM](https://github.com/NUS-HPC-AI-Lab/DATM), [DC/DM](https://github.com/VICO-UoE/DatasetCondensation), [IDC](https://github.com/snu-mllab/Efficient-Dataset-Condensation), [SRe2L](https://github.com/VILA-Lab/SRe2L), [RDED](https://github.com/LINs-lab/RDED), [DANCE](https://github.com/Hansong-Zhang/DANCE). We draw inspiration from these fantastic projects!
