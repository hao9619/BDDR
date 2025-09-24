# BDDR: A Privacy-Preserving Backdoor Defense for LLM Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ICASSP%202026-orange.svg)](https://github.com/hao9619/BDDR)

## 📖 Overview

**This is the official implementation of the paper:**

> **BDDR: A Privacy-Preserving Backdoor Defense for LLM Fine-Tuning via Federated Knowledge Distillation**  
> *Hao Zhou, Hao Huang, Hua Dai, Jing Luo, Yubo Ni, Geng Yang*  
> **ICASSP 2026** - IEEE International Conference on Acoustics, Speech and Signal Processing  
> School of Computer Science, Nanjing University of Post and Telecommunication

BDDR (Backdoor Defense via Distillation and Restoration) is a novel privacy-preserving backdoor defense framework that couples federated knowledge distillation with backdoor detection and data restoration. This framework addresses two critical challenges in LLM fine-tuning: privacy leakage under distributed training and backdoor vulnerabilities introduced by poisoned clients.

### Key Features

- **🔒 Privacy-Preserving**: Federated knowledge distillation protects sensitive data
- **🛡️ Backdoor Defense**: Advanced detection and restoration mechanisms
- **🧠 LLM Compatible**: Supports fine-tuning of large language models
- **📊 High Performance**: Reduces attack success rate to ≤0.2% while maintaining competitive clean accuracy

## 🏗️ Architecture

BDDR operates in three stages:
1. **Privacy-preserving data acquisition** via federated distillation
2. **Backdoor Data Identification (BDI)** by testing per-batch loss-descent statistics
3. **Correct Label Restoration (CLR)** that attenuates backdoor activations while retaining benign features

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.5.1 (recommended)
- CUDA-compatible GPU (NVIDIA RTX 4090 recommended)

### Basic Usage

The BDDR defense module provides three main functionalities:

#### 1. Defense Mode
```bash
python Main.py bddr
```
Runs the complete BDDR defense pipeline including backdoor detection and label restoration.

#### 2. Evaluation Mode
```bash
python Main.py eu
```
Evaluates the model performance or data on clean accuracy (CA) and attack success rate (ASR).

#### 3. Image Splitting Mode
```bash
python Main.py si
```
Splits and processes images for defense module.

### Configuration

All parameters can be configured in the `config` file. 

## 📁 Project Structure

```
BDDR/
├── Main.py                 # Main entry point
├── config/                 # Configuration files
├── ncfm/                   # Federated learning distillation framework
├── ms-swift/               # LLM fine-tuning utilities
├── bddr/                   # Core BDDR implementation
│   ├── bdi.py             # Backdoor Data Identification
│   ├── clr.py             # Correct Label Restoration
│   └── federated.py       # Federated distillation
└── experiments/           # Experimental scripts and results
```

## 🔬 Advanced Usage

### Federated Learning Framework

For federated learning-based distillation:

```bash
cd ncfm/
# Follow instructions in ncfm/README.md
```

### LLM Fine-tuning

For large language model fine-tuning:

```bash
cd ms-swift/
# Follow instructions in ms-swift/README.md
```

## 📊 Experimental Results

BDDR achieves superior performance across multiple attack scenarios on CIFAR-10:

| Attack Type | BDDR (CA/ASR) | ABL (CA/ASR) | FP (CA/ASR) | MCR (CA/ASR) | NAD (CA/ASR) |
|-------------|---------------|--------------|-------------|--------------|--------------|
| BadNets     | 32.14%/0.02%  | 24.8%/51.31% | 22.1%/99.47% | 18.97%/52.86% | 20.7%/50.03% |
| Blend       | 24.47%/0.11%  | 29.4%/56.42% | 27.85%/99.62% | 22%/60.98% | 30%/10.76% |
| SIG         | 34.87%/0.09%  | 33.52%/0.18% | 29.99%/66.9% | 26.06%/0.1% | 30.67%/0.09% |
| Clean Label | 33.65%/0.06%  | 28.21%/0.21% | 26.05%/45.88% | 20.17%/1.44% | 22.54%/1.9% |

*CA: Clean Accuracy, ASR: Attack Success Rate*

## 🎯 Supported Datasets

- **Primary**: CIFAR-10 (32×32 color images, 10 categories)

## 🔧 Supported Attack Types

- **BadNets**: Traditional trigger-based attacks (1×1 white pixel trigger)
- **Blend**: Blended pattern attacks  
- **SIG**: Sinusoidal signal attacks
- **Clean Label (CL)**: Advanced stealthy attacks


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work is supported by:
- National Natural Science Foundation of China (62372244, 62572253)
- Natural Science Research Start-up Foundation of Recruiting Talents at NJUPT (NY224058)
- Jiangsu Youth Science and Technology Talent Support Project (JSTJ-2025-641)
- Open Project of State Key Laboratory for Novel Software Technology (KFKT2025B68)

---

⭐ If you find this project helpful, please consider giving it a star!