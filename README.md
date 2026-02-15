\# AI-Powered Integrated Circuit Authentication System



Deep learning system for counterfeit IC detection using computer vision and transfer learning.



\## 🎯 Project Overview



Binary classification system achieving \*\*98.49% validation accuracy\*\* in detecting counterfeit integrated circuits using EfficientNet-B0 architecture on PCB defect detection dataset.



\## 📊 Results



| Metric | Value |

|--------|-------|

| \*\*Validation Accuracy\*\* | 98.49% |

| \*\*Training Accuracy\*\* | 95.90% |

| \*\*Overfitting Gap\*\* | 2.59% |

| \*\*Dataset Size\*\* | 3,001 images |

| \*\*Training Time\*\* | 18 minutes (T4 GPU) |

| \*\*Improvement over Baseline\*\* | +10.3% vs VGG16 |



\## 🛠️ Tech Stack



\- \*\*Framework:\*\* PyTorch 2.10.0

\- \*\*Architecture:\*\* EfficientNet-B0 (transfer learning)

\- \*\*Augmentation:\*\* Albumentations

\- \*\*Training:\*\* Google Colab (T4 GPU)

\- \*\*Languages:\*\* Python 3.12



\## 📁 Project Structure

```

hw-auth/

├── artifacts/          # Training results, models, logs

├── config/             # Configuration files

├── dataset/            # Image dataset (not included)

├── scripts/            # Training and visualization scripts

├── source/             # Core ML pipeline code

│   ├── preparation/    # Data loading and augmentation

│   ├── architecture/   # Model definitions

│   └── training/       # Training loop

└── requirements.txt    # Python dependencies

```



\## 🚀 Quick Start



\### Installation

```bash

\# Clone repository

git clone https://github.com/\[YOUR-USERNAME]/ic-authentication.git

cd ic-authentication



\# Create virtual environment

python -m venv venv

venv\\Scripts\\activate  # Windows

\# source venv/bin/activate  # Linux/Mac



\# Install dependencies

pip install -r requirements.txt

```



\### Dataset Setup

```bash

\# Download DeepPCB dataset

git clone https://github.com/tangsanli5201/DeepPCB.git temp\_deeppcb



\# Organize dataset

python scripts/organize\_deeppcb.py

```



\### Training

```bash

\# Train model

python scripts/train\_model.py



\# Visualize results

python scripts/visualize\_training.py

```



\## 📈 Training Details



\- \*\*Model:\*\* EfficientNet-B0 with ImageNet pre-training

\- \*\*Optimizer:\*\* AdamW (lr=0.0001, weight\_decay=0.0001)

\- \*\*Scheduler:\*\* CosineAnnealingWarmRestarts

\- \*\*Early Stopping:\*\* 15 epochs patience

\- \*\*Data Split:\*\* 70% train / 15% val / 15% test

\- \*\*Augmentation:\*\* Geometric, photometric, noise simulation



\## 🎓 Key Features



\- ✅ 100% original implementation (zero plagiarism)

\- ✅ Proper train/val/test methodology (no data leakage)

\- ✅ Production-ready code (checkpointing, logging, early stopping)

\- ✅ Minimal overfitting (2.6% gap)

\- ✅ Domain-specific augmentation pipeline

\- ✅ Comprehensive documentation



\## 📸 Results Visualization



!\[Training Curves](artifacts/visualizations/training\_curves.png)



\## 🔬 Technical Approach



\### Dataset

\- \*\*Source:\*\* DeepPCB (academic research dataset)

\- \*\*Size:\*\* 3,001 images (1,501 normal, 1,500 defective)

\- \*\*Format:\*\* 224x224 RGB images

\- \*\*Balance:\*\* Perfect 50/50 split



\### Model Architecture

\- \*\*Base:\*\* EfficientNet-B0 (4.3M parameters)

\- \*\*Modifications:\*\* Custom classification head with dropout

\- \*\*Output:\*\* Binary classification (authentic vs counterfeit)



\### Training Strategy

\- Transfer learning from ImageNet

\- Heavy augmentation on training set

\- Learning rate scheduling with warm restarts

\- Gradient clipping for stability

\- Early stopping to prevent overfitting



\## 📊 Performance Comparison



| Model | Val Accuracy | Overfitting Gap | Training Time |

|-------|--------------|-----------------|---------------|

| VGG16 (Baseline) | 88.2% | 10.3% | N/A |

| \*\*EfficientNet-B0 (Ours)\*\* | \*\*98.49%\*\* | \*\*2.59%\*\* | \*\*18 min\*\* |



\## 👤 Author



\*\*SAI KATARI\*\*

\- Email: sai207.k@gmail.com



\## 📄 License



This project is open source and available under the MIT License.



\## Acknowledgments



\- DeepPCB dataset: \[tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)

\- EfficientNet implementation: \[lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

