# AI Art vs Human Art Classification

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3106/)

> A deep learning project to classify and distinguish AI-generated artwork from human-created artwork.

---

## 👥 Team Members

| Name | GitHub | Role |
|------|--------|------|
| Gechen Ma | [@Gechen989898](https://github.com/Gechen989898) | Team Lead / ML Engineer |
| Didier Peran Ganthier | [@didierganthier](https://github.com/didierganthier) | ML Engineer |
| Alexis Kipiani | [@Alex-gitacc](https://github.com/Alex-gitacc) | Data Engineer |
| Mame | [@kharitsama](https://github.com/kharitsama) | ML Engineer |

---

## 📋 Project Overview

With the rise of AI image generation tools (DALL-E, Midjourney, Stable Diffusion), distinguishing between AI-generated and human-created art has become increasingly challenging. This project aims to build a robust classification model that can accurately identify the origin of artwork.

### Objectives
- Build and compare multiple deep learning architectures
- Achieve high accuracy in classifying AI vs Human art
- Deploy a functional API for real-time predictions
- Create an interactive demo interface

---

## 📊 Dataset

**Tiny GenImage** - A lightweight version of the GenImage dataset, perfect for training models on modern diffusion-generated images.

| Dataset | Description | Link |
|---------|-------------|------|
| **Tiny GenImage** | Compact dataset featuring AI-generated images from modern diffusion models (Stable Diffusion, Midjourney, etc.) vs real images | [Kaggle](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) |

---

## 🏗️ Project Architecture

```
AI_Art_vs_Human_Art/
├── README.md
├── Makefile
├── requirements.txt
├── setup.py
├── .env.sample
├── .gitignore
├── raw_data/
│   ├── train/
│   │   ├── ai/
│   │   └── human/
│   └── test/
│       ├── ai/
│       └── human/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── ai_art_classifier/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_baseline.py
│   │   ├── resnet.py
│   │   ├── efficientnet.py
│   │   └── vision_transformer.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── api/
│       ├── __init__.py
│       └── fast_api.py
├── models/
│   └── (saved model weights)
└── tests/
    └── (unit tests)
```

---

## 🧠 Models to Implement

1. **CNN Baseline** - Custom Convolutional Neural Network
2. **ResNet50** - Transfer Learning with ResNet
3. **EfficientNetB0** - Transfer Learning with EfficientNet
4. **Vision Transformer (ViT)** - Transformer-based approach

---

## 📅 Project Timeline (6 Weeks)

### Week 1: Project Setup & Data Collection
| Task | Assignee | Status |
|------|----------|--------|
| Set up GitHub repository & branch protection | Gechen | ⬜ |
| Create project structure & Makefile | Didier | ⬜ |
| Download and organize Tiny GenImage dataset | Alexis | ⬜ |
| Set up virtual environment & requirements.txt | Didier | ⬜ |
| Create Trello board with all tasks | Mame | ⬜ |

### Week 2: Data Exploration & Preprocessing
| Task | Assignee | Status |
|------|----------|--------|
| Exploratory Data Analysis (EDA) notebook | Alexis | ⬜ |
| Data visualization (class distribution, samples) | Alexis | ⬜ |
| Implement data augmentation pipeline | Gechen | ⬜ |
| Create data loader classes | Didier | ⬜ |
| Implement train/val/test split logic | Mame | ⬜ |
| Document data preprocessing steps | All | ⬜ |

### Week 3: Baseline Model Development
| Task | Assignee | Status |
|------|----------|--------|
| Implement CNN baseline model | Gechen | ⬜ |
| Implement ResNet transfer learning | Didier | ⬜ |
| Create training pipeline with callbacks | Mame | ⬜ |
| Set up experiment tracking (MLflow/W&B) | Alexis | ⬜ |
| Train and evaluate CNN baseline | Gechen | ⬜ |
| Train and evaluate ResNet model | Didier | ⬜ |

### Week 4: Advanced Models & Optimization
| Task | Assignee | Status |
|------|----------|--------|
| Implement EfficientNet model | Mame | ⬜ |
| Implement Vision Transformer (ViT) | Alexis | ⬜ |
| Hyperparameter tuning for best models | Gechen | ⬜ |
| Cross-validation implementation | Didier | ⬜ |
| Model comparison analysis | All | ⬜ |
| Implement ensemble method (optional) | Gechen | ⬜ |

### Week 5: API Development & Deployment
| Task | Assignee | Status |
|------|----------|--------|
| Build FastAPI prediction endpoint | Didier | ⬜ |
| Create Docker container | Gechen | ⬜ |
| Implement image upload functionality | Mame | ⬜ |
| Deploy API to cloud (GCP/AWS) | Alexis | ⬜ |
| Write API documentation | Didier | ⬜ |
| Load testing & optimization | Gechen | ⬜ |

### Week 6: Demo, Testing & Presentation
| Task | Assignee | Status |
|------|----------|--------|
| Build Streamlit/Gradio demo interface | Mame | ⬜ |
| Write unit tests | Alexis | ⬜ |
| Final model evaluation on test set | Gechen | ⬜ |
| Prepare presentation slides | All | ⬜ |
| Record demo video | Didier | ⬜ |
| Final code review & documentation | All | ⬜ |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10.6
- pyenv (recommended)

### Installation

```bash
# Clone the repository
git clone git@github.com:Gechen989898/AI_Art_vs_Human_Art.git
cd AI_Art_vs_Human_Art

# Create and activate virtual environment
pyenv virtualenv 3.10.6 AI_Art_vs_Human_Art
pyenv activate AI_Art_vs_Human_Art

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Using Kaggle CLI
kaggle datasets download -d yangsangtai/tiny-genimage
unzip tiny-genimage.zip -d raw_data/
```

### Training

```bash
# Train baseline CNN
make train_cnn

# Train ResNet
make train_resnet

# Train all models
make train_all
```

### Running the API

```bash
# Start FastAPI server
make run_api
```

---

## 📈 Expected Results

| Model | Target Accuracy | Training Time |
|-------|-----------------|---------------|
| CNN Baseline | ~85% | ~30 min |
| ResNet50 | ~92% | ~1 hour |
| EfficientNetB0 | ~94% | ~1 hour |
| ViT | ~95% | ~2 hours |

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / PyTorch
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: Matplotlib, Seaborn
- **API**: FastAPI
- **Deployment**: Docker, GCP/AWS
- **Demo**: Streamlit / Gradio
- **Experiment Tracking**: MLflow / Weights & Biases

---

## 📚 Resources

- [CIFAKE Paper](https://arxiv.org/abs/2303.14126)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

## 📝 License

This project is for educational purposes.

---

## 🤝 Contributing

1. Create a feature branch from `master`
2. Make your changes
3. Submit a Pull Request
4. Request review from at least one team member

**Branch naming convention**: `feature/<your-name>/<feature-description>`

---

*Project started: February 2026*
