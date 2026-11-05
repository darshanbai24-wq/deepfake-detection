# Robust Deepfake Detection Using Deep Learning

A research-oriented Deep Learning project investigating the detection of manipulated facial videos, with an emphasis on generalization to unseen deepfake manipulations and datasets.

**Project Status:** `In Progress — Foundational Development and Training Experiments`

---

## 1. Project Overview

Deepfakes are AI-generated or manipulated images and videos that can convincingly imitate real people. As deepfake generation techniques become increasingly sophisticated, detecting manipulated media has become an important research challenge.

This project aims to develop a Deep Learning-based system capable of distinguishing authentic facial videos from manipulated ones.

Beyond basic classification, our research focuses on **generalization** — evaluating whether detection models can identify deepfakes generated using manipulation techniques or datasets that were not encountered during training.

## 2. Research Objectives

- Study existing Deep Learning approaches for deepfake detection.
- Implement and evaluate CNN-based detection models.
- Investigate Transformer-based vision architectures.
- Explore spatial and frequency-domain representations.
- Investigate temporal information across video frames.
- Evaluate cross-dataset generalization.
- Conduct ablation studies and error analysis.
- Develop an end-to-end deepfake detection application.

## 3. Proposed Methodology

The planned pipeline consists of the following stages:

1. **Dataset Collection:** Obtain publicly available datasets containing authentic and manipulated facial videos.
2. **Video Preprocessing:** Extract frames and detect and crop facial regions.
3. **Baseline Model:** Implement a CNN-based real/fake classifier.
4. **Architecture Comparison:** Investigate CNN and Vision Transformer approaches.
5. **Frequency Analysis:** Explore Fourier or Discrete Cosine Transform-based representations.
6. **Temporal Modeling:** Investigate relationships between consecutive video frames.
7. **Feature Fusion:** Experiment with combining spatial, frequency, and temporal information.
8. **Evaluation:** Perform cross-dataset testing, ablation studies, and error analysis.
9. **Application Development:** Build an interface for video submission and deepfake analysis.

The final architecture will be determined by literature findings and experimental results.

## 4. Development Progress

### Phase 1: Development Environment

- Configured Ubuntu development environment.
- Created an isolated Python virtual environment.
- Installed PyTorch, TorchVision, and TorchAudio.
- Configured CUDA-enabled PyTorch.
- Verified NVIDIA GPU availability.
- Successfully executed tensor computations on the GPU.

**Development environment:**
- OS: Ubuntu Linux
- Python: 3.10.12
- PyTorch: 2.14.1
- CUDA-enabled PyTorch build: cu130
- GPU: NVIDIA GeForce GTX 1650 Mobile (4 GB VRAM)

### Phase 2: PyTorch and CNN Fundamentals

Implemented introductory experiments to understand the building blocks of Deep Learning.

| File | Implementation |
|---|---|
| `day1_gpu_test.py` | GPU tensor creation and matrix multiplication |
| `tensor_test.py` | Tensor creation, shape, data type, and value inspection |
| `cnn_intro.py` | 2D convolution, edge detection, feature maps, and padding |
| `first_cnn.py` | Basic CNN architecture using PyTorch |

Topics explored:
- Tensor representation and dimensions.
- CPU and GPU computation.
- CUDA availability.
- Convolution, kernels, and feature maps.
- Stride and padding.
- ReLU activation and max pooling.
- CNN architecture and forward propagation.

### Phase 3: Neural Network Learning and Optimization

Implemented small experiments to understand how neural networks learn.

| File | Implementation |
|---|---|
| `loss_experiment.py` | Prediction scores and CrossEntropyLoss |
| `backprop_test.py` | Gradient calculation using PyTorch autograd |
| `gradient_descent.py` | Manual gradient descent and parameter updates |
| `adam_test.py` | Neural network optimization using Adam |

Concepts explored:
- Trainable weights and biases.
- Logits and loss functions.
- CrossEntropyLoss and MSELoss.
- Gradients and backpropagation.
- Gradient descent and learning rate.
- Epochs and convergence.
- Adam optimizer.
- Gradient clearing and reproducibility using random seeds.

### Phase 4: Dataset Handling and CNN Training

Developed a synthetic image classification experiment to understand the complete PyTorch training workflow.

| File | Implementation |
|---|---|
| `dataloader_test.py` | Dataset creation, batching, and DataLoader |
| `train_cnn.py` | CNN training, validation, and evaluation experiment |

**Synthetic dataset details:**
- Total images: 200.
- Image dimensions: 64 × 64.
- Channels: 3 (RGB).
- Class 0: Vertical stripe patterns.
- Class 1: Horizontal stripe patterns.
- Training samples: 160.
- Validation samples: 40.
- Batch size: 16.

**CNN architecture:**
- Conv2D: 3 → 16 channels.
- ReLU activation.
- Max pooling.
- Conv2D: 16 → 32 channels.
- ReLU activation.
- Max pooling.
- Flattening.
- Fully connected layer: 8192 → 128.
- ReLU activation.
- Output layer: 128 → 2.

**Training configuration:**
- Optimizer: Adam.
- Learning rate: 0.001.
- Loss function: CrossEntropyLoss.
- Training epochs: 20.
- Device: CUDA GPU when available.

**Observed experiment results:**
- Training loss converged to approximately zero.
- Validation loss converged to approximately zero.
- Training accuracy: 100% on the synthetic training set.

> **Important:** These results are from a deliberately simple synthetic pattern-classification experiment. They do not represent deepfake detection accuracy or performance on real-world video datasets. The experiment validates our training pipeline, not the effectiveness of a deepfake detection model.

## 5. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | TorchVision, OpenCV |
| GPU Acceleration | NVIDIA CUDA |
| Data Processing | NumPy, Pandas |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Application | Streamlit / FastAPI (planned) |
| Version Control | Git and GitHub |

## 6. Planned Evaluation Metrics

The following metrics will be used for evaluating actual deepfake detection models:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- Equal Error Rate (EER)
- Inference latency
- Model parameter count

Cross-dataset evaluation will be used to investigate generalization to unseen data distributions.

## 7. Planned Datasets

Potential benchmark datasets include:

- FaceForensics++
- Celeb-DF
- DeepFake Detection Challenge (DFDC)

Dataset selection and preprocessing strategies will be finalized after reviewing dataset availability, licensing, and research requirements.

## 8. Planned Project Structure

```text
deepfake-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── day1_gpu_test.py
├── tensor_test.py
├── cnn_intro.py
├── first_cnn.py
├── loss_experiment.py
├── backprop_test.py
├── gradient_descent.py
├── adam_test.py
├── dataloader_test.py
├── train_cnn.py
│
├── data/                  # Dataset storage (not committed)
├── src/                   # Reusable project modules
├── experiments/           # Experimental scripts
├── results/               # Evaluation outputs
├── app/                   # Application interface
└── docs/                  # Research documentation
```

The structure will evolve as the project moves from foundational experiments to the complete implementation.

## 9. Future Work

- Complete foundational Deep Learning concepts.
- Study classification metrics and model evaluation.
- Implement proper training, validation, and testing pipelines.
- Select and preprocess benchmark deepfake datasets.
- Establish a reproducible CNN baseline.
- Investigate pretrained CNN architectures and transfer learning.
- Compare CNN and Vision Transformer approaches.
- Investigate frequency-domain and temporal features.
- Experiment with feature fusion.
- Conduct cross-dataset evaluation and ablation studies.
- Perform error analysis and document findings.
- Develop a video-based inference pipeline and user interface.

## 10. Research Foundation

Initial research resources:

1. [DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection](https://arxiv.org/abs/2307.01426)
2. [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

Additional research papers and references will be documented as the literature survey progresses.

## 11. Project Team

Developed by a team of four undergraduate students pursuing Computer Science Engineering (Artificial Intelligence and Machine Learning).

**Institution:** BMS College of Engineering, Bengaluru

**Project Domain:** Deep Learning | Computer Vision | Digital Media Forensics

---

*This repository is actively under development. Proposed research components and the final architecture are subject to refinement based on literature review and experimental findings.*