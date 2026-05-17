# Part 2 – Computer Vision Problem Formulation and CNN Prototype

## Overview

This project formulates a computer vision problem from an image dataset of car surface defects and builds a CNN-inspired prototype to classify images into four defect categories.

Dataset source: [Google Drive – Shared Assignment Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)  
> Download the Part 2 dataset from the link above. Do not upload the dataset to this repository.  
> Place the `images/` folder in the same directory as `notebook.ipynb` before running.

---

## Task 1: Problem Identification

Problem Type: Image Classification**

Each image in the dataset shows a car panel surface and belongs to exactly one of four categories: `dent`, `normal`, `scratch`, `stain`. 
The task is to assign the correct category label to each image — this is a multi-class image classification problem.

| Problem Type | Reason Not Chosen |
|---|---|
| Object Detection | Would require locating defects with bounding boxes within larger images. Not needed here — each image *is* one defect. |
| Semantic Segmentation | Would require pixel-level labelling. Not needed — the whole image belongs to one class. |
| Instance Segmentation | Extension of segmentation for individual instances. Not applicable. |
| **Image Classification** ✅ | Each image belongs to exactly one class. Task is to predict that label. |

---

## Task 6: CNN Concept Explanation

### What is convolution?
Convolution is the core operation in a CNN. A small filter (e.g. 3×3) slides across the image, computing a dot product at each position. 
This produces a feature map that highlights where specific patterns — edges, corners, textures — appear in the image. 
Different filters learn to detect different patterns automatically during training.

In this project, HOG (Histogram of Oriented Gradients) acts as the convolution-equivalent step as it applies fixed gradient filters across image patches to extract edge direction information, which is exactly what the early layers of a CNN learn to do.

### Why is pooling used?
Pooling (typically max pooling) reduces the spatial size of feature maps by keeping only the strongest activation within each small region. This achieves two things:
1. Dimensionality reduction — fewer parameters, faster computation.
2. Spatial invariance — the model becomes less sensitive to exact position of a feature (a scratch slightly to the left or right is still a scratch).

HOG's block normalisation serves the same role in this project.

### Why is ReLU commonly used in CNNs?
ReLU (`max(0, z)`) is the standard hidden-layer activation for three reasons:
1. No vanishing gradient — unlike sigmoid or tanh, it does not saturate for positive inputs, allowing gradients to flow freely through deep networks.
2. Sparsity — neurons with negative inputs output zero, making the network sparse and efficient.
3. Computational simplicity — just a threshold at zero, extremely fast to compute.

### Why are CNNs better than regular feed-forward networks for image data?
A standard feed-forward network treats each pixel as an independent input feature. A 64×64 RGB image has 12,288 inputs — a dense layer connecting to 256 neurons already has 3 million parameters. More importantly, it loses all spatial structure: it does not know that neighbouring pixels are related.

CNNs solve both problems:
- Weight sharing — the same filter is applied across the entire image, massively reducing parameters.
- Local connectivity — each neuron only sees a small patch, preserving spatial relationships.
- Translation invariance — a cat in the top-left and bottom-right of an image activate the same filters, so the model generalises across positions.

---

## Task 7: Business Use Case Mapping

### Domain: Manufacturing – Automotive Quality Control

Problem: Manually inspecting every car body panel for surface defects (dents, scratches, stains) is slow, expensive, and inconsistent. A single production line can produce hundreds of vehicles per day.

Solution: A CNN-based vision system mounted on the production line captures images of each panel and classifies defects in real time.

Benefits:
- Speed: Automated inspection takes milliseconds per panel vs. seconds for a human.
- Consistency: The model applies the same criteria every time, eliminating inspector fatigue.
- Cost reduction: Fewer panels escape the factory with undetected defects, reducing warranty claims.
- Data logging: Every defect is recorded, enabling process engineers to identify root causes (e.g. a particular press tool causing dents).

Example deployment: BMW, Toyota, and Tesla already use computer vision systems for paint and surface defect detection on production lines. This model prototype demonstrates the core classification capability required for such a system.

---

## Dataset

| Property | Value |
|---|---|
| Classes | `dent`, `normal`, `scratch`, `stain` |
| Images per class | 120 |
| Total images | 480 |
| Image size | 96×96 px (resized to 64×64 for training) |
| Colour mode | RGB |
| Class balance | Perfectly balanced (no imbalance) |

---

## Approach

### Preprocessing
- Resized all images to 64×64 pixels
- Applied data augmentation (horizontal flip, vertical flip, brightness boost) → 4× dataset size
- Extracted HOG (Histogram of Oriented Gradients) features — 1,568 dimensions per image
- Stratified 80/20 train/test split

### Model (CNN-equivalent pipeline)
- Feature extraction: HOG — equivalent to the convolution + pooling layers of a CNN
- Classifier: Multi-Layer Perceptron — Dense(256, ReLU) → Dense(128, ReLU) → Output(4, Softmax)
- Loss:** Cross-entropy
- Optimiser: Adam (adaptive learning rate)

### Why not use TensorFlow/Keras?
TensorFlow is not available in this environment. The HOG + MLP pipeline replicates the computational intent of a shallow CNN and achieves strong, interpretable results.

---

## Results

| Config | Architecture | Activation | LR | Train Acc | Test Acc |
|---|---|---|---|---|---|
| A – Baseline | (128,) | ReLU | 0.001 | 98.4% | 87.0% |
| **B – Deeper** | **(256, 128)** | **ReLU** | **0.001** | **97.7%** | **90.6%** |
| C – Low LR | (128,) | ReLU | 0.0001 | 86.0% | 81.0% |
| D – Tanh | (128,) | Tanh | 0.001 | 98.5% | 88.0% |
| E – Wider+Deep | (256,128,64) | ReLU | 0.001 | 98.6% | 89.6% |

Best model: Config B – Deeper (Test Accuracy = **90.6%**)

### Observations
- Config B (two layers: 256→128) outperforms the single-layer baseline, suggesting the dataset
  benefits from hierarchical feature combination.
- Config C (very low lr=0.0001) underperforms — 200 iterations is insufficient for convergence
  at that learning rate.
- `normal` class achieves near-perfect precision and recall — clean panels are visually distinct.
- `stain` is the hardest class — it is occasionally confused with `dent` and `scratch` due to
  similar surface discolouration patterns.

---

## Repository Structure

```
part-2-cnn-computer-vision/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── sample_images.png
    ├── accuracy_loss_curves.png
    ├── confusion_matrix.png
    └── model_comparison_table.png
```

## How to Run

```bash
pip install -r requirements.txt
```

Place the `images/` folder (downloaded from the Google Drive link above) in this directory, then:

```bash
jupyter notebook notebook.ipynb
```

Run all cells from top to bottom.
