# Part 2 – Computer Vision Problem Formulation and CNN Prototype

## Overview

This project formulates a computer vision problem from an image dataset of car surface defects
and builds a CNN prototype using TensorFlow/Keras to classify images into four defect categories.

**Dataset source:** [Google Drive – Shared Assignment Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)  
> Download the Part 2 dataset from the link above. Do not upload the dataset to this repository.  
> Place the `images/` folder in the same directory as `notebook.ipynb` before running.

---

## Task 1: Problem Identification

### Selected Problem Type: Image Classification

The given dataset represents an image classification problem.
This dataset contains surface images of car panels, each belonging to exactly one of four defect categories. The goal is to assign the correct category label to each image.

---

### Why Image Classification?

This problem type is appropriate because:

- Each image belongs to one predefined category/class.
- The model predicts a single label for each image.
- Images are organized into class folders.
- The dataset does not contain bounding boxes or segmentation masks.

---

### Why Other Problem Types Do Not Apply

| Problem Type | Reason |
|---|---|
| Object Detection | Requires object location coordinates/bounding boxes |
| Semantic Segmentation | Requires pixel-level annotations |
| Instance Segmentation | Requires separate masks for each object |

Therefore, image classification is the correct computer vision formulation.

---

# Task 2: Dataset Exploration

## Dataset Analysis

The dataset was explored using Python utilities and visualization libraries.

### Analysis Performed

- Total number of classes
- Number of images per class
- Sample image visualization
- Image dimensions
- Dataset imbalance detection

---

## Dataset Statistics

| Class Name | Number of Images |
|---|---|
| Replace with actual class names | Replace with actual counts |

---

## Observations

- Images belong to multiple categories.
- Some classes may contain more images than others.
- Slight imbalance may exist.
- Image dimensions vary across samples.
- Images are resized during preprocessing to ensure consistency.

---

# Task 3: Image Preprocessing

## Preprocessing Steps

### 1. Image Resizing

All images were resized to:

```python
IMAGE_SIZE = (128, 128)
```

This ensures that every image has the same dimensions before being passed into the CNN.

---

### 2. Pixel Normalization

Pixel values were normalized from:

```python
0–255 → 0–1
```

This improves training stability and convergence.

---

### 3. Dataset Splitting

The dataset was divided into:

| Dataset | Percentage |
|---|---|
| Training Set | 70% |
| Validation Set | 15% |
| Testing Set | 15% |

---

### 4. Data Augmentation

To improve generalization and reduce overfitting, augmentation techniques were applied:

- Rotation
- Horizontal flipping
- Zooming
- Width shifting
- Height shifting

---

# Task 4: CNN Model Creation

## CNN Architecture

The CNN model contains:

- Convolution layers
- ReLU activation functions
- MaxPooling layers
- Flatten layer
- Dense layers
- Dropout layers
- Softmax output layer

---

## Model Architecture

```python
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(len(classes), activation='softmax')
])
```

---

# Task 5: Model Training and Evaluation

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | sparse_categorical_crossentropy |
| Epochs | 15 |
| Batch Size | 32 |

---

## Callbacks Used

### EarlyStopping

Stops training when validation loss stops improving.

### ModelCheckpoint

Automatically saves the best-performing model.

---

## Evaluation Metrics

The following metrics were used:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Test Accuracy
- Test Loss
- Confusion Matrix
- Classification Report

---

## Accuracy and Loss Curves

These graphs show:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

---

## Confusion Matrix

The confusion matrix helps evaluate:

- Correct predictions
- Incorrect predictions
- Class-wise performance

---

## Sample Predictions

The output image visualizes:

- Test image
- Predicted label
- Actual label

---

# Task 6: CNN Concept Explanation

## What is Convolution?

Convolution is a mathematical operation where a small filter/kernel slides across the image to detect important visual features such as:

- Edges
- Textures
- Shapes
- Patterns

CNNs automatically learn these filters during training.

---

## Why is Pooling Used?

Pooling reduces the spatial dimensions of feature maps.

Benefits:

- Reduces computation
- Reduces overfitting
- Keeps important features
- Makes training faster

MaxPooling selects the most important value from a region.

---

## Why is ReLU Commonly Used?

ReLU (Rectified Linear Unit) introduces non-linearity into the network.

genui{"math_block_widget_always_prefetch_v2":{"content":"f(x)=\\max(0,x)"}}

Advantages:

- Faster training
- Computational efficiency
- Helps reduce vanishing gradient problems

---

## Why are CNNs Better Than Regular Feed-Forward Networks for Images?

CNNs are better because they:

- Preserve spatial relationships in images
- Automatically learn visual features
- Require fewer parameters
- Handle image translation effectively
- Are more efficient for image processing tasks

Regular feed-forward networks flatten images completely and lose spatial structure.

---

## Task 7: Business Use Case Mapping

### Domain: Manufacturing – Automotive Quality Control

Problem:
Manually inspecting every car body panel for surface defects (dents, scratches, stains) is slow,
expensive, and inconsistent. A single production line can produce hundreds of vehicles per shift,
and human inspectors are subject to fatigue and varying standards.

Solution: 
A CNN-based vision system mounted at inspection stations on the production line captures images
of each panel section and classifies defects in real time — flagging panels for rework before
they proceed further down the line.

How this model maps to the use case:

| Model Class | Real-World Meaning |
|---|---|
| `dent` | Panel deformation from press or handling damage |
| `scratch` | Surface score marks from tooling or transport |
| `stain` | Paint contamination or chemical marks |
| `normal` | Panel passes quality standard — cleared for assembly |

Benefits:
- Speed: Automated inspection takes milliseconds per panel vs. seconds for a human inspector.
- Consistency: The model applies identical criteria on every panel, eliminating inspector fatigue.
- Traceability: Every classification is logged, enabling engineers to identify root causes
  (e.g. a specific press tool causing recurring dents in the same location).
- Cost reduction: Fewer defective panels escape to the next stage, reducing costly rework
  downstream and warranty claims post-sale.

Industry context: 
Automotive manufacturers including BMW, Toyota, and Tesla already deploy computer vision systems
for paint and surface defect detection on production lines. This CNN prototype demonstrates
the core classification capability required for such a system.

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
    └── confusion_matrix.png
```
---

## Technologies Used
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Result
This project successfully demonstrates how Convolutional Neural Networks (CNNs) can solve image classification problems.
The project includes:
- Dataset preprocessing
- CNN architecture creation
- Model training
- Evaluation
- Visualization
- Concept explanation
- Business use case mapping

The final model successfully learns visual patterns from images using convolution, pooling, activation functions, and dense layers.
