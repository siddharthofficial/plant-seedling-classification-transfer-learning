# Plant Seedling Classification using Transfer Learning (PyTorch)

## Overview

An end-to-end computer vision system for plant species classification using transfer learning, Grad-CAM explainability, and an interactive Streamlit deployment.

The objective is to accurately classify plant seedlings into 12 species using pretrained convolutional neural networks and evaluate the effectiveness of different transfer learning architectures.

## Key Features

* Multi-class plant seedling classification
* Transfer learning using pretrained CNN architectures
* Advanced data augmentation pipeline
* Model benchmarking across VGG16, ResNet50, and InceptionV3
* Grad-CAM explainability
* UMAP feature space visualization
* Streamlit deployment for real-time inference
* Comprehensive evaluation using Accuracy, Precision, Recall, and F1-Score

---

## Dataset

**Plant Seedlings Classification Dataset**

* 12 plant species
* 4,750 training images
* Multi-class image classification problem

Classes include:

* Black-grass
* Charlock
* Cleavers
* Common Chickweed
* Common wheat
* Fat Hen
* Loose Silky-bent
* Maize
* Scentless Mayweed
* Shepherds Purse
* Small-flowered Cranesbill
* Sugar beet

---
  

## Project Pipeline

### 1. Exploratory Data Analysis

Performed dataset inspection to understand:

* Class distribution
* Dataset balance
* Image resolution distribution
* Sample visualization per class

Key findings:

* Dataset contains varying image resolutions
* Images range from low-resolution to high-resolution samples
* Moderate class imbalance exists across species

---

### 2. Data Preprocessing & Augmentation

To improve model generalization and reduce overfitting, an extensive image augmentation pipeline was applied during training.

### Training Transformations

* Random Horizontal Flip
* Random Rotation
* Random Resized Crop
* Color Jitter
* ImageNet Normalization
* Tensor Conversion

These augmentations expose the model to variations in orientation, scale, lighting conditions, and viewpoint, helping improve robustness on unseen plant images.

### Validation Transformations

For evaluation consistency, validation images were processed using deterministic transformations:

* Resize
* Center Crop
* Tensor Conversion
* ImageNet Normalization

### Input Resolution

| Model       | Input Size |
| ----------- | ---------- |
| VGG16       | 224 × 224  |
| ResNet50    | 224 × 224  |
| InceptionV3 | 299 × 299  |



---

### 3. Transfer Learning Architectures

Three pretrained CNN architectures were benchmarked:

| Model       | Backbone            |
| ----------- | ------------------- |
| VGG16       | ImageNet Pretrained |
| ResNet50    | ImageNet Pretrained |
| InceptionV3 | ImageNet Pretrained |

Fine-tuning strategy:

* Frozen pretrained feature extractor
* Replaced classification head
* Trained custom classifier layers on plant dataset

---

### 4. Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 1e-4             |
| Epochs        | 25               |
| Batch Size    | 32               |
| Loss Function | CrossEntropyLoss |
| Random Seed   | 42               |

---

## Model Performance

### 5. Benchmark Results

| Model       | Accuracy | Weighted F1 Score |
| ----------- | -------- | ----------------- |
| VGG16       | 86.32%   | 86.07%            |
| ResNet50    | 83.26%   | 83.09%            |
| InceptionV3 | 81.37%   | 81.11%            |

### Best Model

**VGG16 achieved the highest performance**

* Accuracy: 86.32%
* Weighted F1 Score: 86.07%

---

## Explainable AI

### 6. Grad-CAM Visualization

Implemented Grad-CAM to interpret model predictions and visualize regions responsible for classification decisions.

Benefits:

* Improved model transparency
* Better debugging of prediction behavior
* Increased confidence in learned visual features

---

## Representation Learning Analysis

### 7. UMAP Feature Embeddings

Extracted deep feature representations from the trained model and projected them into a lower-dimensional space using UMAP.

This helps visualize:

* Class separability
* Feature clustering
* Learned representation quality

---

### 8. Web Application

The trained ResNet50 model is deployed using Streamlit.

Features:

- Upload plant images
- Real-time species prediction
- Prediction confidence scores
- Grad-CAM visual explanations
- Interactive user interface

The deployment pipeline loads the trained model, performs inference, and generates attention heatmaps showing which image regions influenced the prediction.


## Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Grad-CAM
* UMAP



## Repository Structure

```text
plant-seedling-classification-transfer-learning/
│
|──app.py
|──classnames.json
|── architecture.png
|── requirements.txt
|
├──  plant_seedling_classifier.ipynb 
|
├── results/
│   ├── confusion_matrix_inception.png
│   ├── confusion_matrix_resnet.png
│   ├── confusion_matrix_vgg16.png
│   ├── grad_cam.png
│   ├── inception_loss_accuracy_curve.png
│   ├── resnet_loss_accuracy_curve.png
│   ├── vgg_loss_accuracy_curve.png
│   ├── model_results.png
│   ├── umap_embedding.png
│   └── streamlit_upload_page.png
│   ├── streamlit_gradcam_prediction.png
|
├── README.md

```

---

## Key Learnings

* Practical implementation of transfer learning workflows
* CNN architecture benchmarking
* Model interpretability using Grad-CAM
* Deep feature visualization using UMAP
* End-to-end computer vision pipeline development in PyTorch

---

## Future Improvements

* Hyperparameter optimization
* K-Fold cross-validation
* EfficientNet and ConvNeXt benchmarking
* Test-time augmentation
* Model deployment using FastAPI
* ONNX/TorchScript optimization for production inference

## Author 
Siddharth Jain

```
```
