# Distracted Driver Detection Project

This project focuses on **classifying driver behaviors** from dashboard camera images using a **Convolutional Neural Network (CNN)**. It leverages the **State Farm Distracted Driver Detection** dataset to identify 10 classes of driver activity, including safe driving, texting, talking, and more.

The pipeline combines **image preprocessing**, **data augmentation**, **deep CNN modeling**, and **performance visualization** using TensorFlow and Keras.

---

## Project Overview

This project allows you to:

- Load and process driver image data using Keras ImageDataGenerator
- Apply image augmentation techniques for generalization
- Train a deep CNN model with over 27M trainable parameters
- Evaluate model performance using metrics like accuracy, precision, recall, and AUC
- Generate predictions and output a CSV for submission
- Visualize loss curves, accuracy trends, confusion matrix, and classification report

---

## Features

- Automatic data preprocessing and rescaling
- CNN architecture with dropout regularization
- EarlyStopping based on validation accuracy
- Data augmentation (flip, zoom, shear)
- Output CSV with class probabilities for test images
- Visualization:
  - Accuracy and loss curves
  - Confusion matrix
  - Classification report (precision, recall, F1-score)

---

## Technologies Used

### Core Libraries & Tools

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- scikit-learn
- Jupyter Notebooks

---

## Model Architecture

- `Conv2D (128 filters)` + `MaxPooling`
- `Conv2D (64 filters)` + `MaxPooling`
- `Conv2D (32 filters)` + `MaxPooling`
- `Flatten` Layer
- Dense Layers:
  - 1024 units + Dropout
  - 1024 units + Dropout
  - 256 units + Dropout
  - 10-unit Softmax output

**Loss Function**: Categorical Crossentropy  
**Optimizer**: Adam  
**Epochs**: 10  
**Batch Size**: 32  
**EarlyStopping**: patience=5, delta=0.005

---

## Results

| Metric       | Value          |
|--------------|----------------|
| Accuracy     | 97.5%          |
| AUC          | 0.9986         |
| Precision    | 0.9788         |
| Recall       | 0.9724         |
| Loss         | 0.0950         |

> Achieved using validation set on 4481 images (split from ~22k total)

---

## Dataset

- **Source**: State Farm Distracted Driver Detection  
- **Size**: ~22,000 training images across 10 categories  
- **Labels**:
  - c0: Safe driving  
  - c1: Texting - right  
  - c2: Talking on the phone - right  
  - c3: Texting - left  
  - c4: Talking on the phone - left  
  - c5: Operating the radio  
  - c6: Drinking  
  - c7: Reaching behind  
  - c8: Hair and makeup  
  - c9: Talking to passenger

---

## APIs / Integrations

- `ImageDataGenerator` – Data loading and augmentation  
- `TensorFlow/Keras` – CNN modeling  
- `sklearn` – Accuracy, Confusion Matrix, Classification Report  
- `Matplotlib & Seaborn` – Plotting and heatmaps

---

## Author

**Krish Dua**  
[Portfolio Website](https://krishdua.vercel.app) 
[LinkedIn Profile](https://www.linkedin.com/in/krish-dua-9202a4272/)

---
