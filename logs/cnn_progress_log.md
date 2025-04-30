# CNN in Pure C for Facial Recognition - Progress Log

---

## ✅ Project Overview

- **Goal**: Implement a basic 3-layer CNN from scratch in C.
- **Dataset**: Originally 80 grayscale images (32x32), expanded to 480 using data augmentation.
- **Approach**: No external libraries. All operations (convolution, activation, loss, training) implemented manually.

---

## ✅ CNN Architecture

Input (32x32 grayscale) ↓ Conv1: 8 filters (3x3) ↓ ReLU ↓ Conv2: 16 filters (3x3) ↓ ReLU ↓ Max Pooling (2x2) ↓ Flatten (4096 values) ↓ Fully Connected (FC) layer → 10 outputs ↓ Softmax → Classification

---

## ✅ Progress Timeline

### ✅ 1. Dataset Loading & Normalization
- Loaded raw image array `unsigned char image[80][1024]`.
- Converted pixel values to float `[0, 1]` using `normalize_dataset()`.

---

### ✅ 2. Manual Forward Pass
Manually implemented all building blocks:
- Convolution Layers (Conv1 and Conv2)
- ReLU Activation
- Max Pooling (2x2)
- Flattening
- Fully Connected Layer
- Softmax Function

Verified all forward steps with sample inputs.

---

### ✅ 3. Loss Function
- Implemented **Cross Entropy Loss** + Softmax Derivative.
- Confirmed working by observing consistent drop in loss over epochs.

---

### ✅ 4. Full Backpropagation 🔁
Built complete manual backpropagation for:
- Fully Connected Layer
- Conv2 Layer
- Conv1 Layer
- Max Pooling Layer
- ReLU Derivatives
- Softmax + Cross Entropy derivative combined for efficient gradients

Now supports **end-to-end learning**.

---

### ✅ 5. Dataset Expansion 🚀
Used data augmentation to create a synthetic dataset of 480 images:
- Original (80)
- Horizontally Flipped
- Vertically Flipped
- Brightness Increased (+30)
- Contrast Enhanced (scaled about mean)
- Salt & Pepper Noise

```c
unsigned char image[480][1024];  // after expansion

Custom made into 6 variants of the images .