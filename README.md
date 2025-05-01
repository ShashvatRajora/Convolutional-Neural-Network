# 🧠 Convolutional Neural Network (CNN) from Scratch in C

## 🔍 Overview

This project implements a **Convolutional Neural Network (CNN)** **entirely in pure C**, trained on grayscale facial images.  
The goal is to manually build and train a CNN from scratch — with no libraries, no frameworks — just raw C logic .

- Full Forward & Backpropagation ✅  
- Convolution, ReLU, Max Pooling, FC layers ✅  
- Model saving to file ✅  
- Dataset augmentation ✅  
- Realtime loss and accuracy tracking ✅  



---

## 🤖 CNN Architecture

Input: 32x32 grayscale image (1024 values) ↓ Conv1: 8 filters, 3x3 ↓ ReLU ↓ Conv2: 16 filters, 3x3 ↓ ReLU ↓ Max Pooling (2x2) ↓ Flatten (4096) ↓ Fully Connected Layer (10 outputs) ↓ Softmax + Cross Entropy Loss
---

## 🧾 Dataset

- **Images**: 80 grayscale face images  
- **Resolution**: 32 × 32 pixels  
- **Format**: `unsigned char image[480][1024]` in `data_face80.h`  
- **Classes**: 10 (digits 0–9 as labels)

---

### 💡 Augmented to Improve Accuracy

Since 80 images were too little to train a CNN, we **augmented each image into 6 versions**:
- ✅ Horizontal Flip
- ✅ Vertical Flip
- ✅ Brightness +30
- ✅ Contrast Stretch
- ✅ Salt & Pepper Noise

**Total images after augmentation: 480**

---

## ⚙️ Implementation Details

### 🔧 Folder Structure

| Path | Description |
|------|-------------|
| `src/` | Core CNN logic, loss functions, main training loop |
| `include/` | Header files (layer structs, prototypes, macros) |
| `data/` | `data_face80.h` with 480 images, `DatasetGenerator.c` |
| `logs/` | Saved weights (`trained_model.txt`), training logs |
| `cnn.exe` | Final compiled executable |

---

### 🔨 Files

| File | Purpose |
|------|---------|
| `main.c` | Training loop, accuracy printing, testing |
| `cnn.c / cnn.h` | Conv layer logic, pooling, flattening |
| `loss.c / loss.h` | Softmax, cross-entropy, derivatives |
| `activations.c / activations.h` | ReLU and its derivative |
| `utils.c / utils.h` | Flatten/unflatten, pooling backward |
| `DatasetGenerator.c` | Generates augmented image variants |
| `data_face80.h` | Holds the image dataset as unsigned char array |

---

## 🧪 Sample Output

Epoch 1: Average Loss = 2.3045 | Accuracy = 8.54% ... Epoch 50: Average Loss = 0.4831 | Accuracy = 46.88%

Training Finished! Model saved successfully to logs/trained_model.txt

Image 0 - True Label: 0 | Predicted: 0 (Confidence: 93.71%)

Image 1 - True Label: 1 | Predicted: 1 (Confidence: 85.23%)

Image 2 - True Label: 2 | Predicted: 2 (Confidence: 78.09%)

## 🧠 Challenges Faced

| ❌ Problem | ✅ Solution |
|-----------|-------------|
| Small dataset (80 images) | Augmented to 480 using flips, noise, brightness |
| No ML libraries | Manually implemented conv, pool, FC, softmax |
| ReLU, Softmax Backprop | Coded all derivatives and delta propagation |
| Buffer overflows | Carefully managed 3D arrays and memory allocation |
| Dataset format | Wrote own `normalize_dataset()` and label loader |
| Saving weights in C | Built text-based weight exporter (`save_model()`) |

---

## 📈 Accuracy Snapshot (480 Images, 50 Epochs)

| Epoch | Accuracy  | Loss     |
|-------|-----------|----------|
| 1     | 8.54%     | 2.3045   |
| 25    | 20.83%    | 2.2918   |
| 35    | 32.08%    | 2.1558   |
| 45    | 42.71%    | 0.8159   |
| **50** | **46.88%** | **0.4831** 

### 🧪 Final Testing (Post-Training):

All 10 test predictions were correct:

Image 0 - True Label: 0 | Predicted: 0 (Confidence: 93.71%)

Image 1 - True Label: 1 | Predicted: 1 (Confidence: 85.23%)

Image 2 - True Label: 2 | Predicted: 2 (Confidence: 78.09%)

Image 3 - True Label: 3 | Predicted: 3 (Confidence: 94.94%)

Image 4 - True Label: 4 | Predicted: 4 (Confidence: 96.30%)

Image 5 - True Label: 5 | Predicted: 5 (Confidence: 86.16%)

Image 6 - True Label: 6 | Predicted: 6 (Confidence: 88.04%)

Image 7 - True Label: 7 | Predicted: 7 (Confidence: 76.52%)

Image 8 - True Label: 8 | Predicted: 8 (Confidence: 84.04%)

Image 9 - True Label: 9 | Predicted: 9 (Confidence: 91.37%)

---
## Run using this 

```bash
gcc src/main.c src/cnn.c src/loss.c -o cnn -lm
cnn.exe
cnn.exe --load
