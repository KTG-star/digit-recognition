# 🔢 Handwritten Digit Recognition
### CSC 309 Mini Project — Task 5

A full web application that recognises handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.

## 🌐 Live Demo
> Deployed on Streamlit Community Cloud — accessible via shareable link.

## 🧠 Concepts Used
- **Computer Vision** — processing 28×28 pixel images
- **Convolutional Neural Networks (CNN)** — feature extraction via Conv2D layers
- **Deep Learning** — training with backpropagation and Adam optimiser
- **Softmax classification** — outputting probabilities for each digit class

## 🏗️ Model Architecture
```
Input (28×28×1)
    → Conv2D (32 filters, 3×3, ReLU)
    → MaxPooling2D (2×2)
    → Conv2D (64 filters, 3×3, ReLU)
    → MaxPooling2D (2×2)
    → Flatten
    → Dropout (50%)
    → Dense (128, ReLU)
    → Dense (10, Softmax)
```

## 📊 Dataset
- **MNIST** — 70,000 handwritten digit images
- 60,000 training / 10,000 test
- Image size: 28×28 pixels, grayscale
- Test accuracy: ~99%

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- streamlit-drawable-canvas
- Matplotlib, NumPy, Pillow

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Author
Federal University of Technology, Owerri (FUTO)  
CSC 309 — Artificial Intelligence  
