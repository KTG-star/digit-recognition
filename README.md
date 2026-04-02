---
title: Digit Recognition Task 5
emoji: 🔢
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: apache-2.0
---

# CSC 309: Handwritten Digit Recognition (Task 5)

This application is a web-based tool designed to recognize handwritten digits (0-9) using a **Convolutional Neural Network (CNN)**.

## 🚀 Features
* **Spatial Invariance:** Uses a CNN to recognize digits regardless of their orientation or stroke style.
* **Robust Preprocessing:** Automatically crops and centers the drawing to handle digits drawn in corners or at different scales.
* **Real-time Prediction:** Built with Gradio for an instant user experience.

## 🛠️ Tech Stack
* **Model:** TensorFlow/Keras (CNN)
* **Interface:** Gradio
* **Hosting:** Hugging Face Spaces

## 📖 How to Use
1. Draw any single digit (0-9) in the sketchpad.
2. The model will automatically process the image and display the top 3 most likely predictions with confidence scores.
