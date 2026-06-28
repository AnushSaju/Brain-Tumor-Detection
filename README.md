# 🧠 Brain Tumor Detection using Deep Learning

An AI-powered deep learning system for automated brain tumor detection from MRI images. The project leverages a custom Convolutional Neural Network (CNN) to classify MRI scans and includes model training, inference, Grad-CAM visualization, and a backend API for serving predictions.

---

## 📌 Project Overview

Brain tumors are one of the most critical neurological disorders, where early diagnosis can significantly improve treatment outcomes. This project aims to automate brain tumor detection using deep learning techniques applied to MRI images.

The repository includes:

- Model training pipeline
- Model evaluation
- Inference on MRI scans
- Grad-CAM visualization for explainability
- Backend API for prediction

---

## ✨ Features

- 🧠 Brain Tumor Classification from MRI Images
- 🤖 Custom CNN Architecture
- 📊 Model Training & Evaluation
- 🔥 Grad-CAM Explainability
- 🚀 Backend API for Predictions
- 📁 Organized Dataset & Output Management

---

## 🛠️ Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Albumentations
- Pillow

---

## 📂 Project Structure

```text
Brain-Tumor-Detection/
│
├── backend/          # Backend API
├── data/             # Dataset
├── model/            # Model architecture and utilities
├── outputs/          # Trained models and outputs
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/AnushSaju/Brain-Tumor-Detection.git
```

Move into the project directory:

```bash
cd Brain-Tumor-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start the backend server:

```bash
uvicorn backend.main:app --reload
```

---

## 🔮 Future Improvements

- Improve classification accuracy
- Deploy the application
- Support additional tumor classes
- Improve model explainability
- Enhance the frontend interface

---

## 👥 Contributors

Developed as a collaborative academic project.

---

## 📄 License

This project is intended for educational and research purposes.