# 🚦 Traffic Sign Classification using CNN | TensorFlow | OpenCV

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Project Overview

Traffic Sign Classification is a Deep Learning-based computer vision project that automatically identifies traffic signs from images using a Convolutional Neural Network (CNN). The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset and uses TensorFlow/Keras for model development and OpenCV for image preprocessing and prediction.

This project demonstrates how CNNs can be applied in autonomous driving systems and Advanced Driver Assistance Systems (ADAS) for real-time traffic sign recognition.

---

## 🎯 Features

- Train a CNN model for traffic sign classification
- Image preprocessing using OpenCV
- Data augmentation
- TensorFlow/Keras implementation
- Real-time image prediction
- Save and load trained model
- High classification accuracy
- Easy to customize for new datasets

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📂 Dataset

**Dataset Used:** German Traffic Sign Recognition Benchmark (GTSRB)

- 43 Traffic Sign Classes
- More than 50,000 images
- Images of different lighting conditions and viewpoints

Download Dataset:
https://benchmark.ini.rub.de/gtsrb_news.html

---

## 📁 Project Structure

```
Traffic-Sign-Classification/
│
├── dataset/
│   ├── Train/
│   ├── Test/
│   └── Meta/
│
├── model/
│   └── traffic_sign_model.h5
│
├── images/
│
├── train.py
├── predict.py
├── preprocess.py
├── model.py
├── requirements.txt
├── README.md
└── labels.csv
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Traffic-Sign-Classification.git

cd Traffic-Sign-Classification
```

### Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```
tensorflow
opencv-python
numpy
pandas
matplotlib
scikit-learn
Pillow
```

or install manually

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn pillow
```

---

## 🚀 Training the Model

Run

```bash
python train.py
```

The model will:

- Load dataset
- Preprocess images
- Train CNN
- Validate accuracy
- Save trained model (.h5)

---

## 🔍 Predicting Traffic Signs

```bash
python predict.py
```

Example Output

```
Predicted Class:
Speed Limit (60 km/h)

Confidence:
99.21%
```

---

## 🧠 CNN Architecture

The CNN model consists of:

- Input Layer (30×30×3)
- Convolution Layer
- ReLU Activation
- Max Pooling
- Convolution Layer
- ReLU Activation
- Max Pooling
- Flatten Layer
- Dense Layer
- Dropout Layer
- Output Layer (Softmax)

---

## 🖼️ Image Preprocessing

Before training, each image undergoes:

- Resize to 30×30 pixels
- Color conversion (RGB)
- Normalization
- Noise reduction (optional)
- Label encoding

Implemented using OpenCV and NumPy.

---

## 📈 Model Performance

| Metric | Value |
|---------|--------|
| Training Accuracy | ~99% |
| Validation Accuracy | ~98% |
| Test Accuracy | ~97–99% |

*Performance may vary depending on training parameters and dataset split.*

---

## 📊 Future Improvements

- MobileNet implementation
- EfficientNet architecture
- YOLO integration
- Real-time webcam detection
- Raspberry Pi deployment
- TensorFlow Lite conversion
- Flask web application
- Android application

---

## 🚘 Applications

- Autonomous Vehicles
- Driver Assistance Systems (ADAS)
- Smart Transportation
- Intelligent Traffic Monitoring
- Robotics
- Smart Cities

---

## 📚 Learning Outcomes

This project demonstrates:

- Computer Vision
- Deep Learning
- CNN Architecture
- Image Classification
- TensorFlow Model Training
- OpenCV Image Processing
- Model Evaluation
- Real-time Prediction

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Create Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Parth Kalbhor**

BE Artificial Intelligence & Machine Learning

Savitribai Phule Pune University

GitHub: https://github.com/Parth54321

LinkedIn: *(Add your LinkedIn profile here)*

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub!
