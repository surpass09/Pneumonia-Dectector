# **Pneumonia Detector 🩺**  
A deep learning model to classify **chest X-rays** and detect pneumonia with an accuracy of **80%**.

## **🧠 Model Overview**  
This project leverages a **Convolutional Neural Network (CNN)** to predict whether a patient has pneumonia based on chest X-ray images. The model is trained on labeled X-ray datasets and learns to distinguish between **normal lungs** and **lungs with pneumonia**.

## **📂 Dataset**  
The model is trained using **Kaggle’s Chest X-ray dataset**, which contains:  
✔ **Normal X-rays** (Healthy lungs)  
✔ **Pneumonia X-rays** (Lungs with infection)  

The dataset is preprocessed by resizing images, normalizing pixel values, and applying data augmentation to improve generalization.

## **🔍 Model Architecture**  
The neural network consists of:  
- **Convolutional Layers**: Extracts key features from X-ray images.  
- **Batch Normalization & Dropout**: Reduces overfitting and stabilizes training.  
- **Fully Connected Layers**: Classifies images into "Normal" or "Pneumonia".  
- **Activation Functions**: Uses **ReLU** for feature extraction and **Softmax/Sigmoid** for classification.

## **⚙️ Training & Optimization**  
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam  
- **Epochs**: 1000  
- **Batch Size**: 32  
- **Training Accuracy**: **97%**  
- **Test Accuracy**: **80%**  

The **train-test accuracy gap** suggests some level of overfitting, which can be addressed with further data augmentation or regularization techniques.

## **📊 Results & Performance**  
- The model achieved **high training accuracy** but a slightly lower test accuracy, indicating good generalization with room for improvement.  
- It successfully detects pneumonia in **80% of test cases**, making it a viable tool for preliminary screening.  
- Further enhancements could include transfer learning using models like **ResNet50** or **VGG16**.

## **🚀 How to Run the Model**  
1️⃣ Clone the repository:  
   ```bash
   git clone git@github.com:surpass09/Pneumonia-Dectector.git
   cd Pneumonia-Dectector
