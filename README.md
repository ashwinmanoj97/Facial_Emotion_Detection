# **Emotion Detection System**

This repository contains a complete implementation of a **Human Face Emotion Detection System** using Python, OpenCV, TensorFlow, and Keras. It employs a **Convolutional Neural Network (CNN)** architecture and is pre-trained on the **FER-2013 dataset**.

## **Features**
- Real-time emotion detection using webcam feed.
- Pre-trained model using the FER-2013 dataset.
- Modular scripts for:
  - Verifying Python installation (`verify_installation.py`).
  - Checking GPU compatibility (`verify_gpu.py`).
  - Verifying dataset structure (`verify_dataset.py`).
- Easy retraining functionality to use new datasets.

---

## **Project Structure**

```plaintext
├── verify_installation.py       # Script to verify required libraries
├── verify_gpu.py                # Script to verify GPU compatibility
├── verify_dataset.py            # Script to check dataset structure
├── data_loader.py               # Script for loading and preprocessing the dataset
├── model.py                     # Script for defining the CNN model architecture
├── train_model.py               # Script to train the CNN model
├── real_time_detection.py       # Script for real-time emotion detection via webcam
├── models/                      # Folder containing the pre-trained model (FER-2013 dataset)
│   └── emotion_model_final.h5   # Pre-trained CNN model
├── requirements.txt             # Python dependencies
└── README.md                    # Project description and instructions


---

## **Getting Started**

### **Prerequisites**
1. Python 3.8 or later installed.
2. Create a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate     # For Windows
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Usage**

#### **1. Verify Installation**
Run the script to check if all required libraries are installed:
```bash
python verify_installation.py
```

#### **2. Verify GPU Compatibility**
Run the script to check if TensorFlow can access a GPU:
```bash
python verify_gpu.py
```

#### **3. Verify Dataset Structure**
Check the dataset structure for proper organization:
```bash
python verify_dataset.py
```

#### **4. Real-Time Emotion Detection**
Run the pre-trained model to detect emotions in real-time via webcam:
```bash
python real_time_detection.py
```

---

## **Retraining the Model with a New Dataset**

### **Steps to Retrain:**
1. **Organize the Dataset**:
   - Ensure the dataset is structured as follows:
     ```plaintext
     data/
         train/
             angry/
             happy/
             sad/
             neutral/
         test/
             angry/
             happy/
             sad/
             neutral/
     ```
   - Replace "angry", "happy", etc., with your specific emotion categories.

2. **Update `train_model.py`**:
   - Adjust the `num_classes` parameter in `create_model` to match the number of categories in your dataset.

3. **Train the Model**:
   - Run the training script:
     ```bash
     python train_model.py
     ```

4. **Save the New Model**:
   - The updated model will be saved as `models/emotion_model_final.h5`.

5. **Test the New Model**:
   - Use the new model in `real_time_detection.py` to test real-time emotion detection.

---

## **Dependencies**
The project requires the following libraries, specified in `requirements.txt`:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

Install them using:
```bash
pip install -r requirements.txt
```

---

## **Acknowledgments**
- The model is pre-trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).
- Inspired by various open-source implementations of real-time emotion detection.
