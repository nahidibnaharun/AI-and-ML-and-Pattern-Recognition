# 🧠 MNIST Handwritten Digit Classification with CNN

This project trains a **Convolutional Neural Network (CNN)** on the **MNIST handwritten digit dataset** using Google Colab and TensorFlow/Keras.  
The model achieves ~99% accuracy and includes full evaluation with graphs, confusion matrix, and ROC AUC curves.

---

## 📂 Dataset
We use the **MNIST dataset** in CSV format from Kaggle:

**Dataset_Download_Link:**  
👉 [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

It contains:
- `mnist_train.csv` → 60,000 training samples  
- `mnist_test.csv` → 10,000 test samples  

Each row:
- First column = digit label (0–9)  
- Remaining 784 columns = pixel values of a `28x28` grayscale image (flattened)

---

## 🚀 Workflow
1. **Setup & Mount Drive**  
   - Mount Google Drive in Colab  
   - Import libraries (TensorFlow, sklearn, pandas, matplotlib, etc.)

2. **Load & Preprocess Dataset**  
   - Load CSVs from Drive  
   - Normalize pixel values to `[0,1]`  
   - Reshape into `(28,28,1)` for CNN  
   - Train/Validation/Test split  
   - One-hot encode labels  

3. **Build CNN Model**  
   - Data augmentation (rotation, zoom, translation)  
   - Two Conv2D + BatchNorm + MaxPooling + Dropout blocks  
   - Dense layers with BatchNorm & Dropout  
   - Softmax output for 10 classes  

4. **Training**  
   - Optimizer: Adam  
   - Loss: Categorical Crossentropy  
   - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  

5. **Evaluation & Visualization**  
   - Accuracy & Loss curves  
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-score)  
   - ROC AUC per class + macro average  
   - Visualize correct vs incorrect predictions  

6. **Save Model**  
   - Save trained model (`.h5`) to Google Drive or repo

---

## 📊 Results
- **Test Accuracy**: ~99%  
- **Metrics**: High precision/recall for all digits  
- **Visuals**:  
  - Training curves  
  - Confusion matrix  
  - ROC curves  
  - Sample predictions (correct = green, wrong = red)

---

## 🛠 How to Run
1. Open in **Google Colab**.  
2. Upload `mnist_train.csv` and `mnist_test.csv` to your Google Drive.  
3. Update dataset paths in the notebook.  
4. Run all cells sequentially.  
5. Evaluate results and save model.

---

## 📦 Requirements
- Python 3.8+  
- TensorFlow / Keras  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- scikit-learn  

---

## 📌 References
- [Yann LeCun’s MNIST page](http://yann.lecun.com/exdb/mnist/)  
- [Kaggle MNIST CSV Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
- TensorFlow Documentation  

---

## 📜 License
This project is open-source and free to use for educational purposes.
