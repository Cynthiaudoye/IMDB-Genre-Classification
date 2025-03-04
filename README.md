# 🎬 Multimodal IMDB Genre Classification with TensorFlow and Keras

## 📌 Project Overview
This project explores **multimodal film genre classification** using **both textual (NLP) and visual (image) data** from the IMDB dataset. The approach leverages:
- **Convolutional Neural Networks (CNNs)** for classifying film genres based on **movie posters** (image classification).
- **Bidirectional LSTMs (Long Short-Term Memory Networks)** for classifying genres using **film overviews** (text classification).

This project was completed as part of my **MSc Data Science coursework** at the **University of Hertfordshire**, exploring deep learning applications in NLP and computer vision.

---

## 📊 Key Features
✔ **Image Classification:** CNN model with **6 convolutional layers**, trained on resized **64x64 film posters**.  
✔ **Text Classification:** LSTM model trained on **tokenized film overviews** using a **256-dimensional embedding layer**.  
✔ **Multimodal Learning:** Evaluates performance differences between text-based and image-based classification.  
✔ **Performance Metrics:** CNN achieved **0.6174 precision**, LSTM achieved **0.6255 precision**.  
✔ **Challenges & Solutions:** Handled **class imbalance** using **oversampling & data augmentation**.  

---

## 🏗️ Model Performance
| Model  | Precision | Observations |
|--------|-----------|-----------------|
| **CNN (Images)** | 0.6174 | Struggled with multi-genre predictions, favored "Drama". |
| **LSTM (Text)** | 0.6255 | Captured text-based nuances well, misclassified rare genres like "Music". |

---

## 🔍 Sample Model Predictions

| Poster | Overview | Ground Truth | CNN Predictions | LSTM Predictions |
|--------|-----------|-----------------|----------------|----------------|
| ![Movie 1](images/movie1.jpg) | A woman moves into an exclusive NYC apartment... | Comedy, Romance | Drama, Thriller, Crime | Drama, Comedy, Romance |
| ![Movie 2](images/movie2.jpg) | A Jewish girl falls in love with a WWII pilot... | Comedy, Crime, Drama | Drama, Romance, Thriller | Drama, Comedy, Romance |

*(Full results available in the [analysis report](Multimodal_IMDB_Analysis_Report.pdf)).*

---

## ⚙️ Technical Stack
- **Deep Learning Frameworks:** TensorFlow, Keras
- **Preprocessing:** TensorFlow `tf.data` pipeline for images, tokenization for text
- **Libraries Used:** NumPy, Pandas, Matplotlib, Seaborn
- **Model Architectures:** CNNs for images, BiLSTMs for text
- **Evaluation Metrics:** Precision, Loss, Confusion Matrix

---

## 🔥 Challenges & Future Work
### **Challenges Faced**
1. **Class Imbalance:** Certain genres like "Music" were underrepresented, leading to misclassifications.
2. **Overfitting:** Dropout layers were used to regularize the CNN and LSTM models.
3. **Feature Representation:** CNN struggled to differentiate between similar genres visually.

### **Potential Improvements**
✔ Use **ResNet** instead of CNN for better image feature extraction.  
✔ Fine-tune **Transformer-based models (BERT, ViT)** for enhanced text/image understanding.  
✔ Implement **multi-label classification techniques** to improve genre predictions.  

---

## 🚀 How to Run the Notebook
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Cynthiaudoye/IMDB-Genre-Classification.git
cd IMDB-Genre-Classification
