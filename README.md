# 🍽️ Sentiment Analysis of Restaurant Reviews Using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Jupyter%20Notebook-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)


This project analyzes **restaurant customer feedback** to classify sentiments as **Positive**, **Negative**, or **Neutral** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  

The aim is to understand how text data can be leveraged to measure **customer satisfaction** and improve restaurant services through intelligent feedback analysis.  

---

## 📊 Project Showcase  

| Positive WordCloud | Negative WordCloud |
|:-------------------:|:-------------------:|
| ![Positive WordCloud](Word%20Cloud%20for%20Positive%20Reviews.png) | ![Negative WordCloud](Word%20Cloud%20for%20Negative%20Reviews.png) |

| Confusion Matrix | Accuracy Graph |
|:----------------:|:----------------:|
| ![Confusion Matrix](Confusion%20Matrix.png) | ![Accuracy Graph](Accuracy%20Graph.png) |

🎥 **Output Dashboard Video:**  
[![Output Dashboard](https://img.shields.io/badge/▶️%20Watch-Demo-blue)](ML%20DD.mp4)  
*(Click the button above to view model demo video)*  

---

## 📂 Dataset Details  

- **Dataset Name:** `Restaurant_Reviews.tsv`  
- **Source:** Local / UCI Repository-style TSV file  
- **Total Records:** 1,000 restaurant reviews  

### 🧾 Columns  
| Column | Description |
|--------|--------------|
| `Review` | Text review given by customers |
| `Liked` | Target variable — `1 → Positive`, `0 → Negative` |

---

## ⚙️ Workflow Overview  

### 1️⃣ Data Loading and Inspection  
- Loaded dataset using **Pandas**  
- Checked for null values and class distribution  
- Added new text-based features like:
  - **Character Count**
  - **Word Count**
  - **Sentence Count** (via NLTK `sent_tokenize`)

---

### 2️⃣ Text Preprocessing (NLP Pipeline)  

| Step | Description |
|------|--------------|
| **Regex Cleaning** | Removed non-alphabetic characters using Regular Expressions |
| **Lowercasing** | Converted text to lowercase |
| **Tokenization** | Split text into words |
| **Stopword Removal** | Removed stopwords but kept negations (e.g., “not”, “no”) |
| **Stemming** | Used PorterStemmer to reduce words to root forms |
| **Corpus Creation** | Compiled all processed text for model training |

---

### 3️⃣ Data Visualization  

- **Positive Review WordCloud:** Shows frequent positive expressions like *love*, *amazing*, *delicious*, *great*, *excellent*.  
- **Negative Review WordCloud:** Highlights negative expressions like *bad*, *worst*, *disappointed*, *slow*, *terrible*.  

> Generated using `wordcloud` and `matplotlib` libraries.  

---

### 4️⃣ Feature Extraction  

- Used **CountVectorizer** to convert text into numerical features.  
- Limited vocabulary to **1,500 most frequent words** to reduce sparsity.  
- Split data into **80% training** and **20% testing** sets.  

---

### 5️⃣ Model Training & Evaluation  

Experimented with various **classification algorithms**:  
- Naive Bayes (MultinomialNB)  
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Evaluation Metrics:**  
| Metric | Description |
|--------|--------------|
| Accuracy | Overall model performance |
| Precision | Correctness of positive predictions |
| Recall | Model’s ability to detect positives |
| F1-Score | Balance between precision & recall |

---

## 📈 Model Performance  

- Positive reviews generally have **more characters and words** than negative ones.  
- **Negations (“not good”)** strongly influence sentiment prediction.  
- Achieved **high accuracy** (varies by algorithm).  

**Confusion Matrix Example:**  
![Confusion Matrix](Confusion%20Matrix.png)

**Accuracy Visualization:**  
![Accuracy Graph](Accuracy%20Graph.png)

---

## 🧰 Libraries & Tools Used  

| Category | Libraries |
|-----------|------------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `wordcloud` |
| NLP Processing | `nltk`, `re` |
| Machine Learning | `scikit-learn` |

---

## 🚀 Future Enhancements  

- 🔹 Implement **TF-IDF Vectorization** for weighted feature extraction.  
- 🔹 Deploy using **Streamlit** or **Flask** for interactive real-time sentiment analysis.  
- 🔹 Use **Deep Learning** models like **LSTM** or **BERT** for contextual sentiment understanding.  

---

## 🖥️ Output Dashboard  

Below are the sample visuals generated after model training and evaluation:  

| Visualization | Description |
|---------------|-------------|
| **Positive WordCloud** | Highlights words commonly used in positive reviews |
| **Negative WordCloud** | Shows most used words in negative feedback |
| **Confusion Matrix** | Displays true vs predicted classifications |
| **Accuracy Graph** | Compares performance of multiple classifiers |

---

## 🧑‍💻 Author  

**Divyanshu Dharmik**  
🎓 *B.Tech 2022 | Machine Learning & Data Science Enthusiast*  
💡 *Focused on NLP, AI, and real-world model deployment.*  

📧 **Contact:** [LinkedIn Profile or Email if preferred]  

---

## 🪪 License  

This project is licensed under the **MIT License** — feel free to use, modify, and share for educational or research purposes.  

---

### ⭐ Don’t forget to star this repository if you found it helpful!
