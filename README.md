# 🍽️ Sentiment Analysis of Restaurant Reviews Using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Jupyter%20Notebook-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)


This project analyzes restaurant customer feedback to classify sentiments as **Positive**, **Negative**, or **Neutral** using Natural Language Processing (NLP) and Machine Learning techniques. The model is trained on a dataset of restaurant reviews to understand how text data can be used to measure customer satisfaction.

---

## 📂 Dataset Details

**Dataset Name:** Restaurant_Reviews.tsv  
**Source:** Local / UCI Repository-style TSV file  
**Total Records:** 1,000 reviews  
**Columns:**
- **Review** – Text review given by customers.  
- **Liked** – Target variable:  
  - `1` → Positive Review  
  - `0` → Negative Review  

---

## ⚙️ Workflow Overview

### **1️⃣ Data Loading and Inspection**
- Imported the dataset using **Pandas**.  
- Checked dataset dimensions, null values, and distribution of the `Liked` variable.  
- Added new text-based features:
  - Character count
  - Word count
  - Sentence count (using NLTK’s `sent_tokenize`)

---

### **2️⃣ Text Preprocessing (NLP Pipeline)**
Steps applied to clean and normalize the text data:

| Step | Description |
|------|--------------|
| **1. Regex Cleaning** | Removed all non-alphabetic characters. |
| **2. Lowercasing** | Converted all text to lowercase. |
| **3. Tokenization** | Split text into individual words. |
| **4. Stopword Removal** | Used NLTK stopwords, retaining important negations like "not", "no". |
| **5. Stemming** | Used PorterStemmer to reduce words to root forms. |
| **6. Corpus Creation** | Compiled all processed text into a clean corpus for model training. |

---

### **3️⃣ Data Visualization**

#### 🟢 WordCloud for Positive Reviews
A WordCloud was generated to visualize the most frequent positive words, highlighting terms like:
> *"love", "amazing", "great", "delicious", "excellent"*

#### 🔴 WordCloud for Negative Reviews
Shows common negative expressions such as:
> *"bad", "worst", "disappointed", "slow", "terrible"*

*(Visuals generated using the `WordCloud` library and Matplotlib.)*

---

### **4️⃣ Feature Extraction**

Used **CountVectorizer** to convert text corpus into numerical features:
- Limited to the **1500 most frequent words** to reduce sparsity.
- Transformed the processed text into a matrix form suitable for ML algorithms.

---
### **4️⃣ Feature Extraction**

-Split data into 80% training and 20% testing.

-Experimented with multiple classification algorithms (as seen in the notebook, e.g. Naive Bayes, Logistic Regression, etc.).

-Evaluated models using accuracy score, confusion matrix, and classification report.
---
### **6️⃣ Model Evaluation Metrics**

**Typical metrics considered:**

| Metric        | Description                          |
| ------------- | ------------------------------------ |
| **Accuracy**  | Overall correctness of predictions   |
| **Precision** | Positive prediction reliability      |
| **Recall**    | Coverage of actual positives         |
| **F1-Score**  | Balance between Precision and Recall |

### **📊 Key Insights**

-Positive reviews generally contain more characters and words than negative ones.

-Negation words (like “not good”) play a key role in accurate sentiment detection.

The pre-trained model achieved high accuracy (depending on classifier used).
---
| Category             | Libraries                 |
| -------------------- | ------------------------- |
| **Data Handling**    | `pandas`, `numpy`         |
| **Visualization**    | `matplotlib`, `wordcloud` |
| **NLP Processing**   | `nltk`, `re`              |
| **Machine Learning** | `scikit-learn`            |
---
| Visualization                                        | Description                            |
| ---------------------------------------------------- | -------------------------------------- |
| ![Positive WordCloud](assets/wordcloud_positive.png) | Highlights words from positive reviews |
| ![Negative WordCloud](assets/wordcloud_negative.png) | Highlights words from negative reviews |
---
### **🚀 Future Enhancements**

-Implement TF-IDF Vectorization for better feature weighting.

-Deploy model using Streamlit or Flask for real-time sentiment analysis.

Use deep learning (LSTM / BERT) for improved contextual understanding.
---
🧑‍💻 Author

Divyanshu Dharmik
B.Tech 2022 | Machine Learning & Data Science Enthusiast
---
🏁 Conclusion

This project successfully demonstrates how NLP techniques combined with machine learning can extract meaningful insights from textual customer feedback, helping restaurants improve their services based on customer sentiment.

## 🎬 How to Run

1. **Clone this repo**
   ```bash
   git clone https://github.com/yourusername/AI-SPAM-Detector.git
   cd AI-SPAM-Detector
2. Open the notebook
Run AISPAM_detect.ipynb in Jupyter or Google Colab
Upload the dataset
Upload spam.csv when prompted
Try predictions
Enter any review message in the UI box and click “Predict”

### **📷 Screenshots**
- Spam detector interface
- Final UI img of our Project: 
![Screenshot 2025-04-17 065516](https://github.com/user-attachments/assets/5a02d118-c6e5-448c-8b83-311be95188b2)
![Screenshot 2025-04-17 065616](https://github.com/user-attachments/assets/304e8edc-6b3b-4064-99e5-83c3bb434c34)


- Accuracy scores
![image](https://github.com/user-attachments/assets/65aecb04-b1d2-4f53-99b8-20195ac4b60a)

- Confusion matrix plot
![image](https://github.com/user-attachments/assets/18e08afe-e1d2-44f6-b274-96f4cedbf0d0)

🤝 Contributions
Contributions are welcome! Feel free to fork, clone, and submit pull requests.
