# 🍽️ Sentiment Analysis of Restaurant Reviews Using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Jupyter%20Notebook-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

## 🔍 Abstract
This project focuses on analyzing restaurant customer feedback using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to classify sentiments as **Positive**, **Negative**, or **Neutral**.  
The motivation behind this work lies in helping restaurants better understand their customers’ emotions, experiences, and opinions through text analysis.  

Sentiment analysis enables automated opinion mining, providing valuable insights that can improve decision-making and enhance customer satisfaction. The project demonstrates a full NLP pipeline — text cleaning, tokenization, feature extraction, and sentiment classification using supervised learning algorithms.  

The model achieved strong accuracy and interpretability, showing how linguistic patterns like negations and adjectives significantly impact polarity detection.  
Ultimately, this project highlights the power of NLP and ML to analyze human language, revealing actionable business intelligence for the hospitality industry.
This project analyzes **restaurant customer feedback** to classify sentiments as **Positive**, **Negative**, or **Neutral** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  

The aim is to understand how text data can be leveraged to measure **customer satisfaction** and improve restaurant services through intelligent feedback analysis.  

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
## 📘 Project Overview
Restaurants and food services rely heavily on customer feedback to assess service quality and customer satisfaction. However, manually analyzing thousands of reviews is impractical.  
This project automates that process by applying **machine learning models to analyze restaurant reviews** and predict customer sentiment.

The project covers:
- Preprocessing raw text data  
- Converting text into numerical representations  
- Training ML classifiers  
- Evaluating and comparing model performance  
- Visualizing frequently used positive and negative words  

The outcome is an efficient system that predicts whether a review expresses a positive or negative sentiment, enabling data-driven business insights.
---
## 📊 Project Showcase  

| Output Dashboard |  
|:----------------:|
| ![Confusion Matrix](https://github.com/Divyanshuxoxo/Machine-Learning-Theory-/blob/main/Output%20dashboard) 

🎥 **Output Dashboard Video:**  
[![Output Dashboard](https://github.com/Divyanshuxoxo/Machine-Learning-Theory-/blob/main/Output%20genration.mp4))  
*(Click the button above to view model demo video)*  

---
### 🧹 Preprocessing and Data Treatment
Before model training, data underwent several cleaning and normalization steps:
- Removed **missing and duplicate entries**.  
- Applied **regular expressions** to eliminate non-alphabetic characters.  
- Converted all text to **lowercase** for uniformity.  
- Used **NLTK stopwords** (excluding negations like “not”, “no”).  
- Performed **tokenization** to split text into words.  
- Applied **stemming** using `PorterStemmer` to reduce words to their root forms.  
- Created a **clean corpus** to represent all processed text reviews.  

These preprocessing steps ensured that the model could focus on meaningful patterns while ignoring irrelevant noise from punctuation or case variations.

---

## 🧠 Methodology

The following methodological framework was used to develop the sentiment classifier:

1. **Data Collection**  
   - Imported and inspected the dataset using **Pandas** to check for missing values and understand label distribution.

2. **Data Preprocessing**  
   - Cleaned the text data using **Regex**, **lowercasing**, **stopword removal**, and **stemming** to prepare a structured corpus.

3. **Feature Extraction**  
   - Transformed the text into numerical vectors using **CountVectorizer**.  
   - Selected the top 1,500 frequent words to reduce dimensionality and improve efficiency.

4. **Model Training**  
   - Used classification models such as **Naive Bayes**, **Logistic Regression**, and **Support Vector Machine (SVM)** for sentiment prediction.

5. **Model Evaluation**  
   - Compared performance using metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**.  
   - Visualized results using confusion matrices.

6. **Visualization & Interpretation**  
   - Generated **WordClouds** for both positive and negative reviews.  
   - Identified common sentiment terms and explored their relationships to labels.

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
## 🧪 Experiments and Results Summary

### 🧮 Model Performance Comparison

| Model                 | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|------------|--------|----------|
| Naive Bayes            | 84%      | 0.86       | 0.82   | 0.84     |
| Logistic Regression    | 88%      | 0.89       | 0.87   | 0.88     |
| Support Vector Machine | 90%      | 0.91       | 0.89   | 0.90     |

---

### 🔍 Key Observations

- Positive reviews were generally longer and contained more descriptive words like **“delicious”**, **“amazing”**, and **“great”**.  
- Negative reviews commonly included negations such as **“not good”**, **“bad”**, and **“disappointed”**.  
- **SVM** provided the highest accuracy and balanced F1-score, making it the best-performing model.

---

### 🎨 Visualizations

#### ☁️ WordClouds
- **Positive reviews** emphasized: *“excellent”*, *“tasty”*, and *“friendly”*.  
- **Negative reviews** emphasized: *“slow”*, *“cold”*, and *“worst”*.  

| Positive WordCloud | Negative WordCloud |
|:-------------------:|:-------------------:|
| ![Positive WordCloud](Word%20Cloud%20for%20Positive%20Reviews.png) | ![Negative WordCloud](Word%20Cloud%20for%20Negative%20Reviews.png) |

# 📈 Model Performance  

- Positive reviews generally have **more characters and words** than negative ones.  
- **Negations (“not good”)** strongly influence sentiment prediction.  
- Achieved **high accuracy** (varies by algorithm).  

**Confusion Matrix Example:**  
![Confusion Matrix](Confusion%20Matrix.png)

**Accuracy Visualization:**  
![Accuracy Graph](Accuracy%20Graph.png)

---
## ** Model Flow **
![Model Flowchart](https://github.com/Divyanshuxoxo/Machine-Learning-Theory-/blob/main/Model%20Flowchart.jpg)

## 🧰 Libraries & Tools Used  

| Category | Libraries |
|-----------|------------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `wordcloud` |
| NLP Processing | `nltk`, `re` |
| Machine Learning | `scikit-learn` |


## 📊 Insights

- **Text length** and **word richness** directly influence sentiment polarity.  
- **Preprocessing quality** (especially handling of negations) significantly affects accuracy.  
- **Machine Learning-based sentiment analysis** can replace manual review sorting, saving substantial time and effort for restaurants.

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

## 🧑‍💻 Author  **

**Divyanshu Dharmik** 
🎓 *B.Tech 2022 | Machine Learning & Data Science Enthusiast*  
💡 *Focused on NLP, AI, and real-world model deployment.*  

📧 **Contact:** [dharmikdivyanshu1406@gmail.com]  

---
## 📊 Insights

- **Text length** and **word richness** directly influence sentiment polarity.  
- **Preprocessing quality** (especially handling of negations) significantly affects accuracy.  
- **Machine Learning-based sentiment analysis** can replace manual review sorting, saving substantial time and effort for restaurants.

---

## 🏁 Conclusion

This project successfully demonstrates how **Natural Language Processing (NLP)** combined with **Machine Learning (ML)** can classify and analyze restaurant reviews effectively.  
Through careful **text preprocessing**, **feature engineering**, and **model comparison**, the system achieved a high level of accuracy in sentiment classification.

The work illustrates how NLP pipelines can help businesses understand their customers better, **prioritize improvements**, and make **data-driven marketing decisions**.

---

## 📚 References

1. [NLTK Documentation](https://www.nltk.org/)  
2. [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)  
3. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)  
4. [WordCloud Library by Andreas Mueller](https://github.com/amueller/word_cloud)  
5. [Coursera NLP Specialization](https://www.coursera.org/specializations/natural-language-processing)

---
## 🪪 License  

This project is licensed under the **MIT License** — feel free to use, modify, and share for educational or research purposes.  

---

### ⭐ Don’t forget to star this repository if you found it helpful!
