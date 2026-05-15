# 📊 YouTube NLP Intelligence & Sentiment Analyzer
### **CSI4001: Natural Language Processing** | Advanced Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20|%20VADER%20|%20LDA-green)](https://www.nltk.org/)
[![Academic](https://img.shields.io/badge/Course-CSI4001-orange)](file:///Users/joel/Desktop/NLP/YouTube-Video-Intelligence-Analyzer/README.md)

---

## 🚀 Project Overview
The **YouTube NLP Intelligence & Sentiment Analyzer** is a high-performance research platform designed to extract and analyze natural language data from social media. Built as a capstone project for the **Natural Language Processing (CSI4001)** curriculum, this application transforms raw audience feedback into structured intelligence using state-of-the-art NLP pipelines.

### **Core Capabilities**
- 🧠 **Intelligence Assistant:** An interactive rule-based chatbot for semantic querying.
- 📈 **Sentiment Polarity Tracking:** Real-time analysis of emotional trends and public opinion.
- 🔍 **Topic Modeling (LDA):** Automated thematic extraction using Latent Dirichlet Allocation.
- 🏷️ **Entity Extraction:** Named Entity Recognition (NER) for identifying key discourse subjects.
- 🛡️ **Trust Score Engine:** A weighted algorithm for content credibility analysis.

---

## 📚 Syllabus Alignment (CSI4001)
This project serves as a comprehensive implementation of the CSI4001 Natural Language Processing curriculum.

| Module | Topic | Project Implementation |
| :--- | :--- | :--- |
| **Module 1** | **Overview of NLP** | Basic text processing, tokenization, and NLTK integration for data cleaning. |
| **Module 5** | **POS Tagging** | Structural analysis foundation for Named Entity Recognition (NER). |
| **Module 6** | **Lexical Semantics** | Semantic mapping, word similarity, and thematic clustering. |
| **Module 7** | **Applications of NLP** | Sentiment Analysis, Text Summarization, and Information Extraction. |
| **Module 8** | **Contemporary Issues** | Real-world social media data mining and anomaly detection. |

---

## 🛠️ Technical Architecture

### **1. Data Acquisition & Preprocessing**
- **Extraction:** Utilizes `yt-dlp` and `youtube-comment-downloader` for high-fidelity data retrieval.
- **Cleaning:** Advanced Regex-based cleaning (removal of emojis, URLs, and handles) followed by NLTK-driven tokenization and stop-word filtering.

### **2. Sentiment & Opinion Mining**
- **Algorithm:** VADER (Valence Aware Dictionary and sEntiment Reasoner).
- **Metric:** Compound polarity scores mapped across a time-series distribution to identify sentiment shifts.

### **3. Probabilistic Topic Modeling**
- **Algorithm:** Latent Dirichlet Allocation (LDA) via `Scikit-Learn`.
- **Output:** Identifies latent thematic clusters within thousands of comments, providing a "bird's eye view" of the conversation.

### **4. Visual Analytics Suite**
- **WordClouds:** Thematic word frequency visualization.
- **Plotly Integration:** Interactive scatter plots for anomaly detection (Sentiment vs. Content Length).
- **TF-IDF Bar Charts:** Visualizing Information Gain and keyword significance.

---

## 💻 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/NithinJoelJ/YouTube-Video-Intelligence-Analyzer.git
   cd YouTube-Video-Intelligence-Analyzer
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 🎓 Academic Framework
**Course:** Natural Language Processing (CSI4001)  
**Curriculum:** Fall Semester 2026  
**Developer:** Nithin Joel J  

*This project is submitted as part of the academic requirements for the CSI4001 course, demonstrating proficiency in text mining, statistical NLP, and interactive visual analytics.*
