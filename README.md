# YouTube Video Intelligence Analyzer

## Project Overview
The YouTube Video Intelligence Analyzer is a comprehensive text-mining suite designed to extract, process, and analyze video content and audience interactions. Built as part of the CSI4004 Academic Research Framework, this application uses advanced natural language processing (NLP) and machine learning techniques to provide actionable insights into content credibility, public sentiment, and thematic trends.

## Core Features
- **Metadata Analytics:** Extraction of video performance metrics (views, likes, channel data) using local extraction tools.
- **Sentiment & Opinion Analysis:** Real-time sentiment tracking and distribution analysis using the VADER lexicon.
- **Topic Modeling:** Thematic clustering of audience feedback using Latent Dirichlet Allocation (LDA).
- **Information Gain Extraction:** High-importance keyword identification via TF-IDF (Term Frequency-Inverse Document Frequency).
- **Trust Score System:** An automated credibility rating based on engagement ratios and sentiment consistency.
- **Anomaly Detection:** Identification of outlier comments and extreme sentiment deviations.
- **Intelligence Assistant:** A rule-based local chatbot for interactive data querying.

## Technical Architecture
- **Language:** Python 3.9+
- **Data Acquisition:** yt-dlp, youtube-comment-downloader
- **NLP & Mining:** scikit-learn, vaderSentiment, NLTK
- **Visualization:** Plotly, Matplotlib, WordCloud
- **UI Framework:** Streamlit (Custom Dark Theme)

## Academic Alignment (CSI4004)
This project maps directly to the following text mining modules:
- **Module 1:** Information Extraction and Named Entity Recognition.
- **Module 2/5:** Probabilistic Topic Modeling (LDA).
- **Module 4:** Visual Analytics and Data Exploration.
- **Module 7:** Sentiment Analysis and Opinion Mining.

## Installation and Deployment

### Prerequisites
- Python installed on your system.
- Git (optional, for cloning).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/NithinJoelJ/YouTube-Video-Intelligence-Analyzer.git
   cd YouTube-Video-Intelligence-Analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize NLTK resources:
   The application will automatically attempt to download required resources on the first run.

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## System Implementation
The system is divided into two primary modules:
1. **analyzer.py:** Contains the backend logic for data extraction, text preprocessing, sentiment analysis, and topic modeling.
2. **app.py:** Manages the frontend interactive dashboard and visualization suite.

## Methodology
The trust score calculation uses a weighted algorithm:
- **Engagement Quality (40%):** Derived from the like-to-view ratio.
- **Sentiment Positivity (40%):** Calculated from the percentage of positive audience feedback.
- **Negative Penalty (20%):** Deductions based on high toxicity or negative sentiment concentration.

## License
This project is for research and educational purposes as part of the Fall Semester Text Mining curriculum.
