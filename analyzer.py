import yt_dlp
from youtube_comment_downloader import YoutubeCommentDownloader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
import matplotlib.pyplot as plt
import io
import nltk
from nltk.corpus import stopwords

# Download stopwords
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

class YouTubeAnalyzer:
    def __init__(self):
        self.downloader = YoutubeCommentDownloader()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def get_video_data(self, url):
        ydl_opts = {'quiet': True, 'no_warnings': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'views': info.get('view_count', 0),
                    'likes': info.get('like_count', 0),
                    'publish_date': info.get('upload_date'),
                    'channel_name': info.get('uploader'),
                    'duration': info.get('duration')
                }
        except Exception as e:
            raise Exception(f"Failed to fetch video data: {str(e)}")

    def get_comments(self, url, max_comments=150):
        comments = []
        try:
            gen = self.downloader.get_comments_from_url(url, sort_by=0)
            for i, comment in enumerate(gen):
                comments.append({
                    'text': comment['text'],
                    'author': comment['author'],
                    'votes': int(comment.get('votes', 0) or 0),
                    'time': comment.get('time', 'unknown')
                })
                if i + 1 >= max_comments:
                    break
        except Exception as e:
            print(f"Error fetching comments: {e}")
        return comments

    def perform_advanced_text_mining(self, comments_list):
        if not comments_list:
            return None

        texts = [c['text'] for c in comments_list]
        cleaned_texts = [clean_text(t) for t in texts if len(clean_text(t)) > 10]
        
        if len(cleaned_texts) < 5:
            return None

        # 1. Topic Modeling (LDA)
        tf_vectorizer = CountVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        tf = tf_vectorizer.fit_transform(cleaned_texts)
        n_topics = min(5, len(cleaned_texts) // 12 + 1)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tf)
        
        feature_names = tf_vectorizer.get_feature_names_out()
        lda_topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
            lda_topics.append(", ".join(top_words))

        # 2. Entity Frequency for Visualization (Module 1)
        all_text = " ".join(texts)
        entities = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b[A-Z][a-z]+\b', all_text)
        filtered_entities = [e for e in entities if e.lower() not in stop_words and len(e) > 3]
        
        # 3. N-gram Analysis for Module 4/1
        cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_features=10)
        unigrams = cv.fit_transform(cleaned_texts)
        uni_counts = dict(zip(cv.get_feature_names_out(), unigrams.toarray().sum(axis=0)))
        
        cv2 = CountVectorizer(stop_words='english', ngram_range=(2, 2), max_features=10)
        bigrams = cv2.fit_transform(cleaned_texts)
        bi_counts = dict(zip(cv2.get_feature_names_out(), bigrams.toarray().sum(axis=0)))

        return {
            'topics': lda_topics,
            'entities': Counter(filtered_entities).most_common(10),
            'unigrams': uni_counts,
            'bigrams': bi_counts,
            'processed_count': len(cleaned_texts)
        }

    def analyze_opinion_mining(self, comments_list):
        if not comments_list:
            return None

        data = []
        for i, c in enumerate(comments_list):
            score = self.sentiment_analyzer.polarity_scores(c['text'])['compound']
            c_len = len(c['text'].split())
            data.append({
                'index': i, 
                'sentiment': score, 
                'votes': int(c.get('votes', 0) or 0),
                'length': c_len,
                'author': c['author'],
                'text': c['text']
            })
            c['sentiment_val'] = score 
            c['length'] = c_len

        df = pd.DataFrame(data)
        
        return {
            "pos": round((df['sentiment'] > 0.1).mean() * 100, 1),
            "neg": round((df['sentiment'] < -0.1).mean() * 100, 1),
            "neu": round(((df['sentiment'] >= -0.1) & (df['sentiment'] <= 0.1)).mean() * 100, 1),
            "df": df, # Full dataset for plotly
            "anomalies": df[df['sentiment'] < -0.85].head(5).to_dict('records')
        }

    def generate_wordcloud(self, comments_list):
        text = " ".join([clean_text(c['text']) for c in comments_list])
        
        if WORDCLOUD_AVAILABLE:
            wc = WordCloud(width=800, height=400, background_color='#161b22', colormap='Blues', stopwords=stop_words).generate(text)
            
            img = io.BytesIO()
            plt.figure(figsize=(10, 5), facecolor='#161b22')
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(img, format='png', bbox_inches='tight')
            plt.close()
            return img.getvalue()
        else:
            # Fallback: Top words bar chart using Matplotlib
            all_words = text.split()
            filtered = [w for w in all_words if w not in stop_words and len(w) > 4]
            counts = Counter(filtered).most_common(15)
            words = [c[0] for c in counts]
            vals = [c[1] for c in counts]
            
            img = io.BytesIO()
            plt.figure(figsize=(10, 5), facecolor='#161b22')
            plt.barh(words, vals, color='#58a6ff')
            plt.gca().invert_yaxis()
            plt.title('Top Trending Terms (Data Projection)', color='white')
            plt.tick_params(colors='white')
            plt.tight_layout()
            plt.savefig(img, format='png')
            plt.close()
            return img.getvalue()

    def extract_tfidf_keywords(self, comments_list):
        if not comments_list: return []
        texts = [clean_text(c['text']) for c in comments_list if len(clean_text(c['text'])) > 10]
        if not texts: return []
        tfidf = TfidfVectorizer(max_features=10, stop_words='english')
        matrix = tfidf.fit_transform(texts)
        scores = zip(tfidf.get_feature_names_out(), matrix.toarray().sum(axis=0))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def calculate_trust_score(self, metadata, opinion_res):
        # 1. Like-to-View Ratio (Engagement Quality)
        views = metadata.get('views') or 1
        likes = metadata.get('likes') or 0
        lvr = (likes / views) * 100
        lvr_score = min(lvr * 10, 40) # Max 40 points for engagement

        # 2. Sentiment Positivity
        pos_perc = opinion_res.get('pos', 0)
        sen_score = (pos_perc / 100) * 40 # Max 40 points for positivity

        # 3. Negative Penalty
        neg_perc = opinion_res.get('neg', 0)
        penalty = (neg_perc / 100) * 20 # Max 20 points deduction for toxicity

        total = round(lvr_score + sen_score - penalty + 20) # Baseline 20
        total = max(0, min(100, total))
        
        explanation = f"Based on {lvr:.2f}% engagement rate and {pos_perc}% positive sentiment, "
        if total > 75: explanation += "this video is highly trustworthy."
        elif total > 50: explanation += "this video has moderate credibility."
        else: explanation += "caution is advised; community feedback is mixed or negative."
        
        return total, explanation

    def generate_summary(self, metadata, opinion_res, mining_res, trust_score):
        title = metadata.get('title', 'Unknown Video')
        views = metadata.get('views', 0)
        pos = opinion_res.get('pos', 0)
        topics = ", ".join(mining_res.get('topics', [])[:2]) if mining_res else "None"
        
        summary = f"Analysis of '{title}' ({views:,} views) reveals a primary public sentiment of {pos}% positivity. "
        summary += f"The linguistic landscape is dominated by themes such as {topics}. "
        summary += f"With a calculated Trust Score of {trust_score}/100, the content is categorized as "
        summary += "reliable" if trust_score > 60 else "subjective or controversial"
        summary += " in the current academic dataset context."
        return summary

    def chatbot_response(self, query, metadata, opinion_res, mining_res, trust_score, trust_exp):
        q = query.lower()
        if "trust" in q:
            return f"The system has assigned a Trust Score of {trust_score}/100. {trust_exp}"
        elif "topic" in q or "subject" in q:
            topics = "\n".join([f"- {t}" for t in mining_res['topics']])
            return f"The main extracted topics are:\n{topics}"
        elif "sentiment" in q or "feel" in q or "opinion" in q:
            return f"The audience sentiment is {opinion_res['pos']}% Positive, {opinion_res['neg']}% Negative, and {opinion_res['neu']}% Neutral."
        elif "keyword" in q or "trend" in q:
            keys = ", ".join(list(mining_res['unigrams'].keys())[:5])
            return f"Trending keywords include: {keys}."
        else:
            return "I can answer questions regarding the video's trust score, topics, sentiment, or trending keywords. Please try rephrasing."
