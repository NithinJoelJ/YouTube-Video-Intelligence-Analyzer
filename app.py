import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzer import YouTubeAnalyzer

st.set_page_config(page_title="YouTube Video Intelligence Analyzer", page_icon=None, layout="wide")

# Custom CSS for Premium Academic Look
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card { background: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 20px; }
    .syllabus-badge { background-color: #238636; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-right: 5px; }
    h1, h2, h3 { color: #58a6ff !important; }
    .stTabs [data-baseweb="tab-item"] { color: #58a6ff; }
    </style>
""", unsafe_allow_html=True)

st.title("YouTube Video Intelligence Analyzer")
st.markdown("**CSI4004 Academic Research Framework** | Advanced Visual Analytics")

# Sidebar
with st.sidebar:
    st.header("System Status")
    st.success("Visual Engine: Active")
    sub_sample = 200
    st.markdown("---")
    st.header("Academic Alignment")
    st.info("CSI4004 Course Modules: 1, 2, 4, 5, 7")

# Inputs
video_url = st.text_input("YouTube URL:", placeholder="Paste link here...")
analyze_btn = st.button("Analyze Content")

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = YouTubeAnalyzer()

# Analysis
if analyze_btn and video_url:
    if "youtube.com" not in video_url and "youtu.be" not in video_url:
        st.error("Invalid YouTube URL")
    else:
        with st.status(f"Processing {sub_sample} comments...", expanded=True) as status:
            try:
                metadata = st.session_state.analyzer.get_video_data(video_url)
                comments = st.session_state.analyzer.get_comments(video_url, max_comments=sub_sample)
                mining_res = st.session_state.analyzer.perform_advanced_text_mining(comments)
                opinion_res = st.session_state.analyzer.analyze_opinion_mining(comments)
                wc_img = st.session_state.analyzer.generate_wordcloud(comments)
                tfidf_res = st.session_state.analyzer.extract_tfidf_keywords(comments)
                
                # Intelligence Calculations
                trust_score, trust_exp = st.session_state.analyzer.calculate_trust_score(metadata, opinion_res)
                summary = st.session_state.analyzer.generate_summary(metadata, opinion_res, mining_res, trust_score)
                
                st.session_state.update({
                    'active': True, 'm': metadata, 'c': comments, 
                    'mining': mining_res, 'opinion': opinion_res, 'wc': wc_img,
                    'tfidf': tfidf_res, 'trust': trust_score, 'trust_exp': trust_exp,
                    'summary': summary, 'chat_history': []
                })
                status.update(label="Analysis Successful!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error: {e}")

# Results
if st.session_state.get('active'):
    metadata, op, mi = st.session_state.m, st.session_state.opinion, st.session_state.mining
    
    # Summary & Intelligence Header
    st.divider()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Intelligence Summary")
        st.info(st.session_state.summary)
    with c2:
        st.subheader("Trust Score")
        st.metric("System Credibility Rating", f"{st.session_state.trust}/100")
        st.caption(st.session_state.trust_exp)
    
    t1, t2, t3, t4, t5 = st.tabs(["Sentiment Analysis", "Visual Analytics", "Topic Modeling", "Anomaly Detection", "Dataset"])
    
    with t1:
        st.subheader("Sentiment & Opinion Analysis")
        df_stream = op['df']
        fig_stream = px.line(df_stream, x='index', y='sentiment', title="Sentiment Trend",
                             color_discrete_sequence=['#58a6ff'])
        fig_stream.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_stream, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(values=[op['pos'], op['neg'], op['neu']], names=['Positive', 'Negative', 'Neutral'],
                             color=['Positive', 'Negative', 'Neutral'], hole=0.5,
                             color_discrete_map={'Positive':'#238636', 'Negative':'#da3633', 'Neutral':'#8b949e'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.metric("Engagement Votes", f"{df_stream['votes'].sum():,}")
            st.metric("Mean Polarity", f"{df_stream['sentiment'].mean():.2f}")

    with t2:
        st.subheader("Visual Analytics Dashboard")
        
        # Row 1: WordCloud & Phrase Analysis
        r1c1, r1c2 = st.columns([1.2, 1])
        with r1c1:
            st.markdown("**1. Thematic Word Cloud**")
            st.image(st.session_state.wc, width='stretch')
        with r1c2:
            st.markdown("**2. Linguistic Patterns**")
            u_df = pd.DataFrame(list(mi['unigrams'].items()), columns=['Term', 'Count']).sort_values('Count', ascending=False)
            b_df = pd.DataFrame(list(mi['bigrams'].items()), columns=['Phrase', 'Count']).sort_values('Count', ascending=False)
            
            mode = st.radio("Select View", ["Single Words", "Pairs (Bigrams)", "TF-IDF Keywords"], horizontal=True)
            if mode == "Single Words":
                fig_u = px.bar(u_df, x='Count', y='Term', orientation='h', color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig_u, use_container_width=True)
            elif mode == "Pairs (Bigrams)":
                fig_b = px.bar(b_df, x='Count', y='Phrase', orientation='h', color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                t_df = pd.DataFrame(st.session_state.tfidf, columns=['Term', 'Score'])
                fig_t = px.bar(t_df, x='Score', y='Term', orientation='h', title="Top Info-Gain Keywords (TF-IDF)")
                st.plotly_chart(fig_t, use_container_width=True)

        st.divider()

        # Row 2: Scatter & Histogram
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("**3. Sentiment vs. Content Length (Anomaly Discovery)**")
            fig_scat = px.scatter(df_stream, x='length', y='sentiment', size='votes', color='sentiment',
                                  hover_data=['author'], title="Impact of Content Length",
                                  color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_scat, use_container_width=True)
        with r2c2:
            st.markdown("**4. Content Verbosity Distribution (Exploration)**")
            fig_hist = px.histogram(df_stream, x='length', nbins=20, title="Word Count Distribution",
                                    color_discrete_sequence=['#58a6ff'])
            st.plotly_chart(fig_hist, use_container_width=True)

    with t3:
        st.subheader("Topic Modeling (LDA)")
        st.write("Identified primary themes using Latent Dirichlet Allocation:")
        for i, topic in enumerate(mi['topics']):
            st.info(f"**Cluster {i+1}**: {topic}")
        
        st.subheader("Named Entity Recognition")
        e_df = pd.DataFrame(mi['entities'], columns=['Entity', 'Mentions'])
        fig_e = px.bar(e_df, x='Mentions', y='Entity', orientation='h', color='Mentions')
        st.plotly_chart(fig_e, use_container_width=True)

    with t4:
        st.subheader("Anomaly & Influence Detection")
        for a in op['anomalies']:
            st.warning(f"Extreme Sentiment | Author: {a['author']} | Words: {a['length']}")
            st.write(a['text'])
        
        fig_v = px.box(df_stream, y='votes', title="Engagement Variance (Outliers)", points="all")
        st.plotly_chart(fig_v, use_container_width=True)

    with t5:
        st.subheader("Processed Dataset")
        st.dataframe(df_stream, use_container_width=True)

    # Chatbot UI
    st.divider()
    st.subheader("Intelligence Assistant")
    with st.container(border=True):
        st.markdown("**Select a query to execute intelligence analysis:**")
        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
        
        selected_query = None
        if bt_col1.button("Trustworthiness", use_container_width=True): selected_query = "Is this video trustworthy?"
        if bt_col2.button("Main Topics", use_container_width=True): selected_query = "What are the main topics?"
        if bt_col3.button("Sentiment Split", use_container_width=True): selected_query = "What is the audience sentiment?"
        if bt_col4.button("Keywords", use_container_width=True): selected_query = "What are the trending keywords?"
        
        if selected_query:
            response = st.session_state.analyzer.chatbot_response(
                selected_query, st.session_state.m, st.session_state.opinion, 
                st.session_state.mining, st.session_state.trust, st.session_state.trust_exp
            )
            st.session_state.chat_history.insert(0, (selected_query, response))
        
        if st.session_state.chat_history:
            st.markdown("---")
            for q, a in st.session_state.chat_history[:5]: # Show last 5
                st.markdown(f"**Query:** {q}")
                st.info(a)

else:
    st.info("Please provide a valid YouTube URL to generate the academic visual suite.")
