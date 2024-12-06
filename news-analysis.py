import streamlit as st
import pandas as pd
import requests
import xmltodict
from transformers import pipeline
import torch
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from dateutil import parser
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

device = "cuda" if torch.cuda.is_available() else "cpu"
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    batch_size=32,
)

RSS_FEEDS = {
    "Technology": {
        "BBC Tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "NY Times Tech": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        "Guardian Tech": "https://www.theguardian.com/technology/rss",
        "Wired": "https://www.wired.com/feed/rss",
        "TechCrunch": "https://techcrunch.com/feed/",
        "Ars Technica": "http://feeds.arstechnica.com/arstechnica/index",
    },
    "Business": {
        "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    },
    "Science": {
        "NASA": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    },
    "World News": {
        "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    },
}


@st.cache_data(ttl=600)
def get_news_from_rss(categories=None, sources=None, days_back=7):
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=days_back)

    selected_feeds = {
        source: url
        for category in categories
        for source, url in RSS_FEEDS[category].items()
    }

    if sources:
        selected_feeds = {k: v for k, v in selected_feeds.items() if k in sources}

    for source, url in selected_feeds.items():
        try:
            response = requests.get(url, timeout=5)
            data = xmltodict.parse(response.content)
            articles = data["rss"]["channel"]["item"][:30]

            for article in articles:
                description = article.get("description", "")
                if not description or len(description.strip()) < 10:
                    continue

                try:
                    parsed_date = parser.parse(article["pubDate"])
                    if parsed_date < cutoff_date:
                        continue
                except:
                    parsed_date = datetime.now()

                all_articles.append(
                    {
                        "source": source,
                        "category": next(
                            cat
                            for cat, sources in RSS_FEEDS.items()
                            if source in sources
                        ),
                        "title": article["title"],
                        "description": description,
                        "date": parsed_date,
                        "link": article["link"],
                    }
                )
        except Exception as e:
            st.warning(f"Error fetching {source}: {str(e)}")
            continue

    return pd.DataFrame(all_articles)


@st.cache_data(ttl=600)
def analyze_sentiment_batch(texts):
    results = sentiment_analyzer(texts, truncation=True, max_length=512)
    return [-r["score"] if r["label"] == "NEGATIVE" else r["score"] for r in results]


def create_word_cloud(text, mask=None):
    if not text or len(text.strip()) < 10:
        return None
    return WordCloud(
        width=800,
        height=400,
        background_color="white",
        mask=mask,
        contour_width=3,
        contour_color="steelblue",
    ).generate(text)


def extract_key_phrases(text, n=5):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([text])
        importance = np.squeeze(tfidf.toarray())
        feature_names = vectorizer.get_feature_names_out()
        key_phrases = sorted(
            zip(feature_names, importance), key=lambda x: x[1], reverse=True
        )[:n]
        return [phrase for phrase, _ in key_phrases]
    except:
        return []


def main():
    st.set_page_config(layout="wide")

    st.sidebar.title("Configuration")

    with st.sidebar.expander("Analysis Settings", expanded=True):
        categories = st.multiselect(
            "Select Categories", list(RSS_FEEDS.keys()), list(RSS_FEEDS.keys())[:2]
        )
        available_sources = [
            source for cat in categories for source in RSS_FEEDS[cat].keys()
        ]
        sources = st.multiselect(
            "Select Sources", available_sources, available_sources[:2]
        )
        days_back = st.slider("Days to Analyze", 1, 30, 7)
        articles_to_show = st.slider("Articles per Source", 5, 30, 10)

    with st.sidebar.expander("Visual Settings"):
        chart_theme = st.selectbox("Chart Theme", ["plotly", "seaborn"])
        show_wordcloud = st.checkbox("Show Word Clouds", True)
        show_key_phrases = st.checkbox("Show Key Phrases", True)

    st.title("News Sentiment Analysis")

    if st.button("Load and Analyze News", key="load_button"):
        with st.spinner("Fetching and analyzing news..."):
            df = get_news_from_rss(categories, sources, days_back)

            if df.empty:
                st.error("No articles found for the selected criteria")
                return

            sentiments = analyze_sentiment_batch(df["description"].tolist())
            df["sentiment"] = sentiments

            if show_key_phrases:
                df["key_phrases"] = df["description"].apply(extract_key_phrases)

            st.session_state["analyzed_data"] = df

    if "analyzed_data" in st.session_state:
        df = st.session_state["analyzed_data"]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                df.groupby(["date", "source"])["sentiment"].mean().reset_index(),
                x="date",
                y="sentiment",
                color="source",
                title="Sentiment Trends Over Time",
                template=chart_theme,
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                df.groupby("source")["sentiment"].mean().reset_index(),
                x="source",
                y="sentiment",
                title="Average Sentiment by Source",
                template=chart_theme,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df,
                x="category",
                y="sentiment",
                color="category",
                title="Sentiment Distribution by Category",
                template=chart_theme,
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                df,
                x="date",
                y="sentiment",
                color="category",
                title="Sentiment Scatter by Date",
                template=chart_theme,
            )
            st.plotly_chart(fig, use_container_width=True)

        if show_wordcloud or show_key_phrases:
            st.header("Content Analysis")
            cols = st.columns(len(sources))

            for idx, source in enumerate(sources):
                source_df = df[df["source"] == source]
                if not source_df.empty:
                    with cols[idx]:
                        st.subheader(source)

                        if show_wordcloud:
                            source_text = " ".join(source_df["description"])
                            wordcloud = create_word_cloud(source_text)
                            if wordcloud:
                                st.image(wordcloud.to_array())

                        if show_key_phrases:
                            st.caption("Top phrases:")
                            all_phrases = [
                                phrase
                                for phrases in source_df["key_phrases"]
                                for phrase in phrases
                            ]
                            phrase_counts = Counter(all_phrases).most_common(5)
                            for phrase, count in phrase_counts:
                                st.write(f"- {phrase} ({count})")

        st.header("Articles Detail")

        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment_filter = st.select_slider(
                "Filter by Sentiment",
                options=[
                    "Very Negative",
                    "Negative",
                    "Neutral",
                    "Positive",
                    "Very Positive",
                ],
                value=("Very Negative", "Very Positive"),
            )
        with col2:
            date_filter = st.date_input(
                "Date Range", value=(df["date"].min().date(), df["date"].max().date())
            )
        with col3:
            sort_by = st.selectbox("Sort By", ["date", "sentiment"])

        sentiment_ranges = {
            "Very Negative": (-1.0, -0.6),
            "Negative": (-0.6, -0.2),
            "Neutral": (-0.2, 0.2),
            "Positive": (0.2, 0.6),
            "Very Positive": (0.6, 1.0),
        }

        filtered_df = df[
            (
                df["sentiment"].between(
                    sentiment_ranges[sentiment_filter[0]][0],
                    sentiment_ranges[sentiment_filter[1]][1],
                )
            )
            & (df["date"].dt.date.between(date_filter[0], date_filter[1]))
        ]

        st.dataframe(
            filtered_df[["source", "title", "date", "sentiment", "link"]]
            .sort_values(sort_by, ascending=False)
            .head(articles_to_show)
        )


if __name__ == "__main__":
    main()
