import streamlit as st
import pandas as pd
import requests
import xmltodict
from transformers import pipeline
import torch
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud
from dateutil import parser

device = "cuda" if torch.cuda.is_available() else "cpu"
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    batch_size=32,
)

RSS_FEEDS = {
    "BBC Tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "NY Times Tech": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "Guardian Tech": "https://www.theguardian.com/technology/rss",
    "Wired": "https://www.wired.com/feed/rss",
    "TechCrunch": "https://techcrunch.com/feed/",
    "NASA": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "Ars Technica": "http://feeds.arstechnica.com/arstechnica/index",
}


@st.cache_data(ttl=600)
def get_news_from_rss(sources=None):
    all_articles = []
    sources = sources or RSS_FEEDS.keys()

    for source in sources:
        try:
            response = requests.get(RSS_FEEDS[source], timeout=5)
            data = xmltodict.parse(response.content)
            articles = data["rss"]["channel"]["item"][:20]

            for article in articles:
                description = article.get("description", "")
                if not description or len(description.strip()) < 10:
                    continue

                try:
                    parsed_date = parser.parse(article["pubDate"])
                except:
                    parsed_date = datetime.now()

                all_articles.append(
                    {
                        "source": source,
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


def create_word_cloud(text):
    if not text or len(text.strip()) < 10:
        return None
    return WordCloud(width=800, height=400, background_color="white").generate(text)


def main():
    st.title("News Sentiment Analysis")

    sources = st.multiselect(
        "Select Sources", list(RSS_FEEDS.keys()), list(RSS_FEEDS.keys())[:2]
    )
    articles_to_show = st.slider("Articles per Source", 5, 20, 10)

    if st.button("Load News"):
        with st.spinner("Fetching news..."):
            df = get_news_from_rss(sources)
            st.session_state["news_data"] = df

    if "news_data" in st.session_state:
        df = st.session_state["news_data"]

        if st.button("Analyze"):
            with st.spinner("Analyzing sentiments..."):
                sentiments = analyze_sentiment_batch(df["description"].tolist())
                df["sentiment"] = sentiments
                st.session_state["analyzed_data"] = df

        if "analyzed_data" in st.session_state:
            df = st.session_state["analyzed_data"]

            st.plotly_chart(
                px.line(
                    df.groupby(["date", "source"])["sentiment"].mean().reset_index(),
                    x="date",
                    y="sentiment",
                    color="source",
                    title="Sentiment Over Time",
                )
            )

            st.plotly_chart(
                px.bar(
                    df.groupby("source")["sentiment"].mean().reset_index(),
                    x="source",
                    y="sentiment",
                    title="Average Sentiment by Source",
                )
            )

            if st.button("Generate Word Clouds"):
                cols = st.columns(2)
                for idx, source in enumerate(sources):
                    source_df = df[df["source"] == source]
                    if not source_df.empty:
                        cols[idx % 2].subheader(source)
                        source_text = " ".join(source_df["description"])
                        wordcloud = create_word_cloud(source_text)
                        if wordcloud:
                            cols[idx % 2].image(wordcloud.to_array())
                        else:
                            cols[idx % 2].warning(f"Not enough text data for {source}")

            st.dataframe(
                df[["source", "title", "date", "sentiment", "link"]]
                .sort_values("date", ascending=False)
                .head(articles_to_show)
            )


if __name__ == "__main__":
    main()
