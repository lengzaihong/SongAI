import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load song dataset
@st.cache_data
def load_data():
    df = pd.read_csv('/mnt/data/all_songs_data_processed.csv')
    return df

# Emotion Detection using pre-trained NLP model
@st.cache_data
def detect_emotions(lyrics):
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    emotions = emotion_pipeline(lyrics)
    return emotions

# TF-IDF Vectorizer for Lyrics Similarity
@st.cache_data
def compute_similarity(df, song_lyrics):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

# Recommend Songs based on emotion and category
def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['song_name'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []

    song_lyrics = song_data['lyrics'].values[0]
    song_category = song_data['category'].values[0]

    # Detect emotion of the input song
    song_emotion = detect_emotions(song_lyrics)
    
    # Calculate song similarity based on lyrics
    similarity_scores = compute_similarity(df, song_lyrics)
    
    # Filter songs based on emotion and category similarity
    df['similarity'] = similarity_scores
    recommended_songs = df[(df['category'] == song_category)].sort_values(by='similarity', ascending=False).head(top_n)

    return recommended_songs[['song_name', 'artist_name', 'category', 'similarity']]

# Streamlit App
def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Category")
    
    # Load the dataset
    df = load_data()

    # Song selection from the dataset
    song_list = df['song_name'].unique()
    selected_song = st.selectbox("Select a Song", song_list)
    
    if st.button("Recommend Similar Songs"):
        # Recommend songs based on emotion and category
        recommendations = recommend_songs(df, selected_song)
        
        st.write("### Recommended Songs")
        st.dataframe(recommendations)

if __name__ == '__main__':
    main()
