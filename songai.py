import streamlit as st
import pandas as pd
import gdown
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to download the CSV from Google Drive
@st.cache_data
def download_data_from_drive():
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'  # Replace with actual file ID if necessary
    output = 'songTest1.csv'  # Output filename
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

# Genre keyword-based prediction
genre_keywords = {
    'Rock': ['rock', 'guitar', 'band', 'drums'],
    'Pop': ['love', 'dance', 'hit', 'baby'],
    'Jazz': ['jazz', 'swing', 'blues', 'saxophone'],
    'Country': ['country', 'truck', 'road', 'cowboy'],
    'Hip Hop': ['rap', 'hip', 'hop', 'beat', 'flow'],
    'Classical': ['symphony', 'orchestra', 'classical', 'concerto']
}

def predict_genre(row):
    for genre, keywords in genre_keywords.items():
        text = f"{row['Song Title']} {row['Lyrics']}"  # Combine title and lyrics
        if any(keyword.lower() in str(text).lower() for keyword in keywords):
            return genre
    return 'Unknown'

# Remove cache from the emotion detection model
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Emotion Detection (do not cache this function)
def detect_emotions(lyrics, emotion_model):
    emotions = emotion_model(lyrics)
    # Extract the emotion with the highest score
    dominant_emotion = max(emotions[0], key=lambda x: x['score'])['label']
    return dominant_emotion

# TF-IDF Vectorizer for Lyrics Similarity
@st.cache_data
def compute_similarity(df, song_lyrics):
    # Ensure that all lyrics are strings and replace NaN values with empty strings
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

# Recommend songs based on lyrics similarity, genre, and emotion
def recommend_songs(df, selected_song, selected_genre, selected_emotion, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []

    song_lyrics = song_data['Lyrics'].values[0]
    song_genre = song_data['Predicted Genre'].values[0]

    # Detect emotion of the input song
    emotion_model = load_emotion_model()  # Load model without caching
    song_emotion = detect_emotions(song_lyrics, emotion_model)

    # Calculate song similarity based on lyrics
    similarity_scores = compute_similarity(df, song_lyrics)
    
    # Filter songs based on genre, emotion, and similarity
    df['similarity'] = similarity_scores
    recommended_songs = df[
        (df['Predicted Genre'] == selected_genre) &
        (df['Predicted Emotion'] == selected_emotion)
    ].sort_values(by='similarity', ascending=False).head(top_n)

    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'Predicted Genre', 'Predicted Emotion', 'similarity']]

# Streamlit App
def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Genre")

    # Load the dataset
    df = download_data_from_drive()

    # Predict genres for all songs
    df['Predicted Genre'] = df.apply(predict_genre, axis=1)

    # Predict emotions for all songs
    emotion_model = load_emotion_model()
    df['Predicted Emotion'] = df['Lyrics'].apply(lambda lyrics: detect_emotions(lyrics, emotion_model))

    # Sidebar with selection options
    st.sidebar.header("Filter Songs")
    
    # Dropdown for genre selection
    genre_options = ['Select a genre'] + list(df['Predicted Genre'].unique())
    selected_genre = st.sidebar.selectbox("Select Genre", genre_options)

    # Dropdown for emotion selection
    emotion_options = ['Select an emotion'] + list(df['Predicted Emotion'].unique())
    selected_emotion = st.sidebar.selectbox("Select Emotion", emotion_options)

    # Song selection from the dataset
    song_list = df['Song Title'].unique()
    selected_song = st.selectbox("Select a Song", song_list)

    if st.button("Recommend Similar Songs"):
        if selected_genre != 'Select a genre' and selected_emotion != 'Select an emotion':
            # Recommend songs based on the selected song, genre, and emotion
            recommendations = recommend_songs(df, selected_song, selected_genre, selected_emotion)
            
            st.write(f"### Recommended Songs Similar to {selected_song} with Genre: {selected_genre} and Emotion: {selected_emotion}")
            st.write(recommendations)
        else:
            st.write("Please select both a genre and an emotion.")

if __name__ == '__main__':
    main()
