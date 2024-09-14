import streamlit as st
import pandas as pd
import gdown
import ast
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Download the data from Google Drive
@st.cache_data
def download_data_from_drive():
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'
    output = 'songTest1.csv'
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

# Load emotion detection model and tokenizer
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-classification", model=model_name, top_k=None)
    return model, tokenizer

# Function to download emotion cache from GitHub or use local cache
def download_emotion_cache():
    cache_url = 'https://raw.githubusercontent.com/your-github-repo/emotion_cache.joblib'  # Your GitHub link here
    cache_file = 'emotion_cache.joblib'
    
    if not os.path.exists(cache_file):
        try:
            gdown.download(cache_url, cache_file, quiet=True)
        except Exception as e:
            st.write(f"Could not download emotion cache: {e}")
            return {}
    
    try:
        return joblib.load(cache_file)
    except Exception as e:
        st.write(f"Error loading emotion cache: {e}")
        return {}

# Detect emotions in the song lyrics
def detect_emotions(lyrics, emotion_model, tokenizer):
    max_length = 512  # Max token length for the model
    try:
        emotions = emotion_model(lyrics[:max_length])
    except Exception as e:
        st.write(f"Error in emotion detection: {e}")
        emotions = []
    return emotions

# Compute similarity between the input song lyrics and all other songs in the dataset
@st.cache_data
def compute_lyrics_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

# Compute emotion similarity between two sets of emotions
def compute_emotion_similarity(emotions1, emotions2):
    if not emotions1 or not emotions2:
        return 0

    emotion_dict1 = {e['label']: e['score'] for e in emotions1}
    emotion_dict2 = {e['label']: e['score'] for e in emotions2}

    all_emotions = set(emotion_dict1.keys()) | set(emotion_dict2.keys())
    vec1 = [emotion_dict1.get(e, 0) for e in all_emotions]
    vec2 = [emotion_dict2.get(e, 0) for e in all_emotions]

    return cosine_similarity([vec1], [vec2])[0][0]

# Function to process emotions with caching
def get_or_detect_emotions(song_title, lyrics, emotion_model, tokenizer, emotion_cache):
    if song_title in emotion_cache:
        return emotion_cache[song_title]
    
    emotions = detect_emotions(lyrics, emotion_model, tokenizer)
    emotion_cache[song_title] = emotions
    return emotions

# Save updated emotion cache to a file
def save_emotion_cache(emotion_cache):
    joblib.dump(emotion_cache, 'emotion_cache.joblib')

# Recommend similar songs based on both lyrics and emotion
def recommend_songs(df, selected_song, top_n=5):
    # Load emotion detection model, tokenizer, and emotion cache
    emotion_model, tokenizer = load_emotion_model()
    emotion_cache = download_emotion_cache()

    # Get the selected song's lyrics and emotions
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []

    song_lyrics = song_data['Lyrics'].values[0]
    selected_song_emotions = get_or_detect_emotions(selected_song, song_lyrics, emotion_model, tokenizer, emotion_cache)
    st.write(f"### Detected Emotions in {selected_song}:")
    st.write(selected_song_emotions)

    # Compute lyrics similarity
    lyrics_similarity_scores = compute_lyrics_similarity(df, song_lyrics)

    # Compute emotion similarity for all songs
    emotion_similarity_scores = []
    for idx, row in df.iterrows():
        other_song_emotions = get_or_detect_emotions(row['Song Title'], row['Lyrics'], emotion_model, tokenizer, emotion_cache)
        emotion_similarity = compute_emotion_similarity(selected_song_emotions, other_song_emotions)
        emotion_similarity_scores.append(emotion_similarity)

    # Save the updated emotion cache
    save_emotion_cache(emotion_cache)

    # Combine lyrics and emotion similarities (50% weight each)
    df['lyrics_similarity'] = lyrics_similarity_scores
    df['emotion_similarity'] = emotion_similarity_scores
    df['combined_similarity'] = (df['lyrics_similarity'] + df['emotion_similarity']) / 2

    # Recommend top N similar songs
    recommended_songs = df.sort_values(by='combined_similarity', ascending=False).head(top_n + 1)
    recommended_songs = recommended_songs[recommended_songs['Song Title'] != selected_song]

    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'combined_similarity', 'Song URL', 'Media']]

# Main function for the Streamlit app
def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Similarity")
    
    # Download and load dataset
    df = download_data_from_drive()

    # Drop duplicates and convert release dates to datetime
    df = df.drop_duplicates(subset=['Song Title', 'Artist', 'Album', 'Release Date'], keep='first')
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

    # Search bar for song name or artist
    search_term = st.text_input("Enter a Song Name or Artist").strip()

    if search_term:
        filtered_songs = df[
            (df['Song Title'].str.contains(search_term, case=False, na=False)) |
            (df['Artist'].str.contains(search_term, case=False, na=False))
        ]

        if not filtered_songs.empty:
            st.write(f"### Search Results for: {search_term}")
            for idx, row in filtered_songs.iterrows():
                st.write(f"**{idx + 1}. {row['Song Title']}** by {row['Artist']}")
                if st.button(f"Recommend Similar to {row['Song Title']}", key=idx):
                    selected_song = row['Song Title']
                    recommendations = recommend_songs(df, selected_song)
                    st.write(f"### Recommended Songs Similar to {selected_song}:")
                    for rec_idx, rec_row in recommendations.iterrows():
                        st.write(f"**{rec_idx + 1}. {rec_row['Song Title']}** by {rec_row['Artist']}")
                        st.write(f"**Similarity Score:** {rec_row['combined_similarity']:.2f}")
        else:
            st.write("No songs found matching the search term.")
    else:
        st.write("Please enter a song name or artist to search.")

if __name__ == '__main__':
    main()
