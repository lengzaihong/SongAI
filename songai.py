import streamlit as st
import pandas as pd
import gdown
import ast
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import os
import requests

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
    cache_url = 'https://github.com/lengzaihong/SongAI/blob/main/emotion_cache.joblib'
    cache_file = 'emotion_cache.joblib'
    
    # Attempt to download the cache file if not present locally
    if not os.path.exists(cache_file):
        try:
            response = requests.get(cache_url)
            with open(cache_file, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            st.write(f"Could not download emotion cache: {e}")
            return {}  # Return an empty cache if download fails
    
    # Attempt to load the cache, with error handling for corrupted files
    try:
        return joblib.load(cache_file)
    except Exception as e:
        st.write(f"Error loading emotion cache: {e}")
        st.write("Reinitializing the cache...")
        return {}  # Return an empty cache if loading fails

# Cache for storing emotion detection results
emotion_cache = download_emotion_cache()

# Detect emotions in the song lyrics
def detect_emotions(lyrics, emotion_model, tokenizer):
    if pd.isna(lyrics) or not isinstance(lyrics, str):
        return []
    
    max_length = 512
    inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, max_length=max_length)
    
    try:
        emotions = emotion_model(lyrics[:tokenizer.model_max_length])
    except Exception as e:
        st.write(f"Error in emotion detection: {e}")
        emotions = []
    return emotions

# Compute similarity between the input song lyrics and all other songs in the dataset
@st.cache_data
def compute_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

def extract_youtube_url(media_str):
    try:
        media_list = ast.literal_eval(media_str)
        for media in media_list:
            if media.get('provider') == 'youtube':
                return media.get('url')
    except (ValueError, SyntaxError):
        return None

# Compute emotion similarity between two sets of emotions
def compute_emotion_similarity(emotions1, emotions2):
    if not emotions1 or not emotions2:
        return 0
    
    emotion_dict1 = {e['label']: e['score'] for e in emotions1[0]}
    emotion_dict2 = {e['label']: e['score'] for e in emotions2[0]}
    
    all_emotions = set(emotion_dict1.keys()) | set(emotion_dict2.keys())
    
    vector1 = [emotion_dict1.get(e, 0) for e in all_emotions]
    vector2 = [emotion_dict2.get(e, 0) for e in all_emotions]
    
    return cosine_similarity([vector1], [vector2])[0][0]

# Function to process a single song with caching
def process_song(args):
    idx, (song_id, lyrics), emotion_model, tokenizer, selected_song_emotions = args
    if song_id in emotion_cache:
        emotions = emotion_cache[song_id]
    else:
        emotions = detect_emotions(lyrics, emotion_model, tokenizer)
        emotion_cache[song_id] = emotions
    
    similarity = compute_emotion_similarity(selected_song_emotions, emotions)
    return idx, similarity

# Save updated emotion cache
def save_emotion_cache():
    joblib.dump(emotion_cache, emotion_cache_file)

def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    
    song_lyrics = song_data['Lyrics'].values[0]

    # Load emotion detection model and tokenizer
    emotion_model, tokenizer = load_emotion_model()

    # Detect emotions in the selected song
    selected_song_emotions = detect_emotions(song_lyrics, emotion_model, tokenizer)
    st.write(f"### Detected Emotions in {selected_song}:")
    st.write(selected_song_emotions)

    # Compute lyrics similarity
    similarity_scores = compute_similarity(df, song_lyrics)

    # Compute emotion similarity for all songs with progress bar
    total_songs = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Parallel processing of songs
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(process_song, (idx, (row['Song Title'], row['Lyrics']), emotion_model, tokenizer, selected_song_emotions)): idx 
                         for idx, row in df.iterrows()}
        
        emotion_similarities = {}
        for future in as_completed(future_to_idx):
            idx, similarity = future.result()
            emotion_similarities[idx] = similarity
            progress = (len(emotion_similarities) + 1) / total_songs
            progress_bar.progress(progress)
            status_text.text(f"Detecting emotions of {len(emotion_similarities) + 1} out of {total_songs} songs")

    # Save updated emotion cache
    save_emotion_cache()

    # Clear the progress bar and status text
    progress_bar.empty()
    status_text.empty()

    # Combine lyrics similarity and emotion similarity
    df['emotion_similarity'] = df.index.map(emotion_similarities)
    df['combined_similarity'] = (similarity_scores + df['emotion_similarity']) / 2

    # Recommend top N similar songs
    recommended_songs = df.sort_values(by='combined_similarity', ascending=False).head(top_n+1)
    recommended_songs = recommended_songs[recommended_songs['Song Title'] != selected_song]
    
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'combined_similarity', 'Song URL', 'Media']]

# Main function for the Streamlit app
def main():
    # Add custom CSS to change the background image
    st.markdown(
        """
        <style>
        .main {
            background-image: url('https://wallpapercave.com/wp/wp11163687.jpg');
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Song Recommender System Based on Lyrics Emotion and Similarity")
    df = download_data_from_drive()

    # Drop duplicate entries based on 'Song Title', 'Artist', 'Album', and 'Release Date'
    df = df.drop_duplicates(subset=['Song Title', 'Artist', 'Album', 'Release Date'], keep='first')

    # Convert the 'Release Date' column to datetime if possible
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    
    # Search bar for song name or artist
    search_term = st.text_input("Enter a Song Name or Artist").strip()

    if search_term:
        # Filter by song title or artist name
        filtered_songs = df[
            (df['Song Title'].str.contains(search_term, case=False, na=False)) |
            (df['Artist'].str.contains(search_term, case=False, na=False))
        ]

        filtered_songs = filtered_songs.sort_values(by='Release Date', ascending=False).reset_index(drop=True)

        if filtered_songs.empty:
            st.write("No songs found matching the search term.")
        else:
            st.write(f"### Search Results for: {search_term}")
            for idx, row in filtered_songs.iterrows():
                with st.container():
                    st.markdown(f"<h2 style='font-weight: bold;'> {idx + 1}. {row['Song Title']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"**Artist:** {row['Artist']}")
                    st.markdown(f"**Album:** {row['Album']}")
                    
                    # Check for YouTube media and display link if found
                    media_str = row['Media']
                    youtube_url = extract_youtube_url(media_str)
                    if youtube_url:
                        st.video(youtube_url)
                    else:
                        st.markdown("**YouTube Video:** Not Available")

                    # Option to select a song for recommendation
                    if st.button(f"Select {row['Song Title']}", key=idx):
                        selected_song = row['Song Title']
                        recommended_songs = recommend_songs(df, selected_song)

                        st.write(f"### Recommended Songs Based on {selected_song}:")
                        for _, rec_row in recommended_songs.iterrows():
                            st.write(f"**Title:** {rec_row['Song Title']} by {rec_row['Artist']}")
                            youtube_url = extract_youtube_url(rec_row['Media'])
                            if youtube_url:
                                st.video(youtube_url)
                            else:
                                st.write("**YouTube Video:** Not Available")
    else:
        st.write("No search term entered. Please enter a song name or artist to begin.")

if __name__ == '__main__':
    main()
