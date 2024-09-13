import streamlit as st
import pandas as pd
import gdown
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the data from Google Drive
@st.cache_data
def download_data_from_drive():
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'
    output = 'songTest1.csv'
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

# Load emotion detection model
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Detect emotions in the song lyrics
def detect_emotions(lyrics, emotion_model):
    max_length = 512
    truncated_lyrics = ' '.join(lyrics.split()[:max_length])
    try:
        emotions = emotion_model(truncated_lyrics)
    except Exception as e:
        st.write(f"Error in emotion detection: {e}")
        emotions = []
    return emotions

# Preprocess the dataset to detect and store emotions for all songs
@st.cache_data
def preprocess_data(_emotion_model, df):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    df['Emotions'] = df['Lyrics'].apply(lambda x: detect_emotions(x, _emotion_model))
    return df

# Filter songs based on emotion similarity
def filter_by_emotion(emotion_data, target_emotion, threshold=0.5):
    similar_songs = []
    for song_emotion in emotion_data:
        similarity_score = 0
        for emotion in song_emotion:
            for target in target_emotion:
                if emotion['label'] == target['label']:
                    similarity_score += min(emotion['score'], target['score'])
        if similarity_score >= threshold:
            similar_songs.append(True)
        else:
            similar_songs.append(False)
    return similar_songs

# Compute similarity between the input song lyrics and all other songs in the dataset
@st.cache_data
def compute_similarity(df, song_lyrics):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

# Recommend songs based on both emotion and lyrics similarity
def recommend_songs(df, selected_song, emotion_model, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    
    song_lyrics = song_data['Lyrics'].values[0]

    # Detect emotions of the selected song
    song_emotions = detect_emotions(song_lyrics, emotion_model)
    st.write(f"### Detected Emotions in {selected_song}:")
    st.write(song_emotions)

    # Filter by emotions
    emotion_filtered_songs = df[filter_by_emotion(df['Emotions'], song_emotions)]

    # Compute lyrics similarity among emotion-matching songs
    similarity_scores = compute_similarity(emotion_filtered_songs, song_lyrics)

    # Recommend top N similar songs based on lyrics similarity
    emotion_filtered_songs['similarity'] = similarity_scores
    recommended_songs = emotion_filtered_songs.sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'similarity', 'Song URL']]

# Main function for the Streamlit app
def main():
    st.title("Song Recommender System Based on Emotion and Lyrics Similarity")
    
    # Load and preprocess data
    df = download_data_from_drive()
    emotion_model = load_emotion_model()
    df = preprocess_data(emotion_model, df)
    
    # Search bar for song name
    search_term = st.text_input("Enter a Song Name").strip()

    if search_term:
        filtered_songs = df[df['Song Title'].str.contains(search_term, case=False, na=False)]

        filtered_songs['Release Date'] = pd.to_datetime(filtered_songs['Release Date'], errors='coerce')
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
                    st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
                    
                    # Display link to Genius.com page if URL is available
                    song_url = row.get('Song URL', '')
                    if pd.notna(song_url) and song_url:
                        st.markdown(f"[View Lyrics on Genius]({song_url})")

                    with st.expander("Show/Hide Lyrics"):
                        formatted_lyrics = row['Lyrics'].strip().replace('\n', '\n\n')
                        st.markdown(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{formatted_lyrics}</pre>", unsafe_allow_html=True)
                    st.markdown("---")

            song_list = filtered_songs['Song Title'].unique()
            selected_song = st.selectbox("Select a Song for Recommendations", song_list)

            if st.button("Recommend Similar Songs"):
                recommendations = recommend_songs(df, selected_song, emotion_model)
                st.write(f"### Recommended Songs Similar to {selected_song}")
                st.write(recommendations)
    else:
        st.write("Please enter a song name to search.")

if __name__ == '__main__':
    main()
