import streamlit as st
import pandas as pd
import gdown
import ast
import random
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the data from Google Drive
@st.cache_data
def download_data_from_drive():
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'
    output = 'songTest1.csv'
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

# Load emotion detection model and tokenizer
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-classification", model=model_name, top_k=None)
    return model, tokenizer

# Detect emotions in the song lyrics
def detect_emotions(lyrics, emotion_model, tokenizer):
    max_length = 512  # Max token length for the model
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
    """Extract the YouTube URL from the Media field."""
    try:
        media_list = ast.literal_eval(media_str)  # Safely evaluate the string to a list
        for media in media_list:
            if media.get('provider') == 'youtube':
                return media.get('url')
    except (ValueError, SyntaxError):
        return None

# Recommend similar songs based on lyrics and detected emotions
def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    
    song_lyrics = song_data['Lyrics'].values[0]

    # Load emotion detection model and tokenizer
    emotion_model, tokenizer = load_emotion_model()

    # Detect emotions in the selected song
    emotions = detect_emotions(song_lyrics, emotion_model, tokenizer)
    st.write(f"### Detected Emotions in {selected_song}:")
    
    if emotions and len(emotions) > 0:
        # Extract the emotions list from the first item
        emotion_list = emotions[0]
        
        # Find the emotion with the highest score
        if isinstance(emotion_list, list) and len(emotion_list) > 0:
            top_emotion = max(emotion_list, key=lambda x: x['score'])
            emotion_sentence = f"The emotion of the song is **{top_emotion['label']}**."
        else:
            emotion_sentence = "No emotions detected."
        
        st.write(emotion_sentence)
    else:
        st.write("No emotions detected.")

    # Compute lyrics similarity
    similarity_scores = compute_similarity(df, song_lyrics)

    # Add similarity scores to the dataframe
    df['similarity'] = similarity_scores

    # Exclude the selected song from recommendations
    df = df[df['Song Title'] != selected_song]

    # Recommend top N similar songs
    recommended_songs = df.sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'similarity', 'Song URL', 'Media']]

def display_random_songs(df, n=5):
    random_songs = df.sample(n=n)
    st.write("### Discover Songs:")
    for idx, row in random_songs.iterrows():
        youtube_url = extract_youtube_url(row.get('Media', ''))
        if youtube_url:
            # If a YouTube URL is available, make the song title a clickable hyperlink
            song_title = f"<a href='{youtube_url}' target='_blank' style='color: #FA8072; font-weight: bold; font-size: 1.2rem;'>{row['Song Title']}</a>"
        else:
            # If no YouTube URL, just display the song title
            song_title = f"<span style='font-weight: bold; font-size: 1.2rem;'>{row['Song Title']}</span>"

        with st.container():
            st.markdown(song_title, unsafe_allow_html=True)
            st.markdown(f"**Artist:** {row['Artist']}")
            st.markdown(f"**Album:** {row['Album']}")
            st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
            st.markdown("---")

def main():
    # Add custom CSS to change the background image
    st.markdown(
        """
        <style>
        .main {
            background-image: url('data:image/jpeg;base64,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');
            background-size: cover;
            background-position: center;
        }

        .main::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.8); /* White overlay with 50% opacity */
        z-index: 0; /* Make sure the overlay sits behind the content */
        }
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            color: black;
            font-weight: 700;
            text-align: center;
        }
        .stButton>button {
            background-color: #fa8072;
            color: white;
            border-radius: 10px;
        }
        .stTextInput input {
            border: 1px solid #fa8072;
            padding: 0.5rem;
        }
        .stTextInput label {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🎵 Song Recommender System 🎶")
    df = download_data_from_drive()

    # Drop duplicate entries based on 'Song Title', 'Artist', 'Album', and 'Release Date'
    df = df.drop_duplicates(subset=['Song Title', 'Artist', 'Album', 'Release Date'], keep='first')

    # Convert the 'Release Date' column to datetime if possible
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

    # Search bar for song name or artist
    search_term = st.text_input("Search for a Song or Artist 🎤").strip()

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
                    st.markdown(f"*Artist:* {row['Artist']}")
                    st.markdown(f"*Album:* {row['Album']}")

                    if pd.notna(row['Release Date']):
                        st.markdown(f"*Release Date:* {row['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"*Release Date:* Unknown")

                    song_url = row.get('Song URL', '')
                    if pd.notna(song_url) and song_url:
                        st.markdown(f"[View Lyrics on Genius]({song_url})")

                    youtube_url = extract_youtube_url(row.get('Media', ''))
                    if youtube_url:
                        video_id = youtube_url.split('watch?v=')[-1]
                        st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                    with st.expander("Show/Hide Lyrics"):
                        formatted_lyrics = row['Lyrics'].strip().replace('\n', '\n\n')
                        st.markdown(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{formatted_lyrics}</pre>", unsafe_allow_html=True)
                    st.markdown("---")

            song_list = filtered_songs['Song Title'].unique()
            selected_song = st.selectbox("Select a Song for Recommendations 🎧", song_list)

            if st.button("Recommend Similar Songs"):
                recommendations = recommend_songs(df, selected_song)
                st.write(f"### Recommended Songs Similar to {selected_song}")
                
                for idx, row in enumerate(recommendations.iterrows(), 1):
                    st.markdown(f"<h2 style='font-weight: bold;'> {idx}. {row[1]['Song Title']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"*Artist:* {row[1]['Artist']}")
                    st.markdown(f"*Album:* {row[1]['Album']}")

                    if pd.notna(row[1]['Release Date']):
                        st.markdown(f"*Release Date:* {row[1]['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"*Release Date:* Unknown")

                    st.markdown(f"*Similarity Score:* {row[1]['similarity']:.2f}")

                    youtube_url = extract_youtube_url(row[1].get('Media', ''))
                    if youtube_url:
                        video_id = youtube_url.split('watch?v=')[-1]
                        st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                    st.markdown("---")
    else:
        # Display random songs if no search term is provided
        display_random_songs(df)

if __name__ == '__main__':
    main()
