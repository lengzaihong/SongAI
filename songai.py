import streamlit as st
import pandas as pd
import gdown
import ast
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
            emotion_sentence = f"The emotion of the song is {top_emotion['label']}."
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
                    
                    # Check if 'Release Date' is a datetime object before formatting
                    if pd.notna(row['Release Date']):
                        st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"**Release Date:** Unknown")
                    
                    # Display link to Genius.com page if URL is available
                    song_url = row.get('Song URL', '')
                    if pd.notna(song_url) and song_url:
                        st.markdown(f"[View Lyrics on Genius]({song_url})")

                    # Extract and display YouTube video if URL is available
                    youtube_url = extract_youtube_url(row.get('Media', ''))
                    if youtube_url:
                        video_id = youtube_url.split('watch?v=')[-1]
                        st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                    with st.expander("Show/Hide Lyrics"):
                        formatted_lyrics = row['Lyrics'].strip().replace('\n', '\n\n')
                        st.markdown(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{formatted_lyrics}</pre>", unsafe_allow_html=True)
                    st.markdown("---")

            song_list = filtered_songs['Song Title'].unique()
            selected_song = st.selectbox("Select a Song for Recommendations", song_list)

            if st.button("Recommend Similar Songs"):
                recommendations = recommend_songs(df, selected_song)
                st.write(f"### Recommended Songs Similar to {selected_song}")
                for idx, row in recommendations.iterrows():
                    st.markdown(f"**No. {idx + 1}: {row['Song Title']}**")
                    st.markdown(f"**Artist:** {row['Artist']}")
                    st.markdown(f"**Album:** {row['Album']}")
                    
                    # Check if 'Release Date' is a datetime object before formatting
                    if pd.notna(row['Release Date']):
                        st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"**Release Date:** Unknown")
                    
                    st.markdown(f"**Similarity Score:** {row['similarity']:.2f}")
                    
                    # Extract and display YouTube video if URL is available
                    youtube_url = extract_youtube_url(row.get('Media', ''))
                    if youtube_url:
                        video_id = youtube_url.split('watch?v=')[-1]
                        st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                    st.markdown("---")

    else:
        st.write("Please enter a song name or artist to search.")

if __name__ == '__main__':
    main()
