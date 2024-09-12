import streamlit as st
import pandas as pd
import gdown
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ... (previous code remains the same) ...

def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Genre")
    df = download_data_from_drive()
    df['Predicted Genre'] = df.apply(predict_genre, axis=1)
    
    st.write("Original Dataset with Predicted Genres:")
    st.write(df.head())
    
    # Add genre selection
    genres = ['All'] + list(df['Predicted Genre'].unique())
    selected_genre = st.selectbox("Select a Genre", genres)
    
    if selected_genre != 'All':
        filtered_songs = df[df['Predicted Genre'] == selected_genre]
    else:
        filtered_songs = df
    
    # Sort the filtered songs by 'Release Date' in descending order
    filtered_songs['Release Date'] = pd.to_datetime(filtered_songs['Release Date'], errors='coerce')
    filtered_songs = filtered_songs.sort_values(by='Release Date', ascending=False).reset_index(drop=True)
    
    # Display each song in a banner format with an expander to show/hide lyrics
    st.write(f"### Songs Filtered by Genre: {selected_genre}")
    for idx, row in filtered_songs.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{idx + 1}. {row['Song Title']}**")
                st.markdown(f"*Artist:* {row['Artist']}")
                st.markdown(f"*Album:* {row['Album']}")
                st.markdown(f"*Release Date:* {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
            with col2:
                if st.button(f"View Lyrics #{idx}"):
                    st.markdown("**Lyrics:**")
                    st.text(row['Lyrics'].strip())
            st.markdown("---")
    
    # Song recommendation
    st.header("Get Song Recommendations")
    song_list = df['Song Title'].unique()
    selected_song = st.selectbox("Select a Song for Recommendations", song_list)
    
    if st.button("Recommend Similar Songs"):
        recommendations = recommend_songs(df, selected_song)
        st.write(f"### Recommended Songs Similar to {selected_song}")
        st.write(recommendations)

if __name__ == '__main__':
    main()
