
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_pickle("df.pkl")
X_scaled = np.load("X_scaled.npy")
latent_features = np.load("latent_features.npy")
knn = joblib.load("knn_model.pkl")

numeric_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'track_genre_encoded'
]

st.set_page_config(page_title="Spotify Recommender", layout="wide")
st.title("🎧 Music Recommender")

tab1, tab2 = st.tabs(["🎵 Recommendations", "📊 Genre Visualization", ])

def plot_heatmap(df, numeric_features):
    corr = df[numeric_features].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=numeric_features,
        y=numeric_features,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}'
    ))
    fig.update_layout(title='Feature Correlation Heatmap', width=700, height=700)
    return fig

def plot_latent_3d(latent_features, df):
    pca = PCA(n_components=3)
    components = pca.fit_transform(latent_features)
    plot_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
    plot_df['Genre'] = df['track_genre']
    plot_df['Track'] = df['track_name']
    plot_df['Artist'] = df['artists']
    fig = px.scatter_3d(plot_df, x='PC1', y='PC2', z='PC3',
                        color='Genre', hover_data=['Track', 'Artist'],
                        title="3D Genre Visualization")
    return fig

def recommend_tracks(track_index, df, latent_features, knn):
    track_vector = latent_features[track_index].reshape(1, -1)
    distances, indices = knn.kneighbors(track_vector, n_neighbors=30)  # Use a higher number initially
    similar_indices = indices.flatten()[1:]  # Skip the first (it’s the selected track itself)
    
    similar_tracks = df.iloc[similar_indices]
    
    # Drop duplicates based on both track name and artist
    unique_tracks = similar_tracks.drop_duplicates(subset=['track_name', 'artists']).head(9)
    
    return unique_tracks[['track_name', 'artists', 'track_genre']]


with tab1:
    st.subheader("Get Track Recommendations")
    df = df.reset_index(drop=True)

    track_options = [f"{i} - {row['track_name']} by {row['artists']}" for i, row in df.iterrows()]
    selected_option = st.selectbox("Select a Track:", track_options)
    track_index = int(selected_option.split(" - ")[0])

    selected = df.iloc[track_index]
    st.markdown(f"**Selected:** {selected['track_name']} by {selected['artists']} | Genre: *{selected['track_genre']}*")
    
    if 'track_id' in df.columns:
        track_id = selected['track_id']
        components.html(
            f"""
            <iframe src="https://open.spotify.com/embed/track/{track_id}" 
                    width="100%" height="80" frameborder="0" 
                    allowtransparency="true" allow="encrypted-media">
            </iframe>
            """,
            height=100
        )
    else:
        st.warning("Track ID not available - can't play preview")
    
    recs = recommend_tracks(track_index, df, latent_features, knn)
    st.write("**Top Recommendations:**")
    
    st.markdown("""
    <style>
        .dark-card {
            background-color: #000000 !important;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .dark-card iframe {
            border-radius: 5px;
        }
        .track-title {
            color: white !important;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 5px;
            font-size: 14px;
        }
        .track-artist {
            color: #ffffff !important;
            font-size: 12px;
            margin-bottom: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if 'track_id' in df.columns:
        cols = st.columns(2)
        
        for i, (idx, row) in enumerate(recs.iterrows()):
            original_track = df[(df['track_name'] == row['track_name']) & 
                             (df['artists'] == row['artists'])].iloc[0]
            
            with cols[i % 2]:
                components.html(
                    f"""
                    <div class="dark-card">
                        <iframe src="https://open.spotify.com/embed/track/{original_track['track_id']}" 
                                width="100%" height="80" frameborder="0" 
                                allowtransparency="true" allow="encrypted-media">
                        </iframe>
                       
                    </div>
                    """,
                    height=160
                )
    else:
        st.dataframe(recs)

with tab2:
    st.subheader("3D PCA Visualization of Tracks by Genre")
    fig = plot_latent_3d(latent_features, df)
    st.plotly_chart(fig, use_container_width=True)



