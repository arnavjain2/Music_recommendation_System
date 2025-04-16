import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
# data preprocessing and data analysis(outlier detection) 
def load_and_preprocess_data():

    ds = load_dataset("maharshipandya/spotify-tracks-dataset")
    df = pd.DataFrame(ds["train"])
    
    print("✅ Dataset loaded. First few rows:")
    print(df.head(), "\n")

    le = LabelEncoder()
    df['track_genre_encoded'] = le.fit_transform(df['track_genre'])

    numeric_features = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'track_genre_encoded' 
    ]
    
    df = df.dropna(subset=numeric_features).reset_index(drop=True)

    X = df[numeric_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    


    return df, X_scaled, numeric_features


def build_autoencoder(input_dim, encoding_dim=5):

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def train_autoencoder(autoencoder, X_scaled, epochs=5, batch_size=32):

    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    return history

def evaluate_autoencoder(autoencoder, X_scaled):

    X_reconstructed = autoencoder.predict(X_scaled)
    mse = mean_squared_error(X_scaled, X_reconstructed)
    r2 = r2_score(X_scaled, X_reconstructed)
    
    print(f"\n🔹 Autoencoder Accuracy Metrics:")
    print(f"   ✅ Mean Squared Error (MSE): {mse:.5f}")
    print(f"   ✅ R² Score: {r2:.5f}")

def build_knn_model(latent_features, metric='cosine'):

    knn = NearestNeighbors(metric=metric, algorithm='brute')
    knn.fit(latent_features)
    return knn

def get_recommendations(knn, latent_features, df, track_index, n_recommendations=5):

    track_vector = latent_features[track_index].reshape(1, -1)
    distances, indices = knn.kneighbors(track_vector, n_neighbors=n_recommendations + 1)
    similar_indices = indices.flatten()[1:] 
    return df.iloc[similar_indices]

import plotly.express as px
import plotly.graph_objects as go

def plot_tracks_by_genre(latent_features, df):

    pca = PCA(n_components=3)
    latent_3d = pca.fit_transform(latent_features)
    
    plot_df = pd.DataFrame(
        latent_3d, 
        columns=['PC1', 'PC2', 'PC3']
    )
    plot_df['Genre'] = df['track_genre']
    plot_df['Track'] = df['track_name']
    plot_df['Artist'] = df['artists']
    
    fig = px.scatter_3d(
        plot_df, 
        x='PC1', 
        y='PC2', 
        z='PC3',
        color='Genre',
        hover_data=['Track', 'Artist'],
        title='3D Interactive Visualization of Tracks by Genre',
        labels={'PC1': 'First Principal Component',
                'PC2': 'Second Principal Component',
                'PC3': 'Third Principal Component'}
    )
    
    fig.update_layout(
        scene = dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.85
        )
    )
    
    fig.show()

    genre_dist = df['track_genre'].value_counts()
    print("\n🎵 Genre Distribution:")
    for genre, count in genre_dist.items():
        print(f"{genre}: {count} tracks")

def plot_feature_heatmap(df, numeric_features):
    # Calculate correlation matrix
    correlation_matrix = df[numeric_features].corr()
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=numeric_features,
        y=numeric_features,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        width=900,
        height=900,
        xaxis_tickangle=-45
    )
    
    fig.show()
    
    print("\n🔍 Strongest Feature Correlations:")
    correlations = []
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            corr = correlation_matrix.iloc[i,j]
            if abs(corr) > 0.3: 
                correlations.append((
                    numeric_features[i],
                    numeric_features[j],
                    corr
                ))
    
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for feat1, feat2, corr in correlations:
        print(f"{feat1} ↔️ {feat2}: {corr:.3f}")
        
        
def get_track_recommendations(track_index, n_recommendations=5):

    print("\n🎵 Selected Track:")
    selected_track = df.iloc[track_index]
    print(f"Track: {selected_track['track_name']}")
    print(f"Artist: {selected_track['artists']}")
    print(f"Genre: {selected_track['track_genre']}")

    track_vector = latent_features[track_index].reshape(1, -1)
    distances, indices = knn.kneighbors(track_vector, n_neighbors=20)  
    similar_indices = indices.flatten()[1:]  
    similar_tracks = df.iloc[similar_indices]

    selected_genre = selected_track['track_genre']
    filtered_tracks = similar_tracks[similar_tracks['track_genre'] != selected_genre]

    recommendations = similar_tracks.head(n_recommendations)

    print("\n🎶 Recommended Tracks (Different Genre):")
    return recommendations[['track_name', 'artists', 'track_genre']]

def main():
    global df, latent_features, knn
    
    df, X_scaled, numeric_features = load_and_preprocess_data()
    plot_feature_heatmap(df, numeric_features)
    input_dim = X_scaled.shape[1]  
    encoding_dim = 5               
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    
    print("🚀 Training autoencoder...")
    train_autoencoder(autoencoder, X_scaled, epochs=5, batch_size=32)
    
    evaluate_autoencoder(autoencoder, X_scaled)

    print("\n🔍 Generating latent representations...")
    latent_features = encoder.predict(X_scaled)
    knn = build_knn_model(latent_features, metric='cosine')
    
    print("\n📊 Generating genre distribution visualization...")
    plot_tracks_by_genre(latent_features, df)
    
    sample_track_index = 15
    recommendations = get_track_recommendations(sample_track_index)
    print(recommendations)

if __name__ == "__main__":
    main()