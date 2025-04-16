import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load and preprocess
ds = load_dataset("maharshipandya/spotify-tracks-dataset")
df = pd.DataFrame(ds["train"])

le = LabelEncoder()
df['track_genre_encoded'] = le.fit_transform(df['track_genre'])

numeric_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'track_genre_encoded'
]

df = df.dropna(subset=numeric_features).reset_index(drop=True)
for feature in numeric_features[:-1]:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))]

X = df[numeric_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Autoencoder
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)

encoded = Dense(32, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)

encoded = Dense(8, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dropout(0.2)(decoded)

decoded = Dense(64, activation='relu')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dropout(0.2)(decoded)

decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train with history tracking
history = autoencoder.fit(
    X_scaled, X_scaled, 
    epochs=50, 
    batch_size=128, 
    validation_split=0.2,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3)
    ]
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training History')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')
plt.close()

# Calculate metrics
predictions = autoencoder.predict(X_scaled)
mse = mean_squared_error(X_scaled, predictions)
r2 = r2_score(X_scaled, predictions)

print(f"\nModel Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save metrics to file
with open("model_metrics.txt", "w") as f:
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")

# Latent features and KNN
latent_features = encoder.predict(X_scaled)
knn = NearestNeighbors(metric='cosine').fit(latent_features)

# Save artifacts
df.to_pickle("df.pkl")
np.save("X_scaled.npy", X_scaled)
np.save("latent_features.npy", latent_features)
joblib.dump(knn, "knn_model.pkl")

print("\nTraining completed and artifacts saved!")