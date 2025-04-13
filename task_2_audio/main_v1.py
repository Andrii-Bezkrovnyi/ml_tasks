import argparse
import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
import json
import logging

logging.basicConfig(level=logging.INFO)


def extract_features(file_path, target_length=14):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features = np.mean(mfcc, axis=1).tolist() + [tempo]

        # Convert numpy.array elements to regular numbers
        features = [f[0] if isinstance(f, np.ndarray) else f for f in features]

        # Check feature length
        logging.info(f"Extracted {len(features)} features for {file_path}: {features}")

        if len(features) < target_length:
            features.extend([0] * (target_length - len(features)))  # Padding
        elif len(features) > target_length:
            features = features[:target_length]  # Truncating

        return features
    except Exception as e:
        logging.warning(f"Could not process {file_path}: {e}")
        return None


def main(path, n_clusters):
    logging.info("Start processing...")
    songs = []
    features = []

    for filename in os.listdir(path):
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            full_path = os.path.join(path, filename)
            f = extract_features(full_path)
            if f:
                songs.append(filename)
                features.append(f)

    if len(features) == 0:
        logging.error("No valid audio files found.")
        return

    # Convert to NumPy array for KMeans
    features = np.array(features)

    logging.info(
        f"Clustering with {features.shape[0]} files and {features.shape[1]} features per file.")

    try:
        logging.info("Clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        playlists = [{"id": i, "songs": []} for i in range(n_clusters)]
        for song, label in zip(songs, labels):
            playlists[label]["songs"].append(song)

        output = {"playlists": playlists}
        with open("playlists_v1.json", "w") as f:
            json.dump(output, f, indent=2)

        logging.info("Done. Output saved to playlists_v1.json")
    except Exception as e:
        logging.error(f"KMeans clustering failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to folder with audio files")
    parser.add_argument("--n", required=True, type=int,
                        help="Number of clusters/playlists")
    args = parser.parse_args()
    main(args.path, args.n)
