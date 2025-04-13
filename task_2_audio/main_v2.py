import argparse
import os
import json
import logging
import numpy as np
from sklearn.cluster import KMeans
import torch
import torchaudio
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)


class AudioFeatureExtractor(nn.Module):
    def __init__(self):
        super(AudioFeatureExtractor, self).__init__()
        # Using a pre-trained ResNet18 model as a feature extractor
        self.resnet = models.resnet18(pretrained=True)
        # Removing the last classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        return self.resnet(x).squeeze()


def load_audio_to_mel(file_path, n_mels=128, n_fft=2048, win_length=1024,
                      hop_length=512):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Create Mel-spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )(waveform)

        # Apply logarithmic transformation
        log_mel = torch.log(mel_spectrogram + 1e-9)

        # Normalization
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-9)

        # Ensure consistent length (pad or crop)
        target_length = 400  # ~ 5 seconds at hop_length=512
        if log_mel.shape[2] > target_length:
            log_mel = log_mel[:, :, :target_length]
        else:
            pad_size = target_length - log_mel.shape[2]
            log_mel = F.pad(log_mel, (0, pad_size))

        # Repeat Mel-spectrogram across 3 channels for compatibility with ResNet
        # ResNet expects input data in the format [batch, channels, height, width]
        # where channels=3 for RGB images
        log_mel = log_mel.repeat(3, 1, 1)

        return log_mel
    except Exception as e:
        logging.warning(f"Error processing {file_path}: {e}")
        return None


def extract_deep_features(file_path, model):
    try:
        mel_spec = load_audio_to_mel(file_path)
        if mel_spec is None:
            return None

        # Add batch dimension
        mel_spec = mel_spec.unsqueeze(0)

        # Extract features using the model
        with torch.no_grad():
            features = model(mel_spec)

        # Convert tensor to numpy array
        features_np = features.cpu().numpy().flatten()

        return features_np
    except Exception as e:
        logging.warning(f"Failed to extract features for {file_path}: {e}")
        return None


def main(path, n_clusters):
    logging.info("Processing started...")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioFeatureExtractor().to(device)
    model.eval()

    logging.info(f"Using device: {device}")

    songs = []
    features = []

    for filename in os.listdir(path):
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            full_path = os.path.join(path, filename)
            logging.info(f"Processing file: {filename}")

            f = extract_deep_features(full_path, model)
            if f is not None:
                songs.append(filename)
                features.append(f)

    if len(features) == 0:
        logging.error("No suitable audio files found.")
        return

    # Convert to NumPy array for KMeans
    features = np.array(features)

    logging.info(
        f"Clustering {features.shape[0]} files with {features.shape[1]} features per file.")

    try:
        logging.info("Performing clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        playlists = [{"id": i, "songs": []} for i in range(n_clusters)]
        for song, label in zip(songs, labels):
            playlists[label]["songs"].append(song)

        output = {"playlists": playlists}
        with open("playlists_v2.json", "w") as f:
            json.dump(output, f, indent=2)

        logging.info("Done. The result is saved in playlists_v2.json")
    except Exception as e:
        logging.error(f"KMeans clustering error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to the folder with audio files")
    parser.add_argument("--n", required=True, type=int, help="Number of clusters/playlists")
    args = parser.parse_args()
    main(args.path, args.n)
