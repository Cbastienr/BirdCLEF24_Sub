import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from sklearn.preprocessing import LabelEncoder

class BirdCLEFDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        # Encode les labels
        self.label_encoder = LabelEncoder()
        self.annotations['encoded_label'] = self.label_encoder.fit_transform(self.annotations['primary_label'])
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._normalize_length(signal)
        signal = self.transformation(signal)

        # Convertir le label en tenseur
        label = torch.tensor(label, dtype=torch.long)

        return signal, label

    def _normalize_length(self, signal):
        if signal.shape[1] != self.num_samples:
            if signal.shape[1] > self.num_samples:
                # Cut si trop long
                signal = signal[:, :self.num_samples]
            else:
                # Si trop court
                num_missing_samples = self.num_samples - signal.shape[1]
                last_dim_padding = (0, num_missing_samples)
                signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _get_audio_sample_path(self, index):
        # Assurez-vous que le nom du fichier audio est dans la colonne 'filename'
        filename = self.annotations.iloc[index]['filename']
        if isinstance(filename, int):
            raise ValueError("La dernière colonne contient des entiers. Vérifiez la colonne des noms de fichiers.")
        return os.path.join(self.audio_dir, filename)

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index]['encoded_label']

if __name__ == "__main__":
    ANNOTATIONS_FILE = "path/kaggle"
    AUDIO_DIR = "path/kaggle"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dataset = BirdCLEFDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES)
    print(f"There are {len(dataset)} samples in the dataset.")
    print(f"Number of classes: {dataset.num_classes}")
    signal, label = dataset[0]
    print(f"Signal shape: {signal.shape}, Label: {label}, Label type: {type(label)}")
