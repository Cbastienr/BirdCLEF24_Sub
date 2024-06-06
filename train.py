import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for signals, labels in dataloader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * signals.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

def predict(model, dataloader, device):
    model.eval()  # mode d'évaluation
    predictions = []
    actuals = []
    with torch.no_grad():
        for signals, labels in dataloader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    return predictions, actuals

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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = dataset.num_classes
    cnn = CNNNetwork(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.to(device)

    learning_rate = 0.001
    epochs = 10

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the model
    train_model(cnn, dataloader, criterion, optimizer, device, epochs)

    # Predictions
    predictions, actuals = predict(cnn, dataloader, device)
    print(f"Predictions: {predictions[:10]}")
    print(f"Actuals: {actuals[:10]}")
