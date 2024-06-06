import torch
from torch import nn

class CNNNetwork(nn.Module):

    def __init__(self, num_classes):
        super(CNNNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions

if __name__ == "__main__":
    num_classes = 10  # Nombre de classes, à remplacer par le nombre réel de classes dans le dataset
    cnn = CNNNetwork(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.to(device)
    print(cnn)