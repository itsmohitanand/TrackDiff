import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1200)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.reshape(-1, 6,200)
        return x


class TrackGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.initializer = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7*200),
        )

        self.sequence_encoder = nn.LSTM(input_size=7, hidden_size=256, num_layers=1, batch_first=True)
        self.sequence_generator = nn.LSTM(input_size=256, hidden_size=7, num_layers=1, batch_first=True)
        
    def forward(self, x):
        x = self.initializer(x)
        x = x.reshape(-1, 200, 7)
        x, _ = self.sequence_encoder(x)
        x, _ = self.sequence_generator(x)

        sequence = x[:,:,0:6]
        log_probits = torch.sigmoid(x[:,:,6])

        return sequence, log_probits