"""Exact CNN class from the submitted training notebook."""
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Flatten(),

            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

