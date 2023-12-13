import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32, l1_value=0, l2_value=0, dropout=0.05, num_classes=10):  # add num_classes to the parameters
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_dim*3, encoding_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_dim*2, encoding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_dim * 2, encoding_dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_dim * 3, input_dim),
            nn.Sigmoid()  # or use ReLU, depending on your input range
        )

        self.dense = nn.Linear(encoding_dim, num_classes)  # add this line


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        class_probs = F.softmax(self.dense(encoded), dim=1)
        return decoded, encoded, class_probs
