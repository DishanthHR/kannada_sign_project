import torch.nn as nn

class KannadaSignModel(nn.Module):
    def __init__(self, num_classes=28):
        super().__init__()
        
        # Feature extractor (renamed from spatial_encoder to encoder)
        self.encoder = nn.Sequential(
            nn.Conv1d(126, 64, kernel_size=1),  # 2 hands * 21 landmarks * 3 coords
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Temporal processor
        self.lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 256 because bidirectional (128*2)
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: [batch, seq_len, 2, 21, 3]
        batch_size = x.size(0)
        
        # 1. Flatten spatial dimensions
        x = x.view(batch_size, x.size(1), -1)  # [batch, seq_len, 126]
        x = x.permute(0, 2, 1)  # [batch, 126, seq_len] for Conv1d
        
        # 2. Spatial encoding
        x = self.encoder(x)  # [batch, 64, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, 64] for LSTM
        
        # 3. Temporal processing
        x, _ = self.lstm(x)  # [batch, seq_len, 256] (128*2 for bidirectional)
        
        # 4. Classification
        x = x.mean(dim=1)  # Average over time [batch, 256]
        return self.classifier(x)  # [batch, num_classes]