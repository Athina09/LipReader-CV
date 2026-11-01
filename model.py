import torch.nn as nn


def block(cin, cout, pool):
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(pool),
    )


class LipNet3D(nn.Module):
    """Small 3D CNN: (B, 1, T, H, W) -> (B, num_classes).

    Spatial size is halved at every stage; time is only pooled once, so the
    model keeps fine temporal detail (lip open/close).
    """

    def __init__(self, num_classes, width=32, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),  # early downsampling keeps the 3D convs cheap
        )
        self.features = nn.Sequential(
            block(width, width * 2, (1, 2, 2)),
            block(width * 2, width * 4, (2, 2, 2)),
            block(width * 4, width * 8, (1, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 8, num_classes))

    def forward(self, x):
        x = self.features(self.stem(x))
        return self.head(self.pool(x).flatten(1))
