import torch
from torch import nn


def _flat_dim(features, n_ch, n_times):
    with torch.no_grad():
        return features(torch.zeros(1, 1, n_ch, n_times)).shape[1]


class EEGNet(nn.Module):
    """Lawhern et al. 2018."""

    def __init__(self, n_ch, n_times, n_classes, F1=8, D=2, F2=16, kern_len=64, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            nn.Flatten(),
        )
        self.head = nn.Linear(_flat_dim(self.features, n_ch, n_times), n_classes)

    def forward(self, x):
        return self.head(self.features(x))


class Square(nn.Module):
    def forward(self, x):
        return x * x


class SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    """Schirrmeister et al. 2017."""

    def __init__(self, n_ch, n_times, n_classes, n_filters=40, kern_len=25, pool_len=75,
                 pool_stride=15, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, kern_len)),
            nn.Conv2d(n_filters, n_filters, (n_ch, 1), bias=False),
            nn.BatchNorm2d(n_filters), Square(),
            nn.AvgPool2d((1, pool_len), (1, pool_stride)), SafeLog(), nn.Dropout(dropout),
            nn.Flatten(),
        )
        self.head = nn.Linear(_flat_dim(self.features, n_ch, n_times), n_classes)

    def forward(self, x):
        return self.head(self.features(x))


class DeepConvNet(nn.Module):
    """Schirrmeister et al. 2017."""

    def __init__(self, n_ch, n_times, n_classes, dropout=0.5):
        super().__init__()

        def block(c_in, c_out):
            return [nn.Dropout(dropout), nn.Conv2d(c_in, c_out, (1, 10), bias=False),
                    nn.BatchNorm2d(c_out), nn.ELU(), nn.MaxPool2d((1, 3))]

        self.features = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10)),
            nn.Conv2d(25, 25, (n_ch, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3)),
            *block(25, 50), *block(50, 100), *block(100, 200),
            nn.Flatten(),
        )
        self.head = nn.Linear(_flat_dim(self.features, n_ch, n_times), n_classes)

    def forward(self, x):
        return self.head(self.features(x))


MODELS = {"eegnet": EEGNet, "shallow": ShallowConvNet, "deep": DeepConvNet}


def build_model(cfg, n_ch, n_times, n_classes):
    return MODELS[cfg["name"]](n_ch, n_times, n_classes, **cfg[cfg["name"]])
