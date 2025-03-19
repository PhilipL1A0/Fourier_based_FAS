import torch.nn as nn


class base_model(nn.Module):
    def __init__(self, args):
        super(base_model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 50, 5, 1, padding=2),
            nn.BatchNorm2d(50, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(50, 100, 3, 1, padding=1),
            nn.BatchNorm2d(100, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(100, 150, 3, 1, padding=1),
            nn.BatchNorm2d(150, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(150, 200, 3, 1, padding=1),
            nn.BatchNorm2d(200, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(200, 250, 3, 1, padding=1),
            nn.BatchNorm2d(250, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(250, 1000, 3, 1),
            nn.BatchNorm2d(1000, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=False),

        )

        self.classifier = nn.Sequential(
            nn.Linear(1000, 400),
            nn.BatchNorm1d(400, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Linear(400, args.class_num),

        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x