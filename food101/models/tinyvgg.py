import torch

class TinyVGG(torch.nn.Module):
    def __init__(self, in_features: int = 3, hidden_units: int = 10, out_features: int = 102):
        super(TinyVGG, self).__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.hidden_units: int = hidden_units

        self.block1: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = self.in_features, out_channels = self.hidden_units,
                            kernel_size = (3, 3), stride = 1, padding = 0),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels = self.hidden_units, out_channels = self.hidden_units,
                            kernel_size = (3, 3), stride = 1, padding = 0),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
        )
        self.block2: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = self.hidden_units, out_channels = self.hidden_units,
                            kernel_size = (3, 3), stride = 1, padding = 0),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels = self.hidden_units, out_channels = self.hidden_units,
                            kernel_size = (3, 3), stride = 1, padding = 0),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
        )
        self.classification: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = self.hidden_units * 53 * 53, out_features = self.hidden_units),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features = self.hidden_units, out_features = self.out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classification(self.block2(self.block1(x)))