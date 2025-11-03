import torch

class CsiroBiomassModel(torch.nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 5,
                 hidden_units: int = 128, batch_units: int = 64, drop_units: int = 32):
        super(CsiroBiomassModel, self).__init__()
        
        self.in_features: int = in_features
        self.out_featues: int = out_features
        self.hidden_units: int = hidden_units
        self.batch_units: int = batch_units
        self.drop_units: int = drop_units
        
        self.conv_kernel_size: int = 3
        self.maxpool_kernel_size: int = 2
        self.padding: int = 1
        self.stride: int = 1
        self.p: float = 0.25

        self.convolutional_layer1: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = self.in_features, out_channels = self.hidden_units,
                            kernel_size = self.conv_kernel_size, padding = self.padding,
                            stride = self.stride),
            torch.nn.BatchNorm2d(num_features = self.hidden_units),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels = self.hidden_units, out_channels = self.hidden_units,
                            kernel_size = self.conv_kernel_size, padding = self.padding,
                            stride = self.stride),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size = self.maxpool_kernel_size),
            torch.nn.Dropout2d(p = self.p)
        )

        self.convolutional_layer2: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = self.hidden_units, out_channels = self.batch_units,
                            kernel_size = self.conv_kernel_size, padding = self.padding,
                            stride = self.stride),
            torch.nn.BatchNorm2d(num_features = self.batch_units),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels = self.batch_units, out_channels = self.batch_units,
                            kernel_size = self.conv_kernel_size, padding = self.padding,
                            stride = self.stride),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size = self.maxpool_kernel_size),
            torch.nn.Dropout2d(p = self.p)
        )

        self.convolutional_layer3: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = self.batch_units, out_channels = self.drop_units,
                            kernel_size = self.conv_kernel_size, padding = self.padding,
                            stride = self.stride),
            torch.nn.BatchNorm2d(num_features = self.drop_units),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels = self.drop_units, out_channels = self.drop_units,
                            kernel_size = self.conv_kernel_size, padding = self.padding,
                            stride = self.stride),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size = self.maxpool_kernel_size),
            torch.nn.Dropout2d(p = self.p)
        )

        with torch.inference_mode():
            x = torch.zeros(1, self.in_features, 224, 224)
            x = self.convolutional_layer1(x)
            x = self.convolutional_layer2(x)
            x = self.convolutional_layer3(x)
            flattern = x.view(1, -1).shape[1]

        self.regression: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = flattern, out_features = self.batch_units),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = self.p),
            torch.nn.Linear(in_features = self.batch_units, out_features = self.out_featues)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional_layer1(x)
        x = self.convolutional_layer2(x)
        x = self.convolutional_layer3(x)
        x = self.regression(x)
        return x
    
# torch.nn.Bl