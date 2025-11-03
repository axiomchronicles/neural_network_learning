import torch

class CsiroConvolutionalModel(torch.nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 5, hidden_units: int = 128,
                 batch_units: int = 64, drop_units: int = 32):
        super(CsiroConvolutionalModel).__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.hidden_units: int = hidden_units
        self.batch_units: int = batch_units
        self.drop_units: int = drop_units

        self.start_kernel_size: int = (7, 7)
        self.start_padding: tuple = (3, 3)
        self.start_strides: tuple = (2, 2)
        
        self.conv_kernel_size: int = 3
        self.conv_padding: tuple = (2, 2)
        self.conv_strides: tuple = (3, 3)

        self.maxpool_kernel_size: int = 2

        torch.nn.Conv2d(in_channels = self.in_features, out_channels = self.hidden_units,
                        kernel_size = (7, 7))

        self.convolutional_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = self.in_features, out_channels = self.hidden_units,
                            kernel_size = self.conv_kernel_size, padding = self.conv_padding,
                            stride = self.conv_strides),
            torch.nn.BatchNorm2d(num_features = self.hidden_units),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels = self.hidden_units, out_channels = self.hidden_units,
                            kernel_size = self.conv_kernel_size, padding = self.conv_padding,
                            stride = self.conv_strides),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size = self.maxpool_kernel_size, )

        )