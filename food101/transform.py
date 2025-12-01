from torchvision.transforms import v2 as T
import torch

TrainTransformV1 = T.Compose([
    T.Resize(size = (224, 224)),
    T.RandomHorizontalFlip(p = 0.5),
    T.TrivialAugmentWide(num_magnitude_bins = 31),
    T.PILToTensor(),
    T.ToImage(),
    T.ToDtype(dtype = torch.float32, scale = True),
    T.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
])

EvalTransform = T.Compose([
    T.Resize(size = (224, 224)),
    T.ToImage(),
    T.ToDtype(dtype = torch.float32, scale = True)
])