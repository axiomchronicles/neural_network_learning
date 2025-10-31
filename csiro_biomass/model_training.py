import _dataloader, _datasets
import pathlib
import torch
from torchvision import transforms as T

ROOT: pathlib.Path = pathlib.Path.home() / "Downloads" / "csiro-biomass"

transform = T.Compose(transforms = [
    T.Resize(size=(224, 224)),
    T.ToTensor()
])

train_csv, test_csv = _datasets.load_csv_data()
train_datset = _datasets.CsiroBiomassDataLoader(csv_file = train_csv, root = ROOT, transform = transform)

train_loader = _dataloader.TrainingDatasetLoader(dataset = train_datset, batch_size = 32, shuffle = True)

image, target = next(iter(train_loader.load_trainingset()))
print(image.shape), print(target.shape)