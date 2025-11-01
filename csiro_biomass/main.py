from model_training import ModelTrainingEvaluation
from network import CsiroBiomassModel
from model_training import ModelTrainingEvaluation
from _dataloader import TrainingDatasetLoader
from _datasets import CsiroBiomassDataLoader, load_csv_data
from preprocessing import WeightDecay, LearningRate

from torchvision import transforms as T
from torch.utils import data as D
from tqdm.auto import tqdm

import torch
import pathlib

RANDOM_SEEDS: int = 42
EPOCHS: int = 10
ROOT: pathlib.Path = pathlib.Path.home() / "Downloads" / "csiro-biomass"
DEVICE: torch.device = "mps" if torch.mps.is_available() else "cpu"

def _build_dataset():
    transform = T.Compose(transforms = [
        T.Resize(size=(224, 224)),
        T.ToTensor()
    ])

    train_csv, test_csv = load_csv_data()
    train_datset = CsiroBiomassDataLoader(csv_file = train_csv, root = ROOT, transform = transform)

    train_loader = TrainingDatasetLoader(dataset = train_datset, batch_size = 32, shuffle = True)
    training_dataset, validation_dataset = train_loader.load_trainingset(), train_loader.load_validationset()
    return training_dataset, validation_dataset

def train_model():
    model: CsiroBiomassModel = CsiroBiomassModel().to(DEVICE)
    torch.manual_seed(RANDOM_SEEDS)
    criterion: torch.nn.Module = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LearningRate.BEST_FIT.value,
                                 weight_decay = WeightDecay.BEST_FIT.value)
    training_dataset, _ = _build_dataset()
    training = ModelTrainingEvaluation(model = model, criterion = criterion,
                                       optimizer = optimizer, dataset = training_dataset)
    torch.manual_seed(RANDOM_SEEDS)
    result = training.train_model()
    return result

def training_loop():
    total_epochs: list = []
    losses: list = []
    weighted_r2: list = []

    for epoch in tqdm(range(EPOCHS)):
        training = train_model()

        total_epochs.append(epoch)
        losses.append(training.get("criterion"))
        weighted_r2.append(training.get("weighted_r2"))

if __name__ == "__main__":
    training_loop()