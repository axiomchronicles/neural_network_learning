import _dataloader, _datasets, network, metrics
import pathlib
import torch
from torchvision import transforms as T
from torch.utils import data as D
from tqdm.auto import tqdm

ROOT: pathlib.Path = pathlib.Path.home() / "Downloads" / "csiro-biomass"
DEVICE: torch.device = "mps" if torch.mps.is_available() else "cpu"

# transform = T.Compose(transforms = [
#     T.Resize(size=(224, 224)),
#     T.ToTensor()
# ])

# train_csv, test_csv = _datasets.load_csv_data()
# train_datset = _datasets.CsiroBiomassDataLoader(csv_file = train_csv, root = ROOT, transform = transform)

# train_loader = _dataloader.TrainingDatasetLoader(dataset = train_datset, batch_size = 32, shuffle = True)

# image, target = next(iter(train_loader.load_trainingset()))
# print(image.shape), print(target.shape)

# model = network.CsiroBiomassModel().to(DEVICE)

# rand_image = torch.randn(size = (1, 3, 224, 224))

# with torch.inference_mode():
#     pred = model(rand_image.to(DEVICE))
#     print(pred)


class ModelTrainingEvaluation:
    def __init__(self, model: torch.nn.Module = None, criterion: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None, dataset: D.Dataset = None):
        
        # self.epochs: int = epochs
        self.model: torch.nn.Module = model
        self.criterion: torch.nn.Module = criterion
        self.optimizer: torch.optim.Optimizer = optimizer
        self.dataset: D.Dataset = dataset

    def train_model(self):

        # for _ in tqdm(range(self.epochs if epochs == None else epochs)):
        self.model.train()

        running_loss: float = 0.0
        target_true_list: list = []
        target_prediction_list: list = []

        for index, (image, target) in enumerate(self.dataset):
            image, target = image.to(DEVICE), target.to(DEVICE)

            target_prediction = self.model(image)
            loss = self.criterion(target_prediction, target)
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            target_true_list.append(target)
            target_prediction_list.append(target_prediction)

        true_target = torch.cat(target_true_list)
        prediction_target = torch.cat(target_prediction_list)
        average_loss = running_loss / len(self.dataset)

        r2_scores, weighted_r2 = metrics.weighted_r2(true_target, prediction_target)
        print(f"Training Loss: {average_loss} | Weighted R² Score : {weighted_r2}")

        return {"model": self.model.__class__.__name__,
                "criterion": average_loss,
                "r2_scores": r2_scores,
                "weighted_r2": weighted_r2}