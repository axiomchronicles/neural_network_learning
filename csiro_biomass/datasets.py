import torch
import torchvision
import pathlib
import enum
import pandas as pd
import PIL.Image as Image

from torch.utils import data as D
from torchvision import datasets, transforms as T

class DataLoadingMode(enum.Enum):
    Local: str = "local"
    GLOBAL: str = "global"

class FileTypeMode(enum.Enum):
    # csv FileType Modules
    TRAIN: str = "train.csv"
    TEST: str = "test.csv"
    SUBMISSION: str = "sample_submission.csv"

    # Directory FileType Module
    TRAIN_DIR: str = "train"
    TEST_DIR: str = "test"

DATA_LOADING_MODE = DataLoadingMode.Local # use "local to use it from you local device, 'global' to use it on kaggel"
DATA_LOADING_LOCAL_URL = pathlib.Path = pathlib.Path().home() / "Downloads" / "csiro-biomass" # (object) pathlib.Path
DATA_LOADING_KAGGEL_URL = "" # this url refer to kaggel house only use on submission

def _load_csv():
    # csv FileType Loading Loading using pandas
    root: pathlib.Path = DATA_LOADING_LOCAL_URL if DataLoadingMode.Local else DATA_LOADING_KAGGEL_URL
    train_csv: pd.DataFrame = pd.read_csv(filepath_or_buffer = root / FileTypeMode.TRAIN)
    test_csv: pd.DataFrame = pd.read_csv(filepath_or_buffer = root / FileTypeMode.TEST)

    # Return Type: pandas.DataFrame -> train.csv | test.csv File
    return train_csv, test_csv

class CsiroBiomassDataLoader(D.DataLoader): 
    def __init__(self, csv_file: pd.DataFrame, root: pathlib.Path, transform: T.transforms):
        super(CsiroBiomassDataLoader, self).__init__()

        self.annonation = csv_file
        # Root path as pathlib.Path like (object) point towards the csiro-biomass data files
        self.root = pathlib.Path(root)
        # Torchvision Transform (preprocessing) ToTensor, ImagePixel, Resize.
        self.transform = transform

        # Grouped Transaction on pandas.DataFrame to sort the Row using the label form data(train.csv)
        self.grouped = (
            self.annonation
            .pivot_table(index = "image_path", columns = "target_name", values = "target")
            .reset_index()
        )
        # Target coloums to use as lable which model's train on excluding the sample_id to avoid overfitting
        self.target_columns: list = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]

    def __len__(self):
        # Torch Dataset required lenght of the dataset to return
        return len(self.grouped)
    
    def __getitem__(self, idx):
        # Main (object like structure) split training data into image, target format (X, y) -> formally knowns
        rows = self.annonation.iloc[idx]
        image_path = self.root / rows["image_path"]
        # Using PIL.Image to read the image files this runs on CPU (no acceleration here) -> CPU might Throttle
        image: Image = Image.open(image_path).convert("RGB")

        # Targeted columns are in string(dtype) requires to convet in to flot before changing them into torch.Tensor
        target_nums = pd.to_numeric(rows[self.target_columns], errors = "coerce").astype("float32")
        # Tensor Formation
        target = torch.tensor(target_nums).type(torch.float32)

        # Applying Transform if user passed the transform
        if self.transform:
            image = self.transform(image)

        # Return Type tuple like object (image, target)
        # Use -> image, target = dataset
        return image, target