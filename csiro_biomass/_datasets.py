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

DATA_LOADING_MODE = DataLoadingMode.Local.value # use "local to use it from you local device, 'global' to use it on kaggel"
DATA_LOADING_LOCAL_URL = pathlib.Path = pathlib.Path(pathlib.Path.home() / "Downloads" / "csiro-biomass") # (object) pathlib.Path
DATA_LOADING_KAGGEL_URL = "" # this url refer to kaggel house only use on submission

def load_csv_data():
    # csv FileType Loading Loading using pandas
    # print(FileTypeMode.TRAIN.value)
    root: pathlib.Path = DATA_LOADING_LOCAL_URL if DataLoadingMode.Local.value else DATA_LOADING_KAGGEL_URL
    train_csv: pd.DataFrame = pd.read_csv(filepath_or_buffer = root / FileTypeMode.TRAIN.value)
    test_csv: pd.DataFrame = pd.read_csv(filepath_or_buffer = root / FileTypeMode.TEST.value)

    # Return Type: pandas.DataFrame -> train.csv | test.csv File
    return train_csv, test_csv

class CsiroBiomassDataLoader(D.Dataset): 
    def __init__(self, csv_file: pd.DataFrame, root: pathlib.Path, transform: T.transforms):
        super(CsiroBiomassDataLoader, self).__init__()

        # CsiroBiomassDataLoader use to load the Traning Dataset which you can use to train and validate model
        # This dataset doesn't contain much data so create a batch_size of (16, 32) if you go further memory can freeze
        # Original Training image with the images and target use to train and val model use split_size of 0.8 & 0.2

        self.annonation = csv_file
        # Root path as pathlib.Path like (object) point towards the csiro-biomass data files
        self.root = pathlib.Path(root) if isinstance(root, str) else root
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
    
    def __getitem__(self, index):
        # Main (object like structure) split training data into image, target format (X, y) -> formally knowns
        rows = self.grouped.iloc[index]
        image_path = self.root / rows["image_path"]
        # Using PIL.Image to read the image files this runs on CPU (no acceleration here) -> CPU might Throttle
        # Converion of image into RGB is essential, colour_channels of the image is 3 and the image shape is 4D
        image: Image = Image.open(image_path).convert("RGB")

        # Targeted columns are in string(dtype) requires to convet in to flot before changing them into torch.Tensor
        target_nums = pd.to_numeric(rows.loc[self.target_columns], errors="coerce").astype("float32")
        # Tensor Formation
        target = torch.tensor(target_nums.values).type(torch.float32)

        # Applying Transform if user passed the transform
        if self.transform:
            image = self.transform(image)

        # Return Type tuple like object (image, target)
        # Use -> image, target = dataset
        return image, target
    
class CsiroBiomassTestDataLoader(D.Dataset):
    def __init__(self, csv_file: pd.DataFrame, root: pathlib.Path, transform: T.transforms):
        super(CsiroBiomassTestDataLoader, self).__init__()

        # CsiroBiomassTestDataLoader is a Testing dataset only contain 5(rows) of data
        # Use batch_size(1) default else data may lost or return type(NaN)
        # This dataset can only use for evalute the model and to create the final (submission.csv) File

        self.annonation = csv_file
        # Root path (pathlib.Path) Type object for train.csv file 
        self.root = pathlib.Path(root) if isinstance(root, str) else root
        # Torchvision Transform (preprocessing) ToTensor, ImagePixel, Resize.
        self.transform = transform

    def __len__(self):
        # Torch Dataset required lenght of the dataset to return
        return len(self.annonation)
    
    def __getitem__(self, index):
        rows = self.annonation.iloc[index]
        image_path = self.root / rows["image_path"]
        # Test Dataset only contain 1 Image and linked with 5 cols so it required co-current loading 
        # Image colour_channel is 3 with the dimesion to 4 so it requires conversion to RGB
        image: Image = Image.open(image_path).convert("RGB")

        # For evaluation we don't need target as tensor wen can use the target to get the sample_id's
        # Forwading the image to the model we can extract the prediction and build the submission file on it.
        sampel_id = rows["sample_id"]
        target = rows["target_name"]

        # torch.Transforms if user passed the transform Compose
        if self.transform:
            image = self.transform(image)

        # Tuple of 3
        # use image, target, sid = testdataset
        return image, target, sampel_id