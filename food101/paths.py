import pathlib
import os

ROOT = pathlib.Path("../datasets").resolve()
DATASET_DIR = ROOT / "food-101"
IMAGE_DIR = DATASET_DIR / "images"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
META_DIR = DATASET_DIR / "meta"


# if __name__ == "__main__":
#     results = list(entry.name for entry in os.scandir(IMAGE_DIR))
#     print(results[10])
#     print(len(results))