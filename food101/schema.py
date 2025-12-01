import enum


class Food101MetaData(enum.Enum):
    TRAIN: str = "train.txt"
    TEST: str = "test.txt"
    CLASSES: str = "classes.txt"
    LABELS: str = "labels.txt"
    # ROOT_PATH: str = ""