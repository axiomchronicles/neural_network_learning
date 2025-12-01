import os
import pathlib

from torch.utils.data import Dataset
from PIL import Image

from paths import TRAIN_DIR, TEST_DIR, IMAGE_DIR, META_DIR
from utils import resolve_train_test_paths
from schema import Food101MetaData
from transform import TrainTransformV1

class Food101Dataset(Dataset):
    def __init__(self, root_dir = None, meta_file = None, transform = None):
        super(Food101Dataset, self).__init__()

        self.root_dir = pathlib.Path(root_dir) if isinstance(root_dir, pathlib.Path) else root_dir

        with open(meta_file, "r") as f:
            self.annotation = [
                self.root_dir / (line.strip() + ".jpg")
                for line in f
            ]
        # self.annotation = self.annotation = [img
        #            for path in self.root_dir
        #            for img in path.parent.glob("*.jpg")]
        # # print(self.annotation)
        self.transform = transform
        self.classes, self.classes_idx = self.scan_dirs()
        # self.samples = [
        #     (img_path, self.classes_idx[img_path.parent.stem])
        #     for img_path in self.annotation
        # ]

    def __len__(self):
        return len(self.annotation)
    
    def scan_dirs(self):
        classes = sorted(list(entry.name for entry in os.scandir(IMAGE_DIR)))
        classes_idx = {key: index for index, key in enumerate(classes)}
        return classes, classes_idx
    
    def __getitem__(self, index):
        image = Image.open(self.annotation[index])
        classnames = self.annotation[index].parent.stem
        class_idx = self.classes_idx[classnames]

        if self.transform:
            return self.transform(image), class_idx
        
        return image, class_idx
    
    def __repr__(self):
        # return f"{len(self.annotation)}"
        return (f"Food101Dataset \n"
        f"   Number of Datapoints: {len(self.annotation)} \n"
        f"   Root Location: {self.root_dir} \n"
        f"   Transforms: {self.transform}")
    

# if __name__ == "__main__":
#     train_meta = META_DIR / Food101MetaData.TRAIN.value
#     # train_dir, test_dir = resolve_train_test_paths()
#     #     for paths in train_dir:
#     #         print(list(paths.parent.glob("*.jpg")))
#     dataset = Food101Dataset(root_dir = IMAGE_DIR, meta_file = train_meta, transform = TrainTransformV1)
#     print(dataset)