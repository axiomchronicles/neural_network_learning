from transform import EvalTransform, TrainTransformV1
from torch.utils import data

from paths import META_DIR, IMAGE_DIR
from schema import Food101MetaData
from dataset import Food101Dataset
from transform import TrainTransformV1, EvalTransform

train_data_meta = META_DIR / Food101MetaData.TRAIN.value
test_data_meta = META_DIR / Food101MetaData.TEST.value

def create_dataset():
    training_data = Food101Dataset(root_dir = IMAGE_DIR,
                                   meta_file = train_data_meta,
                                   transform = TrainTransformV1)
    
    testing_data = Food101Dataset(root_dir = IMAGE_DIR,
                                   meta_file = test_data_meta,
                                   transform = EvalTransform)
    
    return (training_data, testing_data)


def load_dataset(dataset: data.Dataset, batch_size: int = 1, 
                shuffle: bool = True, num_workers: int = 0,
                pin_memory: bool = False, drop_last: bool = False,
                timeout: float = 0):
    
    return data.DataLoader(dataset = dataset, batch_size = batch_size,
                           shuffle = shuffle, num_workers = num_workers,
                           pin_memory = pin_memory, drop_last = drop_last,
                           timeout = timeout)
    

if __name__ == "__main__":
    train, test = create_dataset()
    train_loader = load_dataset(train, batch_size=32, shuffle=True)
    # print(train_loader)
    image, target = next(iter(train_loader))
    print(image[0])