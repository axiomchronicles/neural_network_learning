import pathlib

from schema import Food101MetaData


def fetch_train_test_metadata(root: pathlib.Path):
    if not root.is_dir():
        raise FileNotFoundError(f"Directory: {root} not found")
    
    TrainMeta = root / Food101MetaData.TRAIN.value
    TestMeta = root / Food101MetaData.TEST.value

    with open(TrainMeta) as tf:
        train_img_paths = [lines.strip() for lines in tf if lines.strip()]

    with open(TestMeta) as f:
        test_img_path = [line.strip() for line in f if line.strip()]

    return train_img_paths, test_img_path


def resolve_train_test_paths():
    metaroot = pathlib.Path("../datasets/food-101/meta")
    food101root = pathlib.Path("../datasets/food-101/images")

    train_modulas, test_modulas = fetch_train_test_metadata(root=metaroot)
    
    train_resolve = [(food101root / path).with_suffix(".jpg").resolve() for path in train_modulas]
    test_resolve = [(food101root / path).with_suffix(".jpg").resolve() for path in test_modulas]

    return train_resolve, test_resolve

# if __name__ == "__main__":
#     train_resolve, test_resolve = resolve_train_test_paths()
#     print(type(train_resolve[10]))
#     path = pathlib.Path(__file__).parent.resolve()
#     root = path / "../datasets/food-101/meta"
#     train, test = fetch_train_test_metadata(root=root)
#     print(train[78], test[87])
