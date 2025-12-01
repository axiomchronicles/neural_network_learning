import torch
import matplotlib.pyplot as plt

from models.tinyvgg import TinyVGG
from loader import create_dataset, load_dataset

DEVICE = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")

if __name__ == "__main__":
    # torch.backends.mps.matmul.allow_tf32 = True
    train, test = create_dataset()
    # print(len(train.annotation))
    train_loader = load_dataset(train, batch_size = 32, shuffle = True, num_workers=4)
    test_loader = load_dataset(test, batch_size=32, shuffle=False, num_workers=4)

    model = TinyVGG(in_features = 3, out_features = len(train.classes), hidden_units = 64).to(device = DEVICE, memory_format = torch.channels_last)
    with torch.inference_mode():
        for image, target in train_loader:
            logits = model(image.to(DEVICE, memory_format = torch.channels_last))
            pred = logits.argmax(dim = 1)
            break

        print(pred)
 
    # print(len(train_loader), len(test_loader))

    # image, target = next(iter(train_loader))

    # plt.figure(figsize=(10, 5))
    # plt.imshow(image[0].permute(1, 2, 0))
    # plt.title(train.classes[target[0]])
    # plt.axis("off")
    # plt.show()