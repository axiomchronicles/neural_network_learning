import torch
import matplotlib.pyplot as plt

from models.tinyvgg import TinyVGG
from loader import create_dataset, load_dataset
from training import train_modelv1
from evaluate import eval_modelv1
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")

def execute(epochs: int = 10, dataset: torch.utils.data.DataLoader = None, model: torch.nn.Module = None,
                criterion = None, optimizer: torch.optim.Optimizer = None, accuracy_fn = None, device = "cpu",
                writer: SummaryWriter = None):
    
    totalEpochs = []
    trainingLosses, trainingAccuracy = [], []
    testingLosses, testingAccuracy = [], []

    for epoch in range(epochs):
        trainLoss, trainAccuracy = train_modelv1(dataset = dataset, model = model,
                                                criterion = criterion, optimizer = optimizer,
                                                accuracy_fn = accuracy_fn, device = device)
        
        testLoss, testAccuracy = eval_modelv1(dataset = dataset, model = model,
                                                criterion = criterion,
                                                accuracy_fn = accuracy_fn, device = device)
        
        trainingLosses.append(trainLoss), trainingAccuracy.append(trainAccuracy)
        testingLosses.append(testLoss), testingAccuracy.append(testAccuracy)
        totalEpochs.append(epoch)

        writer.add_scalars(main_tag = "Loss",
                        tag_scalar_dict = {
                            "train_loss": trainLoss,
                            "test_loss": testLoss
                        }, global_step = epoch)

        writer.add_scalars(main_tag = "Accuracy",
                        tag_scalar_dict = {
                            "train_acc": trainAccuracy,
                            "test_acc": testAccuracy
                        }, global_step = epoch)

        writer.add_graph(model = model, input_to_model = 
                        torch.randn([16, 3, 224, 224]).to(device))

        if epoch % (epochs * 0.2) == 0:
            print(f"Training Losses: {trainLoss} | Training Accuracy: {trainAccuracy}")
            print(f"Testing Losses: {testLoss} | Testing Accuracy: {testAccuracy} \n")

    writer.close()


if __name__ == "__main__":
    # torch.backends.mps.matmul.allow_tf32 = True
    train, test = create_dataset()
    # print(len(train.annotation))
    train_loader = load_dataset(train, batch_size = 32, shuffle = True, num_workers=4)
    test_loader = load_dataset(test, batch_size=32, shuffle=False, num_workers=4)

    model = TinyVGG(in_features = 3, out_features = len(train.classes), hidden_units = 64).to(device = DEVICE, memory_format = torch.channels_last)
    # with torch.inference_mode():
    #     for image, target in train_loader:
    #         logits = model(image.to(DEVICE, memory_format = torch.channels_last))
    #         pred = logits.argmax(dim = 1)
    #         break

    #     print(pred)


 
    # print(len(train_loader), len(test_loader))

    # image, target = next(iter(train_loader))

    # plt.figure(figsize=(10, 5))
    # plt.imshow(image[0].permute(1, 2, 0))
    # plt.title(train.classes[target[0]])
    # plt.axis("off")
    # plt.show()