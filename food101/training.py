import torch


def train_modelv1(dataset: torch.utils.data.DataLoader, model: torch.nn.Module,
                criterion, optimizer: torch.optim.Optimizer, accuracy_fn = None, device = "cpu"):
    
    running_loss = 0.0
    model.train()
    accuracy_fn.reset()

    for batch_idx, (image, target) in enumerate(dataset):
        image, target = image.to(device = device), target.to(device)

        # print(f"Image Shape: {image.shape}")

        predictionLogits = model(image)
        modelPrediction = predictionLogits.argmax(dim = 1)

        trainingloss = criterion(predictionLogits, target)
        optimizer.zero_grad()

        trainingloss.backward()
        optimizer.step()

        running_loss += trainingloss.item()
        accuracy_fn.update(target, modelPrediction)

    averageLoss = running_loss / len(dataset)
    averageAccuracy = accuracy_fn.compute().item()

    return (averageLoss, averageAccuracy)