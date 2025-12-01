import torch


def eval_modelv1(dataset: torch.utils.data.DataLoader, model: torch.nn.Module,
                criterion, accuracy_fn = None, device = "cpu"):
    
    running_loss = 0.0
    model.eval()
    accuracy_fn.reset()
    
    with torch.inference_mode():
        for batch_idx, (image, target) in enumerate(dataset):
            image, target = image.to(device = device, memory_format = torch.channels_last), target.to(device)

            predictionLogits = model(image)
            modelPrediction = predictionLogits.argmax(dim = 1)

            trainingloss = criterion(predictionLogits, target)

            running_loss += trainingloss.item()
            accuracy_fn.update(target, modelPrediction)

        averageLoss = running_loss / len(dataset)
        averageAccuracy = accuracy_fn.compute().item()

        return (averageLoss, averageAccuracy)