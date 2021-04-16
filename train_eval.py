import torch
from tqdm import tqdm
from sklearn.metrics import classification_report


# Normal training step with the poisoning dataset


def train(model, data_loader, criterion, optimizer):
    """
    Function for model training step
    """
    running_loss = 0
    model.train()
    for step, (batch_img, batch_label) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()  # Set gradients to zero
        output = model(batch_img)  # Forward pass
        loss = criterion(output, batch_label)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss
    return running_loss


# Simple evaluation with the addition of a classification report with precision and recall


def eval(model, test_loader, batch_size=64, report=True):
    """
    Simple evaluation with the addition of a classification report.
    """
    ret = 0
    preds = []
    gt = []
    with torch.no_grad():
        model.eval()
        for step, (batch_img, batch_label) in enumerate(test_loader):
            output = model(batch_img)
            label_predict = torch.argmax(output, dim=1)
            preds.append(label_predict)
            batch_label = torch.argmax(batch_label, dim=1)
            gt.append(batch_label)
            ret += torch.sum(batch_label == label_predict)

        if report:
            gt = torch.cat(gt, 0)
            preds = torch.cat(preds, 0)
            print(classification_report(gt.cpu(), preds.cpu()))

    return int(ret) / (step * batch_size)

