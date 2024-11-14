import torch
from tqdm.auto import tqdm
from dataset import test_loader
from src.model.ConvNet import ConvNet
from src.model.FNNet import FNNet
from train import device

model = ConvNet()
model.load_state_dict(torch.load('model.pt'))
model.to(device)
def test():
    # set model to evaluation mode
    model.eval()
    num_correct = 0
    total = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)
            label_pred = model(data)
            _, pred = torch.max(label_pred, 1)
            num_correct += (pred == label).sum().item()
            total += label.size(0)

    accuracy = num_correct / total
    print(f'Accuracy: {accuracy:.4f}')