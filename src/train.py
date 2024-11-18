from datasets import tqdm
import torch
import numpy as np
from src.model.ConvNet import ConvNet
from dataset import train_loader, valid_loader

epochs=10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train(model, train_loader, valid_loader, criterion,optimizer, epochs=epochs):
    valid_loss_min = np.Inf  # set initial "min" to infinity
    num_epochs_without_val_loss_reduction = 0
    early_stopping_window = 5
    for epoch in tqdm(range(epochs)):
        # monitor losses
        train_loss = 0
        num_correct_preds_train = 0  # number of correct predictions to calculate accuracy
        valid_loss = 0
        num_correct_preds_valid = 0  # number of correct predictions to calculate accuracy
        model.train()
        for data,label in train_loader:
            data,label = data.to(device), label.to(device)
            optimizer.zero_grad()
            label_pred = model(data)
            loss = criterion(label_pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            num_correct_preds_train += (label_pred.softmax(-1).argmax(-1) == label).sum()

        model.eval()
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            label_pred = model(data)
            loss = criterion(label_pred, label)
            valid_loss = loss.item() * data.size(0)
            # update the number of correct predictions
            num_correct_preds_valid += (label_pred.softmax(-1).argmax(-1) == label).sum()


        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        train_acc = num_correct_preds_train / len(train_loader.sampler)
        valid_acc = num_correct_preds_valid / len(valid_loader.sampler)

        print('Epoch: {} \tTraining Loss: {:.6f} \tTrain Accuracy: {:.4f} \tValidation Loss: {:.6f} \Validation Accuracy: {:.4f}'.format(
                epoch + 1,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc
            ))

        # save model if validation loss has decreased (early stopping)
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

        else: num_epochs_without_val_loss_reduction += 1

        if num_epochs_without_val_loss_reduction >= early_stopping_window:
            print(f'No reduction in validation loss for {early_stopping_window} epochs. Stopping training...')
            break


if __name__ == '__main__':
    model = ConvNet()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train(model, train_loader, valid_loader,criterion, optimizer)