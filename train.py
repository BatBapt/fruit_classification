import os
import torch
from torchmetrics import Accuracy
from tqdm import tqdm
import matplotlib.pyplot as plt

import custom_dataset as custom_dataset
from model import get_model
import configuration as cfg


def train(train_loader, val_loader, model, num_epochs, criterion, optimizer, device):
    train_losses = []
    val_losses = []
    val_accuracies = []
    plot_every = 2
    best_accuracy = 0.0

    accuracy = Accuracy(task='multiclass', num_classes=len(class_name)).to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_loop = tqdm(val_loader, desc=f"Validation {epoch}")
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy += accuracy(outputs, labels).item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f'Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Modèle sauvegardé avec une précision de validation de {best_accuracy:.4f}")

        if (epoch + 1) % plot_every == 0:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.savefig(f'data_training.png')
            plt.close()

if __name__ == "__main__":
    dataset_train = custom_dataset.CustomDataset(
        csv_path=os.path.join(cfg.BASE_PATH, "train.csv"),
        transforms=custom_dataset.get_transform(train=True)
    )

    dataset_valid = custom_dataset.CustomDataset(
        csv_path=os.path.join(cfg.BASE_PATH, "val.csv"),
        transforms=custom_dataset.get_transform(train=False)
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=64,
        shuffle=True,
    )

    class_name = custom_dataset.get_classes_name(os.path.join(cfg.BASE_PATH, "classname.txt"))


    model = get_model(len(class_name)).to(cfg.DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    train(
        train_data_loader,
        val_data_loader,
        model,
        num_epochs,
        criterion,
        optimizer,
        cfg.DEVICE,
    )
