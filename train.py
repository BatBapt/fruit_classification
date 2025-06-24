import os
import torch
from torchmetrics import Accuracy
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from torchsummary import summary

import custom_dataset as custom_dataset
from model import get_model
import configuration as cfg


def run_config_yaml(path):
    """
    read the config YAML file and return the dictionary
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config


def setup_training(model, stage):

    if stage == "stage1":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True


def train(train_loader, val_loader, model, num_epochs, criterion, optimizer, device, scheduler=None):
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

        if scheduler:
            scheduler.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_loop = tqdm(val_loader, desc=f"Validation {epoch+1}")
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
            f'Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')


        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), cfg.MODEL_WEIGHTS)
            print(f"Modèle sauvegardé avec une précision de validation de {best_accuracy:.4f}")

        if (epoch + 1) % plot_every == 0:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid()

            plt.savefig(f'data_training.png')
            plt.close()

if __name__ == "__main__":
    config = run_config_yaml("config.yaml")

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
        num_workers=2,
        pin_memory=True
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    class_name = custom_dataset.get_classes_name(os.path.join(cfg.BASE_PATH, "classname.txt"))


    model = get_model(len(class_name)).to(cfg.DEVICE)
    criterion = torch.nn.CrossEntropyLoss()

    for stage in list(config.keys()):
        setup_training(model, stage)
        current_stage = config[stage]

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(trainable_params, lr=current_stage["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=current_stage["step_size"], gamma=current_stage["gamma"])
        num_epochs = current_stage["num_epochs"]

        print(f"Starting stage: {stage.upper()}")
        print(f"Optimizer: {len(trainable_params)} paramètres, LR={current_stage['lr']}")
        print(f"Training: {current_stage['num_epochs']} epochs")
        print("-" * 40)
        train(
            train_data_loader,
            val_data_loader,
            model,
            num_epochs,
            criterion,
            optimizer,
            cfg.DEVICE,
            scheduler=scheduler
        )
