import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(config, model):
    train_cfg = config["training"]

    if train_cfg["optimizer"] == "adam":
        return optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    elif train_cfg["optimizer"] == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            momentum=0.9,
        )

    else:
        raise ValueError("Unknown optimizer")


def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_model(config, model, train_loader, val_loader, device, checkpoint_path):
    train_cfg = config["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    actual_epochs_ran = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(train_cfg["epochs"]):
        actual_epochs_ran += 1
        model.train()
        running_train_loss = 0.0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            total_train += images.size(0)

        train_loss = running_train_loss / total_train
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{train_cfg['epochs']}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Val acc:    {val_acc:.4f}")

        if val_loss < best_val_loss - train_cfg["min_delta"]:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved new best model")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        print()

        if train_cfg["early_stopping"] and epochs_without_improvement >= train_cfg["patience"]:
            stopped_early = True
            print(f"Stopping early at epoch {epoch+1}")
            break

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "actual_epochs_ran": actual_epochs_ran,
        "stopped_early": stopped_early,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }


def test_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    return {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }