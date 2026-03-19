import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_train_transform(config):
    aug = config["dataset"]["augmentation"]

    if aug == "none":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif aug == "flip":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif aug == "flip_crop":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        raise ValueError(f"Unknown augmentation: {aug}")


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_data(config):
    data_cfg = config["dataset"]
    train_transform = get_train_transform(config)
    test_transform = get_test_transform()

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_cfg["data_dir"],
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_cfg["data_dir"],
        train=False,
        download=True,
        transform=test_transform,
    )

    train_size = int(data_cfg.get("train_fraction", 0.9) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    val_dataset.dataset.transform = test_transform

    batch_size = config["training"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "dataset_name": "cifar10",
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": 10,
        "input_channels": 3,
        "image_size": (32, 32),
        "class_names": CLASSES,
    }