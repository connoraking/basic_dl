from src.data.cifar10 import get_data as get_cifar10_data
# later:
# from src.data.cifar100 import get_data as get_cifar100_data


def get_dataset(config):
    name = config["dataset"]["name"]

    if name == "cifar10":
        return get_cifar10_data(config)
    # elif name == "cifar100":
    #     return get_cifar100_data(config)

    raise ValueError(f"Unknown dataset: {name}")