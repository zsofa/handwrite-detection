from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64):
    normalize_with_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=normalize_with_transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=normalize_with_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader