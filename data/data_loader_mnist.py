from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_data_loader(data_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    train_set = MNIST(data_path,
                      train=True,
                      download=True,
                      transform=transform)
    test_set = MNIST(data_path,
                     train=False,
                     download=True,
                     transform=transform)

    return train_set, test_set
