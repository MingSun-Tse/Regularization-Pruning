from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms


def get_data_loader(data_path, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # ref to EigenDamage code
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    train_set = CIFAR100(data_path,
                         train=True,
                         download=True,
                         transform=transform_train)
    test_set = CIFAR100(data_path,
                        train=False,
                        download=True,
                        transform=transform_test)

    return train_set, test_set
