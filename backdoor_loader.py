from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PoisonedDataset
import sys

# In this function the two datasets are loaded. Either mnist or cifar10


def load_sets(datasetname, download, dataset_path):
    try:
        if datasetname == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            train_data = datasets.MNIST(root=dataset_path, train=True, download=download, transform=transform)
            test_data = datasets.MNIST(root=dataset_path, train=False, download=download, transform=transform)
            return train_data, test_data

        elif datasetname == 'cifar':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            train_data = datasets.CIFAR10(root=dataset_path, train=True, download=download, transform=transform)
            test_data = datasets.CIFAR10(root=dataset_path, train=False, download=download, transform=transform)
            return train_data, test_data
        else:
            raise NotAcceptedDataset

    except NotAcceptedDataset:
        print('Dataset Error. Choose "cifar" or "mnist"')
        sys.exit()


# With this function 3 dataloaders are returned. The first is the training dataloader with a portion of poisoned data,
# second is the test dataloader without any poisoned data to test the performance of the trained model, and the third is
# the dataloader with poisoned test data to test the poisoned model of new poisoned test data.


def backdoor_data_loader(datasetname, train_data, test_data, trigger_label, proportion, batch_size, attack):
    train_data = PoisonedDataset(train_data, trigger_label, proportion=proportion, mode="train", datasetname=datasetname, attack=attack)
    test_data_orig = PoisonedDataset(test_data,  trigger_label, proportion=0, mode="test", datasetname=datasetname, attack=attack)
    test_data_trig = PoisonedDataset(test_data,  trigger_label, proportion=1, mode="test", datasetname=datasetname, attack=attack)

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=batch_size, shuffle=False)
    test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_orig_loader, test_data_trig_loader


# Just a simple custom exception that is raised when the dataset argument is not accepted


class NotAcceptedDataset(Exception):
    """Not accepted dataset as argument"""
    pass
