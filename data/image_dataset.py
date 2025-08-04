# data/image_dataset.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root="./data", train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
