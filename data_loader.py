import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import multiprocessing  # Import for getting the number of CPU cores

def get_data_loaders(data_dir, batch_size=32):
    # Define transformations for training and validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),  # Randomly crop and resize to 224x224
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256x256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])

    # Determine the number of workers based on CPU cores
    num_workers = multiprocessing.cpu_count() - 4 # Get the number of CPU cores
    print(f"Number of workers for dataloader: {num_workers}")

    # Load the training data using ImageFolder
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/train'), transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Load the validation data using ImageFolder
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/val'), transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader  # Return the data loaders for training and validation

