import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

# how many samples per batch to load
batch_size = 32
# percentage of training set to use as validation
valid_size = 0.2
train_data = datasets.MNIST(root = 'data',
                            download=True,
                            train = True)
test_data = datasets.MNIST(root = 'data',
                           download=True,
                           train = False)
train_data.transform = transforms.ToTensor()
mean = 0
std = 0
# Calculate the mean and standard deviation
for img, target in train_data:
    mean += img.mean() / len(train_data)
    std += img.std() / len(train_data)
print(f'Train Data Mean: {mean}')
print(f'Train Data Standard Deviation: {std}')

normalization = transforms.Normalize((mean,), (std,))
train_transforms = transforms.Compose([transforms.ToTensor(), normalization])
test_transforms = transforms.Compose([transforms.ToTensor(), normalization])

train_data.transform = train_transforms
test_data.transform = test_transforms

# obtain training indices that will be used for validation
indices = np.arange(len(train_data))
train_indices, valid_indices = train_test_split(indices, test_size=valid_size, random_state=42, stratify=train_data.targets)

print(f'Number Training Samples: {len(train_indices)}')
print(f'Number Validation Samples: {len(valid_indices)}')

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                           sampler = train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                          sampler = valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                         )