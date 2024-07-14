import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

#Transforms
training_transform= transforms.Compose([
    transforms.ToTensor()
])
testing_transform= transforms.Compose([
    transforms.ToTensor()
])

#Defining Data

training_data = datasets.MNIST(root="HandwrittenDigitClassification\Data", train=True, download=True, transform=training_transform)
testing_data = datasets.MNIST(root="HandwrittenDigitClassification\Data", train=False, download=True, transform=testing_transform)

#DataLoading

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=False)

#printing a sample
ex=next(iter(train_dataloader))
sample,label=ex
print(sample[0].shape)
print(label[0].item())


print(len(test_dataloader))

print(len(test_dataloader.dataset))

#displaying image
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(sample[i][0],cmap='gray')
    plt.title(f"Ground Truth: {label[i].item()}")

plt.show()

