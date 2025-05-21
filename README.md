# DL- Convolutional Autoencoder for Image Denoising
## AIM
To develop a convolutional autoencoder for image denoising application.

## THEORY
A convolutional autoencoder is a neural network that learns to compress (encode) an image into a lower‐dimensional representation and then reconstruct (decode) it back to its original size. By training on pairs of noisy and clean images, the encoder learns to extract robust feature maps that ignore noise, while the decoder uses these features to reconstruct the denoised image. Convolutional layers preserve spatial structure, making this approach well suited for image‐level tasks like denoising.

## DESIGN STEPS
### STEP 1: 
Problem Understanding and Dataset Selection

### STEP 2: 
Preprocessing the Dataset
 
### STEP 3: 
Design the Convolutional Autoencoder Architecture

### STEP 4: 
Compile and Train the Model

### STEP 5: 
Evaluate the Model

### STEP 6: 
Visualization and Analysis

## PROGRAM
### Name: Elamaran S E
### Register Number: 212222230036
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
transforms. ToTensor()
])
dataset = datasets.MNIST (root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST (root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader (test_dataset, batch_size=128, shuffle=False)
def add_noise(inputs, noise_factor=0.5):
  noisy = inputs + noise_factor * torch.randn_like (inputs)
  return torch.clamp (noisy, 0., 1.)
class DenoisingAutoencoder(nn.Module):
  def __init__(self):
    super(DenoisingAutoencoder, self).__init__() # Changed _init_ to __init__
    self.encoder = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, output_padding=1, padding=1),
        nn.Sigmoid()
    )

  def forward(self, x): # Added the forward method to the class
    x = self.encoder(x)
    x = self.decoder(x)
    return x
def add_noise(inputs, noise_factor=0.5):
  noisy = inputs + noise_factor * torch.randn_like (inputs)
  return torch.clamp (noisy, 0., 1.)

def forward(self, x):
  x = self.encoder(x)
  x = self.decoder(x)
  return x
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("Name:Elamaran S E ")
print("Register Number:212222230036 ")
summary(model, input_size=(1, 28 , 28))
def train(model, loader, criterion, optimizer, epochs=5):
  model.train()
  print("Name: Elamaran S E ")
  print("Register Number:212222230036")
  for epoch in range(epochs):
    running_loss = 0.0
    for images, _ in loader:
      images = images.to(device)
      noisy_images = add_noise(images).to(device)

      outputs = model(noisy_images)
      loss = criterion(outputs, images)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
def visualize_denoising(model, loader, num_images = 10):
  model.eval()
  with torch.no_grad():
    for images, _ in loader:
      images = images.to(device)
      noisy_images = add_noise(images).to(device)
      outputs = model(noisy_images)
      break

  images = images.cpu().numpy()
  noisy_images = noisy_images.cpu().numpy()
  outputs = outputs.cpu().numpy()

  print("Name: Elamaran S E")
  print("Register Number: 212222230036")
  plt.figure(figsize=(18,6))
  for i in range(num_images):
    ax = plt.subplot(3,num_images, i + 1 )
    plt.imshow(images[i].squeeze(), cmap = 'gray')
    ax.set_title("Original")
    plt.axis("off")

    ax = plt.subplot(3, num_images, i +1 +num_images)
    plt.imshow(noisy_images[i].squeeze(), cmap = 'gray')
    ax.set_title("Noisy")
    plt.axis("off")

    ax = plt.subplot(3, num_images, i +1 +num_images*2)
    plt.imshow(outputs[i].squeeze(), cmap = 'gray')
    ax.set_title("Denoised")
    plt.axis("off")

  plt.tight_layout()
  plt.show()
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```
### OUTPUT
### Model Summary
![image](https://github.com/user-attachments/assets/e25b5b8d-35aa-429b-b76b-ca63c3d9ef7b)

### Training loss
![image](https://github.com/user-attachments/assets/8bc4ac9b-ac02-4211-9fb8-c2079d97d156)

### Original vs Noisy Vs Reconstructed Image
![image](https://github.com/user-attachments/assets/578c6ebd-f22e-4326-a0a3-0c17865252d9)

## RESULT
Thus, develop a convolutional autoencoder for image denoising application excuted succesfully
