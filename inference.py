import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2
from cnn import CNN
import os

# Check GPU Availability. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for Inference.")

# Reading Image for inference
image_path = "./data/image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)
height, width, channel = image.shape
print(f"Image shape : {height, width, channel}")

# Output path
output_path = "./output_5xconv_maxpool_relu"
if not os.path.exists(output_path):
    os.mkdir(output_path)

# Convert numpy ndarray image to pytorch tensor.
transform = transforms.ToTensor()
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

# loading image tensor to GPU
image_tensor = image_tensor.to(device)

# Instantiate cnn 
CNN = CNN(channel)
CNN.eval()
CNN.to(device)


# Running Inference
pred = CNN(image_tensor).squeeze(0)
print(f"Output Shape : ({pred.shape})")

# Saving output
for id, output in enumerate(pred):
    save_image(output, f"{output_path}/{id}.jpg")

