import os
import sys
import requests
import random
from zipfile import ZipFile, BadZipFile
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

def system_config(SEED_VALUE=42, package_list=None):
    """Configure system settings"""
    if torch.cuda.is_available():
        print('Using CUDA GPU')
        random.seed(SEED_VALUE)
        np.random.seed(SEED_VALUE)
        torch.manual_seed(SEED_VALUE)
        DEVICE = torch.device("cuda:0")  # Simplified device setting
        print("Device: ", DEVICE)
        GPU_AVAILABLE = True
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print(f"Using Apple Silicon GPU")
        DEVICE = torch.device("mps")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        GPU_AVAILABLE = True
        torch.mps.manual_seed(SEED_VALUE)
        torch.use_deterministic_algorithms(True)
    else:
        print('Using CPU')
        DEVICE = torch.device("cpu")
        print("Device: ", DEVICE)
        GPU_AVAILABLE = False
        torch.use_deterministic_algorithms(True)

    return str(DEVICE), GPU_AVAILABLE

DEVICE, GPU_AVAILABLE = system_config()

def download_file(url, save_name):
    """Download a file from a URL"""
    response = requests.get(url, stream=True)
    with open(save_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk: 
                file.write(chunk)
    print(f"Downloaded: {save_name}") 

def unzip(zip_file_path=None):
    """Unzip a file"""
    try:
        with ZipFile(zip_file_path, 'r') as z:
            z.extractall("./")
            print(f"Extracted: {os.path.splitext(zip_file_path)[0]}\n")
    except FileNotFoundError:
        print("File not found")
    except BadZipFile:
        print("Invalid or corrupt zip file")
    except Exception as e:
        print(f"Error occurred: {e}")
    return

URL = r"<insert url here.>"
archive_name = "<archive name here.>"
zip_name = f"./{archive_name}.zip"
if not os.path.exists(archive_name):
    download_file(URL, zip_name)
    unzip(zip_name)

train_root = os.path.join("<file_name>", "training", "training")  # Load the training data
train_data = datasets.ImageFolder(root=train_root) 
print("Classes:", train_data.classes)
print("Class to Index Mapping:", train_data.class_to_idx) 
print("Data Length:", len(train_data)) 

# Image 1
img, target = train_data[5]  # Get image and target
print('PIL image size: {}, target: {}'.format(img.size, target)) 
plt.imshow(img)
plt.show()

# Image 2
img, target = train_data[500] 
print('image size: {}, target: {}'.format(img.size, target))
plt.imshow(img)
plt.show()

"""
Each image has a different size, but for training, we need fixed-size images. 
So, we can not use these images as they are for training. We can use 'torchvision.transforms.Resize'
to resize images.
"""

# transform=transforms.Resize(224, 224) Resizes all images to 224 x 224
train_data = datasets.ImageFolder(root=train_root, transform=transforms.Resize((224, 224)))

# Image 1
img, target = train_data[5]
print('image size: {}, target: {}'.format(img.size, target))
plt.imshow(img)
plt.show()

# Image 2
img, target = train_data[500]
print('image size: {}, target: {}'.format(img.size, target))
plt.imshow(img)
plt.show()

"""
Even after resizing our images, Image 1 is showing up as distorted. The ratio of (width:height)
is not programmatically taken into account with the resizing calculation alone. 

To fix this, pass an integer instead of a tuple argument through transforms.Resize. This will resize 
the lower pixel value to the given integer value, and the higher pixel value will be such that it will maintain the aspect ratio. 
"""

train_data = datasets.ImageFolder(root=train_root, transform=transforms.Resize(224))
img, target = train_data[5]
print('image size: {}, target: {}'.format(img.size, target))
plt.imshow(img)  
plt.show()

# Data Loader with Image Folder Dataset
def get_data(batch_size, data_root, num_workers=4):
    """Get data loaders for training and testing"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor()
    ])
    # Train Dataloader
    train_data_path = os.path.join(data_root, 'training', 'training')
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=train_data_path, transform=preprocess),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers  
    )
    # Test Dataloader
    test_data_path = os.path.join(data_root, 'validation', 'validation')
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=test_data_path, transform=preprocess),  
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader

# Plot few Images
data_root = "10_Money_Species"
train_loader, test_loader = get_data(10, data_root)

plt.rcParams["figure.figsize"] = (15, 6) 
for images, labels in train_loader:
    for i in range(len(labels)):
        plt.subplot(2, 5, i+1)
        img = transforms.functional.to_pil_image(images[i])
        plt.imshow(img)
        plt.axis('off')
        plt.gca().set_title('Target: {0}'.format(labels[i]))
    plt.show()
    break
