# imports the model architecture
# loads the saved weights: Use torch.load function
# loads the test set of a DatasetLoader (see train.py)
# Iterate over the test set images, generate predictions, save segmentation masks

import os
from PIL import Image
import numpy as np
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
import torch
from lib.dataset.PH2Dataset import PH2Dataset
from create_split import SegmentationDataManager
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def save_mask(array, path):
    # array should be a 2D numpy array with 0s and 1s
    np.unique(array) == [0, 1]
    len(np.shape(array)) == 2
    im_arr = (array*255)
    Image.fromarray(np.uint8(im_arr)).save(path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose the model.
MODEL_NAME = "UNet"  # "EncDec" or "UNet"
LOSS_FNC = "WBCE"  # "BCE", "Dice", "Focal", "BCE_TV"

if MODEL_NAME == "EncDec":
    model = EncDec().to(device)
elif MODEL_NAME == "UNet":
    model = UNet().to(device)

# Load the saved model weights
model_path = "model_project3_derm_" + MODEL_NAME.lower() + "_" + LOSS_FNC.lower() + ".pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# Load the test set
size = 128
batch_size = 6
DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images/'
data_manager = SegmentationDataManager(DATA_PATH)
train_paths, val_paths, test_paths = data_manager.split()

image_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

label_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0).float())  # Converts 255 to 1.0
])
testset = PH2Dataset(file_paths=test_paths, image_transform=image_transform, label_transform=label_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                         num_workers=0)

# Iterate over the test set and generate predictions
# for i, (images, _) in enumerate(test_loader):
for images, _, image_names in test_loader:
    with torch.no_grad():
        outputs = model(images)
        y_pred = F.sigmoid(outputs)
        preds = (y_pred > 0.5).float().squeeze().cpu().numpy()  # Binarize predictions

    # Save each mask
    for j in range(preds.shape[0]):
        # Save mask in the folder with the original image name
        save_folder = os.path.join(r"/zhome/ff/6/51582/02516_project3/results", MODEL_NAME, LOSS_FNC)
        save_path = os.path.join(save_folder, f"pred_{image_names[j]}")
        save_mask(preds[j], save_path)
        print(f"Saved {save_path}")


# print(test_paths)
# weights = torch.load("model_project3_derm_encdec.pth", weights_only=True)

# # Print all layer names and their shapes
# for name, param in weights.items():
#     print(f"{name}: {param}")
