import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
from lib.model.EncDecModel import EncDec
# from lib.model.DilatedNetModel import DilatedNet
from lib.model.UNetModel import UNet
from lib.losses import BCELoss, DiceLoss, FocalLoss, WeightedBCELoss #BCELoss_TotalVariation
from lib.dataset.PH2Dataset import PH2Dataset
from lib.metrics import dice_coefficient, pixel_accuracy, iou_score, specificity, sensitivity
from create_split import SegmentationDataManager

import csv
import os

MODEL_NAME = "EncDec"  # "EncDec" or "UNet"
LOSS_FNC = "WBCE"  # "BCE", "Dice", "Focal", "WBCE"

csv_filename = MODEL_NAME + "_" + LOSS_FNC + "_segmentation_metrics_log.csv"
csv_header = ["Epoch", "Dice_Coefficient", "IoU", "Accuracy", "Sensitivity", "Specificity", "Val. Dice_Coefficient", "Val. IoU", "Val. Accuracy", "Val. Sensitivity", "Val. Specificity"]

# Create the file with headers if it doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)


def log_metrics(epoch, dice, iou, acc, sens, spec, val_dice, val_iou, val_acc, val_sens, val_spec):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, dice, iou, acc, sens, spec, val_dice, val_iou, val_acc, val_sens, val_spec])


DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images/'

# Dataset
size = 128
# train_transform = transforms.Compose([transforms.Resize((size, size)),
#                                     transforms.ToTensor()])
# test_transform = transforms.Compose([transforms.Resize((size, size)),
#                                     transforms.ToTensor()])

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



data_manager = SegmentationDataManager(DATA_PATH)
train_paths, val_paths, test_paths = data_manager.split()


batch_size = 6
trainset = PH2Dataset(file_paths=train_paths, image_transform=image_transform, label_transform=label_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                          num_workers=0)
testset = PH2Dataset(file_paths=test_paths, image_transform=image_transform, label_transform=label_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                          num_workers=0)
valset = PH2Dataset(file_paths=val_paths, image_transform=image_transform, label_transform=label_transform)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                          num_workers=0)


print(f"Loaded {len(trainset)} training images")
print(f"Loaded {len(testset)} test images")
print(f"Loaded {len(valset)} validation images")

# # Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if MODEL_NAME == "EncDec":
    model = EncDec().to(device)
elif MODEL_NAME == "UNet":
    model = UNet().to(device)

learning_rate = 0.001
opt = optim.Adam(model.parameters(), learning_rate)

# Choose loss function
if LOSS_FNC == "BCE":
    loss_fn = BCELoss()
elif LOSS_FNC == "Dice":
    loss_fn = DiceLoss()
elif LOSS_FNC == "Focal":
    loss_fn = FocalLoss()
elif LOSS_FNC == "WBCE":
    loss_fn = WeightedBCELoss(pos_weight=torch.tensor(2.0))
else:
    raise ValueError(f"Unsupported LOSS_FNC: {LOSS_FNC}")
#elif LOSS_FNC == "BCE_TV":
#    loss_fn = BCELoss_TotalVariation()


epochs = 100

# Training loop
model.train()  # train mode
for epoch in range(epochs):
    tic = time()
    print(f'* Epoch {epoch+1}/{epochs}')

    avg_loss = 0
    avg_loss_test = 0
    epoch_dice = 0.0
    epoch_iou = 0.0
    epoch_acc = 0.0
    epoch_sensitivity = 0.0
    epoch_specificity = 0.0
    val_epoch_dice = 0.0
    val_epoch_iou = 0.0
    val_epoch_acc = 0.0
    val_epoch_sensitivity = 0.0
    val_epoch_specificity = 0.0
    for X_batch, y_true, _ in train_loader:
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)

        # set parameter gradients to zero
        opt.zero_grad()

        # forward
        y_pred = model(X_batch)
        # IMPORTANT NOTE: Check whether y_pred is normalized or unnormalized
        # and whether it makes sense to apply sigmoid or softmax.
        y_pred = F.sigmoid(y_pred)
        
        loss = loss_fn(y_pred, y_true)  # forward-pass
        loss.backward()  # backward-pass
        opt.step()  # update weights

        # calculate metrics to show the user
        avg_loss += loss / len(train_loader)


        # Compute Dice Coefficient for this batch
        preds = y_pred > 0.5
        dice = dice_coefficient(preds, y_true)
        epoch_dice += dice

        iou_score_value = iou_score(preds, y_true)
        epoch_iou += iou_score_value

        acc = pixel_accuracy(preds, y_true)
        epoch_acc += acc.item()

        sensitivity_value = sensitivity(preds, y_true)
        epoch_sensitivity += sensitivity_value

        specificity_value = specificity(preds, y_true)
        epoch_specificity += specificity_value


    # IMPORTANT NOTE: It is a good practice to check performance on a
    # validation set after each epoch.
    model.eval()  # testing mode    
    with torch.no_grad():
        for X_val, Y_val, _ in val_loader:
            Y_hat = F.sigmoid(model(X_val.to(device))).detach().cpu()
            # preds = (Y_hat > 0.5).float().squeeze().cpu().numpy()  # Binarize predictions
            preds = Y_hat > 0.5
            
            # Y_hat = F.sigmoid(model(X_val.to(device))).detach().cpu()
            # loss_val = loss_fn(Y_hat, Y_val)  # forward-pass
            # avg_loss_val += loss_val / len(val_loader)

            # Calculate metrics on validation set
            y_true = Y_val
            dice = dice_coefficient(preds, y_true)
            val_epoch_dice += dice

            iou_score_value = iou_score(preds, y_true)
            val_epoch_iou += iou_score_value

            acc = pixel_accuracy(preds, y_true)
            val_epoch_acc += acc.item()

            sensitivity_value = sensitivity(preds, y_true)
            val_epoch_sensitivity += sensitivity_value

            specificity_value = specificity(preds, y_true)
            val_epoch_specificity += specificity_value
    print(f' - loss: {avg_loss}')
    # print(f' - val_loss: {avg_loss_val}')
    #model.eval()  # testing mode
    #Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
    # print(f' - loss: {avg_loss}')
    avg_dice = epoch_dice / len(train_loader)
    print(f' - dice_coeff: {avg_dice}')
    avg_iou = epoch_iou / len(train_loader)
    print(f' - iou: {avg_iou}')
    avg_acc = epoch_acc / len(train_loader)
    print(f' - pixel_acc: {avg_acc}')
    avg_sens = epoch_sensitivity / len(train_loader)
    print(f' - sensitivity: {avg_sens}')
    avg_spec = epoch_specificity / len(train_loader)
    print(f' - specificity: {avg_spec}')


    val_avg_dice = val_epoch_dice / len(val_loader)
    val_avg_iou = val_epoch_iou / len(val_loader)
    val_avg_acc = val_epoch_acc / len(val_loader)
    val_avg_sens = val_epoch_sensitivity / len(val_loader)
    val_avg_spec = val_epoch_specificity / len(val_loader)

    # Log metrics to CSV.
    log_metrics(epoch, avg_dice, avg_iou, avg_acc, avg_sens, avg_spec, val_avg_dice, val_avg_iou, val_avg_acc, val_avg_sens, val_avg_spec)

# Save the model
save_fname = "model_project3_derm_" + MODEL_NAME.lower() + "_" + LOSS_FNC.lower() + ".pth"
torch.save(model.state_dict(), save_fname)

print("Training has finished!")