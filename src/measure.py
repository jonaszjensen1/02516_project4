# Here, you can load the predicted segmentation masks, and evaluate the
# performance metrics (accuracy, etc.)

# Load the predicted masks from the predict.py output folder
# Load the ground truth masks from the dataset
# Compute and print the performance metrics

import os
import re
from PIL import Image
import numpy as np
from lib.metrics import dice_coefficient, pixel_accuracy, iou_score, specificity, sensitivity
import torch


def load_mask(path):
    im = Image.open(path)  # Convert to grayscale
    im = im.resize((128, 128))  # Resize to match prediction size
    im_arr = np.array(im)
    # Binarize if not already binary.
    if im_arr.max() > 1:
        im_arr = (im_arr > 127).astype(np.uint8)  # Binarize
    # If boolean, convert to uint8
    if im_arr.dtype == np.bool_:
        im_arr = im_arr.astype(np.uint8)
    return im_arr

MODEL_NAME = "UNet"  # "EncDec" or "UNet"
LOSS_FN = "WBCE"  # "BCE", "Dice", "Focal", "BCE_TV"
DATA_PATH_GT = '/dtu/datasets1/02516/PH2_Dataset_images/'
DATA_PATH_PRED = '/zhome/ff/6/51582/02516_project3'

# Loop through the images in the folder where predicted masks are saved.
predicted_folder = os.path.join(DATA_PATH_PRED, 'results', MODEL_NAME, LOSS_FN)  # Folder where predicted masks are saved

avr_dice = 0.0
avr_iou = 0.0
avr_acc = 0.0
avr_sens = 0.0
avr_spec = 0.0
count = 0

for filename in os.listdir(predicted_folder):
    if filename.endswith('.bmp'):
        pred_path = os.path.join(predicted_folder, filename)

        match = re.search(r"IMD(\d+)", filename)
        if match:
            number = match.group(1)
            gt_id = f'IMD{number}'
            gt_path = os.path.join(DATA_PATH_GT, gt_id, gt_id + '_lesion', gt_id + '_lesion' + '.bmp')

            gt_mask = load_mask(gt_path)

        pred_mask = load_mask(pred_path)
        # 

        # # Convert to torch tensors
        pred_tensor = torch.tensor(pred_mask)
        gt_tensor = torch.tensor(gt_mask)

        # Calculate metrics
        dice = dice_coefficient(pred_tensor, gt_tensor)
        iou = iou_score(pred_tensor, gt_tensor)
        acc = pixel_accuracy(pred_tensor, gt_tensor)
        sens = sensitivity(pred_tensor, gt_tensor)
        spec = specificity(pred_tensor, gt_tensor)

        # Print the results
        # print(f'File: {filename} | Dice: {dice:.4f} | IoU: {iou:.4f} | Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}')

        # Calculate average metrics over all images
        avr_dice += dice
        avr_iou += iou
        avr_acc += acc
        avr_sens += sens
        avr_spec += spec
        count += 1

# Print average metrics
print(f'Average Dice: {avr_dice/count:.4f} | Average IoU: {avr_iou/count:.4f} | Average Acc: {avr_acc/count:.4f} | Average Sens: {avr_sens/count:.4f} | Average Spec: {avr_spec/count:.4f}')
    