import torch
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
from PIL import Image
from torchvision.ops import nms
import random

# Import your model
from lib.model.cnn import PotholeCNN

# Configuration
TEST_DATA_PATH = '/dtu/datasets1/02516/potholes' 
ANNOT_FOLDER = os.path.join(TEST_DATA_PATH, 'annotations')
MODEL_PATH = 'project4/experiments/cnn_baseline_v1/best_model.pth'
OUTPUT_VIS_DIR = 'project4/experiments/cnn_baseline_v1'
IMG_SIZE = 64
CONF_THRESHOLD = 0.9  
NMS_THRESHOLD = 0.4   

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

# Define Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_ground_truth_boxes(xml_path):
    """Parses XML and returns pixel coordinates [xmin, ymin, xmax, ymax]"""
    if not os.path.exists(xml_path):
        return []
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # We don't need to normalize here, we want pixel values for plotting
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes

def get_selective_search_boxes(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

def detect_potholes(img_path, model):
    original_img = cv2.imread(img_path)
    if original_img is None: return np.empty((0, 4)), np.array([])
        
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Selective Search (Resize for speed)
    resize_w, resize_h = 256, 256
    ss_img = cv2.resize(original_img, (resize_w, resize_h))
    rects = get_selective_search_boxes(ss_img)
    
    rects = rects[:500] # Top 500
    
    scale_x = original_img.shape[1] / resize_w
    scale_y = original_img.shape[0] / resize_h
    
    proposals = []
    batch_tensors = []
    
    for (x, y, w, h) in rects:
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        w_real = int(w * scale_x)
        h_real = int(h * scale_y)
        x2 = x1 + w_real
        y2 = y1 + h_real
        
        crop = pil_img.crop((x1, y1, x2, y2))
        batch_tensors.append(transform(crop))
        proposals.append([x1, y1, x2, y2])
        
    if len(batch_tensors) == 0:
        return np.empty((0, 4)), np.array([])

    # Classify
    input_batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = model(input_batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    pothole_scores = probs[:, 1].cpu().numpy()
    proposals = np.array(proposals)
    
    # Filter
    keep_indices = pothole_scores > CONF_THRESHOLD
    final_boxes = proposals[keep_indices]
    final_scores = pothole_scores[keep_indices]
    
    if final_boxes.ndim == 1 and len(final_boxes) > 0:
        final_boxes = final_boxes.reshape(1, 4)
    
    return final_boxes, final_scores

def apply_nms(boxes, scores, iou_thresh=0.3):
    if len(boxes) == 0: return np.empty((0, 4)), np.array([])
    if boxes.ndim == 1: boxes = boxes.reshape(1, 4)

    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    keep_indices = nms(box_tensor, score_tensor, iou_thresh)
    keep_indices_np = keep_indices.cpu().numpy()
    
    return boxes[keep_indices_np], scores[keep_indices_np]

def visualize_and_save(img_path, pred_boxes, pred_scores, gt_boxes, output_path):
    img = cv2.imread(img_path)
    
    # 1. Draw Ground Truth (Green)
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, "GT", (int(x1), int(y1)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 2. Draw Predictions (Red)
    if pred_boxes.ndim == 1: pred_boxes = pred_boxes.reshape(1, 4)
        
    for i, (x1, y1, x2, y2) in enumerate(pred_boxes):
        score = pred_scores[i] if i < len(pred_scores) else 0.0
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), int(y2)+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
    cv2.imwrite(output_path, img)

# --- MAIN ---
if __name__ == "__main__":
    model = PotholeCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")

    img_folder = os.path.join(TEST_DATA_PATH, 'images')
    all_images = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])
    random.seed(42)
    random.shuffle(all_images)
    test_images = all_images[int(0.8 * len(all_images)):]
    print(f"Found {len(test_images)} test images.")

    print(f"Running inference on 10 samples. Saving to {OUTPUT_VIS_DIR}...")

    # Run on first 10 images for better variety
    for i, img_name in enumerate(test_images[:10]): 
        img_path = os.path.join(img_folder, img_name)
        xml_path = os.path.join(ANNOT_FOLDER, os.path.splitext(img_name)[0] + '.xml')
        
        # A. Get Ground Truth
        gt_boxes = get_ground_truth_boxes(xml_path)
        
        # B. Detect
        boxes, scores = detect_potholes(img_path, model)
        
        # C. NMS
        nms_boxes, nms_scores = apply_nms(boxes, scores, iou_thresh=NMS_THRESHOLD)
        
        # D. Visualize
        out_name = os.path.join(OUTPUT_VIS_DIR, f"vis_{img_name}")
        visualize_and_save(img_path, nms_boxes, nms_scores, gt_boxes, out_name)

    print("Done.")