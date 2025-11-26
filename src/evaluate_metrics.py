import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
from PIL import Image
from torchvision.ops import nms, box_iou
import random

# Import your model
from lib.model.cnn import PotholeCNN

# --- CONFIGURATION ---
TEST_DATA_PATH = '/dtu/datasets1/02516/potholes' 
ANNOT_FOLDER = os.path.join(TEST_DATA_PATH, 'annotations')
MODEL_PATH = 'project4/experiments/cnn_baseline_v1/best_model.pth'
OUTPUT_DIR = 'project4/experiments/cnn_baseline_v1'

# Parameters for evaluation
IOU_THRESHOLD = 0.5       # Standard Pascal VOC requirement: Box must overlap 50% to be correct
CONF_THRESHOLD = 0.1     # KEEP LOW! We want all detections for the PR-Curve, not just the good ones.
NMS_THRESHOLD = 0.3       # To clean up duplicates before evaluation

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- HELPER FUNCTIONS (Similar to Inference) ---

def get_ground_truth_boxes(xml_path):
    if not os.path.exists(xml_path): return np.empty((0, 4))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        boxes.append([
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)
        ])
    return np.array(boxes, dtype=float)

def get_selective_search_boxes(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process() # x, y, w, h
    return rects

def detect_image(img_path, model):
    original_img = cv2.imread(img_path)
    if original_img is None: return np.empty((0, 4)), np.array([])

    # SS
    resize_w, resize_h = 256, 256
    ss_img = cv2.resize(original_img, (resize_w, resize_h))
    rects = get_selective_search_boxes(ss_img)[:500] # Limit to 500
    
    scale_x = original_img.shape[1] / resize_w
    scale_y = original_img.shape[0] / resize_h
    
    # Prepare Batch
    batch_tensors = []
    proposals = []
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    for (x, y, w, h) in rects:
        x1, y1 = int(x * scale_x), int(y * scale_y)
        w_real, h_real = int(w * scale_x), int(h * scale_y)
        x2, y2 = x1 + w_real, y1 + h_real
        
        crop = pil_img.crop((x1, y1, x2, y2))
        batch_tensors.append(transform(crop))
        proposals.append([x1, y1, x2, y2])
        
    if not batch_tensors: return np.empty((0, 4)), np.array([])

    # Inference
    input_batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = model(input_batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    scores = probs[:, 1].cpu().numpy()
    boxes = np.array(proposals)
    
    # Filter by Low Confidence (keep more for PR curve)
    keep = scores > CONF_THRESHOLD
    boxes = boxes[keep]
    scores = scores[keep]
    
    # Apply NMS
    if len(boxes) > 0:
        if boxes.ndim == 1: boxes = boxes.reshape(1, 4)
        t_boxes = torch.tensor(boxes, dtype=torch.float32)
        t_scores = torch.tensor(scores, dtype=torch.float32)
        keep_idxs = nms(t_boxes, t_scores, NMS_THRESHOLD)
        boxes = boxes[keep_idxs.numpy()]
        scores = scores[keep_idxs.numpy()]
        
    return boxes, scores

# --- METRIC CALCULATION ---

def calculate_ap(all_detections, all_annotations):
    """
    all_detections: List of [image_id, confidence, x1, y1, x2, y2]
    all_annotations: Dict mapping image_id -> array of GT boxes
    """
    # 1. Sort detections by confidence (High -> Low)
    all_detections.sort(key=lambda x: x[1], reverse=True)
    
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    
    # Track which GT boxes have already been matched (to avoid double counting)
    gt_matched = {img_id: np.zeros(len(boxes)) for img_id, boxes in all_annotations.items()}
    
    # 2. Loop through all detections
    for i, det in enumerate(all_detections):
        img_id = det[0]
        confidence = det[1]
        bb_det = det[2:] # x1, y1, x2, y2
        
        gt_boxes = all_annotations[img_id]
        
        # If image has no GT, this is a False Positive
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
            
        # Find best overlapping GT
        bb_gt_tensor = torch.tensor(gt_boxes, dtype=torch.float)
        bb_det_tensor = torch.tensor([bb_det], dtype=torch.float)
        
        ious = box_iou(bb_det_tensor, bb_gt_tensor).squeeze(0) # [num_gt]
        
        max_iou, max_idx = torch.max(ious, 0)
        max_iou = max_iou.item()
        max_idx = max_idx.item()
        
        if max_iou >= IOU_THRESHOLD:
            # Check if this GT was already used
            if gt_matched[img_id][max_idx] == 0:
                tp[i] = 1
                gt_matched[img_id][max_idx] = 1 # Mark as used
            else:
                fp[i] = 1 # Duplicate detection for same object
        else:
            fp[i] = 1 # IoU too low
            
    # 3. Compute Cumulative Sums
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    total_positives = sum([len(b) for b in all_annotations.values()])
    
    # 4. Precision and Recall
    recalls = cum_tp / total_positives
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    
    # 5. Average Precision (Area Under Curve)
    # We use the 11-point interpolation or simple integration. 
    # Here we use standard integration.
    ap = np.trapz(precisions, recalls)
    
    # Fix zig-zags for plotting (standard practice)
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
    return recalls, precisions, ap, total_positives, cum_tp[-1], cum_fp[-1]

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("--- STARTING EVALUATION ---")
    
    # 1. Load Model
    model = PotholeCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Get Test Split (Same Seed)
    img_folder = os.path.join(TEST_DATA_PATH, 'images')
    all_images = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])
    random.seed(42)
    random.shuffle(all_images)
    test_images = all_images[int(0.8 * len(all_images)):]
    
    print(f"Evaluating on {len(test_images)} test images...")

    # 3. Collect Data
    all_detections = [] # List of [img_id, score, x1, y1, x2, y2]
    all_annotations = {} # Dict {img_id: [[x1,y1,x2,y2], ...]}

    for i, img_name in enumerate(test_images):
        # Progress
        if (i+1) % 10 == 0: print(f"Processing {i+1}/{len(test_images)}...")
        
        img_id = img_name
        img_path = os.path.join(img_folder, img_name)
        xml_path = os.path.join(ANNOT_FOLDER, os.path.splitext(img_name)[0] + '.xml')
        
        # Store GT
        gt = get_ground_truth_boxes(xml_path)
        all_annotations[img_id] = gt
        
        # Run Inference
        boxes, scores = detect_image(img_path, model)
        
        # Store Detections
        for b, s in zip(boxes, scores):
            all_detections.append([img_id, s, b[0], b[1], b[2], b[3]])

    print(f"\nInference Complete. Total Detections: {len(all_detections)}")
    
    # 4. Calculate AP
    recalls, precisions, ap, total_gt, total_tp, total_fp = calculate_ap(all_detections, all_annotations)
    
    print("\n--- RESULTS ---")
    print(f"Total Ground Truth Potholes: {total_gt}")
    print(f"Total True Positives (at IoU=0.5): {int(total_tp)}")
    print(f"Total False Positives: {int(total_fp)}")
    print(f"AVERAGE PRECISION (AP): {ap:.4f}")

    # 5. Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color='blue', lw=2, label=f'AP = {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Pothole Detection)')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    save_path = os.path.join(OUTPUT_DIR, 'pr_curve.png')
    plt.savefig(save_path)
    print(f"PR Curve saved to {save_path}")