import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import torchvision.transforms as transforms
from PIL import Image

class PotholeDataset(Dataset):
    def __init__(self, data_file, transform=None):
        """
        Args:
            data_file (str): Path to the .pt file generated in Part 1
            transform (callable): Augmentations/Normalizations
        """
        self.data = torch.load(data_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample['img_path']
        label = sample['label']
        box = sample['box'] # [xmin, ymin, xmax, ymax] normalized
        
        try:
            # Load image
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                
                # Denormalize coordinates
                xmin = int(box[0] * w)
                ymin = int(box[1] * h)
                xmax = int(box[2] * w)
                ymax = int(box[3] * h)
                
                # Clamp coordinates to image boundaries
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)
                
                # Crop logic
                if xmax <= xmin or ymax <= ymin:
                    # Fallback for invalid boxes
                    crop = img
                else:
                    crop = img.crop((xmin, ymin, xmax, ymax))
                
                if self.transform:
                    crop = self.transform(crop)
                    
                return crop, torch.tensor(label, dtype=torch.long)
                
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor if file fails (shouldn't happen)
            return torch.zeros((3, 64, 64)), torch.tensor(0, dtype=torch.long)

def get_dataloaders(data_path, batch_size=64, num_workers=4):
    # 1. Define Transforms (Resize to 64x64 is mandatory for our CNN)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Load Full Dataset
    full_dataset = PotholeDataset(data_path, transform=transform)
    
    # 3. Split Train (80%) / Val (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # 4. Handle Class Imbalance (Weighted Sampling for Training)
    print("Calculating class weights for sampling...")
    train_indices = train_set.indices
    # Access labels directly from the underlying data list
    train_labels = [full_dataset.data[i]['label'] for i in train_indices]
    
    class_counts = {0: 0, 1: 0}
    for l in train_labels:
        class_counts[l] += 1
        
    print(f"Train set balance: Background={class_counts[0]}, Pothole={class_counts[1]}")
    
    weight_0 = 1.0 / class_counts[0] if class_counts[0] > 0 else 0
    weight_1 = 1.0 / class_counts[1] if class_counts[1] > 0 else 0
    
    samples_weights = [weight_1 if l == 1 else weight_0 for l in train_labels]
    samples_weights = torch.tensor(samples_weights, dtype=torch.double)
    
    sampler = WeightedRandomSampler(weights=samples_weights, 
                                    num_samples=len(samples_weights), 
                                    replacement=True)
    
    # 5. Create Loaders
    # Note: shuffle=False is required when using a sampler
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    
    # Validation loader does NOT need sampling (we want to measure real accuracy)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader