import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import from our lib folders
# Since this script is in src/, and lib is in src/lib, we can import like this:
from lib.model.cnn import PotholeCNN
from lib.dataset.dataloader import get_dataloaders

def train(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data
    print(f"Loading data from {args.data_path}...")
    train_loader, val_loader = get_dataloaders(args.data_path, args.batch_size)
    
    # 3. Model
    model = PotholeCNN().to(device)
    
    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Arrays to store metrics
    train_losses, val_accuracies = [], []
    best_acc = 0.0
    
    # 5. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 6. Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        val_accuracies.append(acc)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f} | Val Acc: {acc:.2f}%")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            # print(f"Saved best model to {save_path}")

    # 7. Plotting Results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plot_path = os.path.join(args.output_dir, 'training_plot.png')
    plt.savefig(plot_path)
    print(f"Training finished. Best Val Acc: {best_acc:.2f}%. Plot saved to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pothole Detector')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .pt file')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Where to save results')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)