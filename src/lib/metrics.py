import torch



def dice_coefficient(pred, target, epsilon=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    
    return dice.item()


def iou_score(pred, target, epsilon=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def pixel_accuracy(pred, target):
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return correct / total


def sensitivity(pred, target, epsilon=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    TP = ((pred == 1) & (target == 1)).sum()
    FN = ((pred == 0) & (target == 1)).sum()
    return ((TP + epsilon) / (TP + FN + epsilon)).item()

def specificity(pred, target, epsilon=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    TN = ((pred == 0) & (target == 0)).sum()
    FP = ((pred == 1) & (target == 0)).sum()
    return ((TN + epsilon) / (TN + FP + epsilon)).item()
