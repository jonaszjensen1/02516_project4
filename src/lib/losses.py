import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn(y_pred, y_true)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = 1 - (torch.mean(2 * y_true * y_pred) + 1) / (torch.mean(y_true + y_pred) + 1)
        return loss



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # y_pred: logits (not probabilities)
        # y_true: binary labels (0 or 1)
        # probs = torch.sigmoid(y_pred)
        probs = y_pred
        pt = torch.where(y_true == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        loss = - (self.alpha * y_true * torch.log(probs + 1e-8) +
                  (1 - self.alpha) * (1 - y_true) * torch.log(1 - probs + 1e-8))
        loss = focal_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true):
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        return loss_fn(y_pred, y_true)
