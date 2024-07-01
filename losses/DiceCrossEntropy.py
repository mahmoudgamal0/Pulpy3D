import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, num_classes, smooth=1e-5):
    # target_one_hot and pred are both [B C P P P]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    intersection = torch.sum(pred * target_one_hot, dim=(0, 2, 3, 4))
    cardinality = torch.sum(pred + target_one_hot, dim=(0, 2, 3, 4))

    dice_scores = (2.0 * intersection + smooth) / (cardinality + smooth)
    average_dice = torch.mean(dice_scores)
    dice_loss = 1.0 - average_dice
    return dice_loss


def softmax_dice_loss(pred, target, num_classes, ce_weight=0.5, dice_weight=0.5, partition_weights=None):
    # Target is GT
    # Pred are Probabilities and not Logits
    target = target.squeeze(1).long()
    ce_loss = nn.CrossEntropyLoss(weight=partition_weights)(pred, target)
    dice_loss_val = dice_loss(pred, target, num_classes)
    loss = ce_weight * ce_loss + dice_weight * dice_loss_val
    return loss


class DiceCrossEntropy(torch.nn.Module):
    def __init__(self, weight=None, num_classes=1):
        super().__init__()
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, pred, gt):
      return softmax_dice_loss(pred, gt, self.num_classes, partition_weights=self.weight)
