import torch
import numpy as np

# ---------------------- Loss Functions ----------------------
def dice_loss(gold_mask, pred_mask, smooth=1e-6):
    # Convert pred_mask to binary
    pred_mask_binary = pred_mask > 0.5

    mask = gold_mask != -1

    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]

    intersection = (gold_mask_masked * pred_mask_masked).sum()
    union = gold_mask_masked.sum() + pred_mask_masked.sum()

    return 1 - ((2.0 * intersection + smooth) / union)

def WBCE(gold_mask, pred_mask, w0=0.01, w1=1):
    mask = gold_mask != -1

    gold_mask_masked = gold_mask[mask].float()
    pred_mask_masked = pred_mask[mask].float()

    weights = gold_mask_masked * w1 + (1 - gold_mask_masked) * w0
    return torch.nn.functional.binary_cross_entropy_with_logits(pred_mask_masked, gold_mask_masked, weights, reduction='mean')

# Pred mask is the predicted fire segmentation mask with probability scores that it is fire (class 1)
# Gold mask is the ground truth fire segmentation mask with values -1 (no data), 0 (no fire), 1 (fire) 
def loss(gold_mask, pred_mask):
    return (WBCE(gold_mask, pred_mask) + dice_loss(gold_mask, pred_mask)) / 2

# ---------------------- Evaluation Functions ----------------------

if __name__ == "__main__":
    pred_mask = np.random.randint(0, 2, (32, 64, 64))
    gold_mask = np.random.choice([-1, 0, 1], (32, 64, 64))

    val_1 = dice_loss(torch.tensor(gold_mask), torch.tensor(pred_mask))
    val_2 = WBCE(torch.tensor(gold_mask), torch.tensor(pred_mask))
    loss = loss(torch.tensor(gold_mask), torch.tensor(pred_mask))