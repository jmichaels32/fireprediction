import torch
import numpy as np
from tqdm import tqdm
from .utils import generate_dataloader, graph_losses

# ---------------------- Loss Functions ----------------------
def dice_loss(gold_mask, pred_mask, smooth=1e-6):
    # Convert pred_mask to binary
    pred_mask_binary = (pred_mask > 0.5).int()

    mask = gold_mask != -1

    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]

    intersection = (gold_mask_masked * pred_mask_masked).sum()
    union = gold_mask_masked.sum() + pred_mask_masked.sum()

    return 1 - ((2.0 * intersection + smooth) / union)

def WBCE(gold_mask, pred_mask, w0=1, w1=100):
    mask = gold_mask != -1

    gold_mask_masked = gold_mask[mask].float()
    pred_mask_masked = pred_mask[mask].float()

    weights = gold_mask_masked * w1 + (1 - gold_mask_masked) * w0
    return torch.nn.functional.binary_cross_entropy_with_logits(pred_mask_masked, gold_mask_masked, weights, reduction='mean')

# Pred mask is the predicted fire segmentation mask with probability scores that it is fire (class 1)
# Gold mask is the ground truth fire segmentation mask with values -1 (no data), 0 (no fire), 1 (fire) 
def loss(gold_mask, pred_mask):
    return WBCE(gold_mask, pred_mask) + 2 * dice_loss(gold_mask, pred_mask)

# ---------------------- Evaluation Functions ----------------------
def mean_iou(gold_mask, pred_mask):
    pred_mask_binary = (pred_mask > 0.5).int()

    mask = gold_mask != -1

    # Calculate intersection and union of class 1 pixels
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]

    intersection = (gold_mask_masked * pred_mask_masked).sum()
    union = gold_mask_masked.sum() + pred_mask_masked.sum() - intersection

    # Calculate intersection over union of class 0 pixels
    gold_mask_masked = 1 - gold_mask_masked
    pred_mask_masked = 1 - pred_mask_masked

    intersection_0 = (gold_mask_masked * pred_mask_masked).sum()
    union_0 = gold_mask_masked.sum() + pred_mask_masked.sum() - intersection_0

    return ((intersection / union) + (intersection_0 / union_0)) / 2

def accuracy(gold_mask, pred_mask):
    pred_mask_binary = (pred_mask > 0.5).int()

    mask = gold_mask != -1

    # Calculate accuracy of class 1 pixels
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]

    accuracy_1 = (gold_mask_masked == pred_mask_masked).sum() / len(gold_mask_masked)

    # Calculate accuracy of class 0 pixels
    gold_mask_masked = 1 - gold_mask_masked
    pred_mask_masked = 1 - pred_mask_masked

    accuracy_0 = (gold_mask_masked == pred_mask_masked).sum() / len(gold_mask_masked)
    
    return (accuracy_1 + accuracy_0) / 2

def distance(gold_mask, pred_mask):
    mask = gold_mask != -1

    gold_mask_masked = gold_mask[mask].float()
    pred_mask_masked = pred_mask[mask].float()

    return torch.linalg.norm(gold_mask_masked - pred_mask_masked)

# ---------------------- Training Functions ----------------------
def evaluate_model(model, val_data):
    model.eval()
    iou = []
    accs = []
    dists = []
    losses = []
    for images, masks in val_data:
        predictions = model(images)
        losses.append(loss(masks, predictions).item())
        iou.append(mean_iou(masks, predictions))
        accs.append(accuracy(masks, predictions))
        dists.append(distance(masks, predictions))
    return torch.tensor(iou).mean(), torch.tensor(accs).mean(), torch.tensor(dists).mean(), torch.tensor(losses).mean()

def train(model, batch_size, epochs):
    train_data = generate_dataloader('train', batch_size)
    val_data = generate_dataloader('eval', batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epoch_losses = []
    validation_losses = []
    best_iou = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        model.train()

        losses = []
        losses_dice = []
        losses_wbce = []
        progress = tqdm(train_data)

        for images, masks in progress:
            optimizer.zero_grad()

            predictions = model(images)

            loss_value = loss(masks, predictions)
            loss_dice = dice_loss(masks, predictions)
            loss_wbce = WBCE(masks, predictions)

            losses.append(loss_value.item())
            losses_dice.append(loss_dice.item())
            losses_wbce.append(loss_wbce.item())

            progress.set_postfix({'batch_loss': loss_value.item(), 
                                  'dice_loss': loss_dice.item(),
                                  'wbce_loss': loss_wbce.item()})
            
            loss_value.backward()
            optimizer.step()
        
        print("Evaluation...")
        iou_val, acc_val, dist_val, validation_loss = evaluate_model(model, val_data)
        print("Validation set metrics:")
        print(f"Mean IoU: {iou_val}\nMean accuracy: {acc_val}\nMean dist: {dist_val}")

        if iou_val > best_iou:
            best_iou = iou_val
            torch.save(model.state_dict(), "best_baseline.pth")
        
        epoch_losses.append(np.mean(losses))
        print(f'Epoch: {epoch}, Train loss: {epoch_losses[-1]}, Validation loss: {validation_loss}')
        validation_losses.append(validation_loss)

    print(f"Best model IoU: {best_iou}")
    graph_losses(losses, losses_dice, losses_wbce, validation_losses)
    return epoch_losses, validation_losses

def test(model, batch_size):
    model.eval()
    test_data = generate_dataloader('test', batch_size)
    iou, acc, dist, loss = evaluate_model(model, test_data)
    print(f"Test set metrics:\nMean IoU: {iou}\nMean accuracy: {acc}\nMean dist: {dist}\nMean loss: {loss}")

if __name__ == "__main__":
    pred_mask = np.random.randint(0, 2, (32, 64, 64))
    gold_mask = np.random.choice([-1, 0, 1], (32, 64, 64))

    dice = dice_loss(torch.tensor(gold_mask), torch.tensor(pred_mask))
    wbce = WBCE(torch.tensor(gold_mask), torch.tensor(pred_mask))
    l = loss(torch.tensor(gold_mask), torch.tensor(pred_mask))

    meaniou = mean_iou(torch.tensor(gold_mask), torch.tensor(pred_mask))
    acc = accuracy(torch.tensor(gold_mask), torch.tensor(pred_mask))
    dist = distance(torch.tensor(gold_mask), torch.tensor(pred_mask))

    print(dice)
    print(wbce)
    print(l)
    print(meaniou)
    print(acc)
    print(dist)