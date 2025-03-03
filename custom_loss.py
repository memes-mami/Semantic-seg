import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, preds, targets):
        # Ensure inputs are tensors
        if isinstance(preds, list):
            preds = torch.stack(preds)  # Convert list to tensor
        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.long, device=preds.device)  # Ensure same device

        

        # Convert labels {0, 4, 8, ..., 24} → {0, 1, 2, ..., 6}
        new_targets = (targets // 4).long()

        # Ensure target shape matches expected format
        #if new_targets.dim() == 4 and new_targets.shape[1] == 1:
        #    new_targets = new_targets.squeeze(1)  # Convert (batch, 1, H, W) → (batch, H, W)

        print(f"Targets shape after conversion: {new_targets.shape}")  # Expected: (batch, H, W)

        return F.cross_entropy(preds, new_targets, ignore_index=-1)
