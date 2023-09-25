import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.rank = args.rank
        # cache state
        self.prev_batch_size = 0
        self.labels = {}

    def forward(self, text_features, image_features, logit_scale: float = 2.659):
        device = image_features.device

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # calculated ground-truth and cache if enabled
        batch_size = len(logits_per_image)
        if self.prev_batch_size != batch_size or device not in self.labels:
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss
