import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.rank = args.rank
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, text_features, image_features, logit_scale=2.659):
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss


class DualCLIPLoss(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.rank = args.rank
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, text_features, image_features, text_features_proj, image_features_proj, logit_scale=2.659):

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        image_features_proj = F.normalize(image_features_proj, dim=-1)
        text_features_proj = F.normalize(text_features_proj, dim=-1)

        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        logits_per_image_proj = logit_scale * image_features_proj @ text_features_proj.T
        logits_per_text_proj = logit_scale * text_features_proj @ image_features_proj.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]

        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        loss_proj = (
            F.cross_entropy(logits_per_image_proj, labels) +
            F.cross_entropy(logits_per_text_proj, labels)
            ) / 2
        total_loss = loss + loss_proj
        return total_loss

