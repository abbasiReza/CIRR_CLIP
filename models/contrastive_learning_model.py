import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ContrastiveLearningModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, fusion_model, temperature=0.07):
        super(ContrastiveLearningModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.image_encoder_target = copy.deepcopy(image_encoder)
        self.fusion_model = fusion_model
        self.temperature = temperature

        for name, param in self.image_encoder.named_parameters():
            if name not in freeze_image:
                param.requires_grad = False

        for name, param in self.image_encoder_target.named_parameters():
            param.requires_grad = False

        for name, param in self.text_encoder.named_parameters():
            if name not in freeze_text:
                param.requires_grad = False

    def forward(self, text, ref_images, tar_images):
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(ref_images)
        fused_features = self.fusion_model(text_features, image_features)
        image_only_features = self.image_encoder_target(tar_images)
        return fused_features, image_only_features

    def contrastive_loss_clip(self, text_embeddings, image_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
