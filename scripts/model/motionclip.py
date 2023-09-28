import numpy as np
import torch
import torch.nn as nn

import clip
from .motionclip_module import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from .motionclip_loss import compute_rcxyz_loss, compute_velxyz_loss
loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

class MOTIONCLIP(nn.Module):
    def __init__(self, encoder, decoder, promptLearner, cfg):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.promptLearner = promptLearner
        self.cfg = cfg


    def compute_loss(self, batch, clip_model):

        # compute all losses other than clip
        losses = {}
        xyz_loss = compute_rcxyz_loss(batch)
        vel_loss = compute_velxyz_loss(batch)
        

        # compute clip losses
        cosine_loss, cos_sim, _ = self.compute_clip_losses(batch, clip_model)

        if 'z_logvar' in batch.keys():
            z_mu = batch['z_mu']
            z_logvar = batch['z_logvar']
            kld_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            losses["kld"] = kld_loss
        # mix and add clip losses
        mixed_loss_with_clip = xyz_loss + vel_loss + self.cfg.lam_cos_loss * cosine_loss + 0.1* losses.get('kld', 0.)   # this is the ultimate loss to optimize, combining ALL losses
        losses["xyz_loss"] = xyz_loss
        losses["vel_loss"] = vel_loss
        losses["clip_loss"] = cosine_loss
        losses['cos_sim'] = cos_sim
        losses["sum"] = mixed_loss_with_clip
        

        return mixed_loss_with_clip, losses

    def compute_clip_losses(self, batch, clip_model):
        with torch.no_grad():
            texts = texts = clip.tokenize(batch['clip_text']).cuda()
            features = clip_model.encode_text(texts)
        batch['feature_scale'] = features.norm(dim=-1, keepdim = True).mean()
        batch['z_scale'] = batch["z"].norm(dim=-1).mean()

        print(batch['feature_scale'],)
        features_norm = features / features.norm(dim=-1, keepdim=True)
        seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)

        cos = cosine_sim(features_norm, seq_motion_features_norm)
        cosine_loss = (1 - cos).mean()

        return cosine_loss, cos.mean(), 0

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def forward(self, batch):
        batch.update(self.encoder(batch))
        batch["z"] = batch["mu"]
        # decode
        batch.update(self.decoder(batch))

        # if we want to output xyz
        batch["output_xyz"] = batch["output"]
        return batch


def get_SAG(cfg):
    textEncoder = get_clip()
    promptLearner = None
    dim = 512
    encoder = Encoder_TRANSFORMER(latent_dim=dim)
    decoder = Decoder_TRANSFORMER(latent_dim=dim, n_pre_poses = cfg.n_pre_poses, use_style = cfg.use_style)
    return MOTIONCLIP(encoder, decoder, promptLearner, cfg), textEncoder



def get_clip():
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda:0", jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    return clip_model.float()

