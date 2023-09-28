import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from .audio_enc import WavEncoder
from .mlp_module import build_mlps


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class RAG(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', clip_dim=512,
                 arch='trans_enc', mlpact = 'silu',**kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.cond_mode = kargs.get('cond_mode', 'no_cond')

        self.latent_dim = latent_dim


        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats


        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.mlpact = mlpact
        self.backbone = build_mlps(seq_len=35,num_layers=num_layers, hidden_dim=latent_dim, act=self.mlpact)
        self.input_process = InputProcess(self.data_rep, self.input_feats * 2+1+self.gru_emb_dim, self.latent_dim)


        audio_feat_dim = 256

        self.input_mapping = nn.Linear(self.input_feats * 2+1 + audio_feat_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        n_words = 1400
        self.speaker_embedding = nn.Embedding(n_words, 256)
        nn.init.constant_(self.speaker_embedding.weight, 1e-6)
        self.speaker_mu = nn.Linear(256, self.latent_dim)
        self.speaker_logvar = nn.Linear(256, self.latent_dim)
        self.n_pre_seq = 4

        self.audio_encoder = WavEncoder()
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            if cond.ndim>2:
                ori_shape = cond.shape
                cond = cond.flatten(1)
                mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
                ret = cond * (1. - mask)
                ret = ret.reshape(ori_shape)
            else:
                mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
                ret = cond * (1. - mask)
            return ret
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        force_mask = y.get('uncond', False)

        af = self.audio_encoder(y['audio_input'],num_frames = nframes)
        audio_emb = self.mask_cond(af, force_mask=force_mask) # B, T, D
        origin_x = y['origin_x']

        origin_x[...,self.n_pre_seq:]  = 0
        x = self.input_process(torch.cat([x,origin_x], dim = 1), n_pre_seq = self.n_pre_seq)
        x = torch.cat([x, audio_emb], dim = -1)

        x = self.input_mapping(x)

        n_pre_emb = 1
        z_context = self.speaker_embedding(y['vid_indices'])[:,None]
        z_mu = self.speaker_mu(z_context)
        z_logvar = self.speaker_logvar(z_context)
        style_emb = reparameterize(z_mu, z_logvar)

        xseq = torch.cat((style_emb, x),dim =1)

        output = self.backbone(xseq, timesteps)[:,n_pre_emb:].permute(1,0,2)
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]

        ret = {
            'output': output,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
        }

        return ret
    
  
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim


    def forward(self, x, n_pre_seq=-999):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        tmp = x.new_zeros((x.shape[0], x.shape[1], x.shape[2] + 1))
        tmp[:,:, :njoints*nfeats] = x

        tmp[:n_pre_seq, :, -1] = 1  # indicating bit for constraints
        return tmp.permute(1,0,2)


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
     
    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
       
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


