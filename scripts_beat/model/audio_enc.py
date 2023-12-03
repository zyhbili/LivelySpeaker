import torch
import torch.nn.functional as F
from torch import nn

    
class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=5, padding=1600),
            nn.InstanceNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 128, 15, stride=6),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(128, 256, 15, stride=6),
        )

    def forward(self, wav_data, num_frames):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)
    

if __name__ == "__main__":
    model = WavEncoder()
    a = torch.randn(64,128,70)
    out = model(a, 34)
    print('out:',out.shape)
