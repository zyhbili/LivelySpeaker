import os
import argparse
from pkg_resources import require
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from train_inpainting import build_dataloader
from torch.optim import Adam
from SAG_trainertrainer import TrainerMotionClip

import torch
torch.manual_seed(233)
import random
random.seed(233)
np.random.seed(233)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MotionClip")
    parser.add_argument( "--exp", default='test', type=str, help="expname")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num_codebook_vectors', type=int, default=256, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')

    parser.add_argument("-dt", "--data_type", default="shortseq", type= str)
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=401, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=16, help="dataloader worker size")
    parser.add_argument("--n_pre_poses", type=int, default=4, help="n_pre_poses")

    parser.add_argument('--use_cond_motion', default= False, type=bool, help="produce cond motion")
    parser.add_argument('--use_reparam', default= False, type=bool, help="reparam")
    parser.add_argument('--use_style', default= False, type=bool, help="style")
    parser.add_argument('--autoregressive', default= False, type=bool, help="ar")
    parser.add_argument('--audio_model', default= "default", type=str, help="produce cond motion")

    parser.add_argument("-condm",'--condmod', default= 'fft', type=str, help="change speed")


    # parser.add_argument('--batch-size', type=int, default=512, help='Input batch size for training (default: 6)')
    # parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    # parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    # parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    # parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--depth', type=int, default=3, help='transformer depth (default: 3)')

    parser.add_argument('--N_CTX', type=int, default=4, help='')
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default="end", help='Which device the training is on')
    parser.add_argument("--LearnablePrompt", action='store_true',  default= False)
    parser.add_argument("--lam_cos_loss", type = float, default=1.0)
    parser.add_argument("--lam_inter_cos_loss", type = float, default=0.0)
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--use_bert", type = bool, default= False)
    parser.add_argument("--use_bert_feature", type = str, default= "sos")
    parser.add_argument("--mdm_condm", type = str, default= "None")



    args = parser.parse_args()

    output_dir = '/apdcephfs_cq2/share_1290939/yihaozhi/tx_a2g/MotionClip_3'


    train_data_loader = build_dataloader('train', args, shuffle=False)
    test_data_loader = build_dataloader('test', args)

    ckpt_path = None


    ckpt_path = f'/apdcephfs_cq2/share_1290939/yihaozhi/tx_a2g/MotionClip_2/ckpt/motionclipprompt_preseq4_allmetric/model_ep400.pth' 
    trainer = TrainerMotionClip(args, output_dir, train_data_loader, test_data_loader, ckpt_path=ckpt_path)

    if args.test_only:
        assert ckpt_path is not None
        with torch.no_grad():
            avg_loss = trainer.test(400)
    else:
        for epoch in range(args.epochs):
            trainer.train(epoch)
            best_loss = 999999999
            if epoch %100 == 0:
                if test_data_loader is not None:
                    with torch.no_grad():
                        avg_loss = trainer.test(epoch)
            if epoch%100==0:
                # best_loss = avg_loss
                trainer.save(epoch)


    # model = VQGAN(args)
    # a = torch.randn(10,34,27)
    # decoded_motions, codebook_indices, q_loss = model(a)
    # print(decoded_motions.shape)
    # print(codebook_indices.shape)
    # print(q_loss.shape)


