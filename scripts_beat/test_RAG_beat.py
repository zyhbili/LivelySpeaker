import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from utils import other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from mdm_utils.parser_util import generate_args
from mdm_utils import dist_util
# from data_loaders.get_data import get_dataset_loader
from mdm_utils.model_util import load_model_wo_clip,create_model_and_diffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from dataloaders import rot_utils
import random

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

device = torch.device("cuda:0")

class BaseTrainer(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.best_epochs = {
            'fid_val': [np.inf, 0],
            'rec_val': [np.inf, 0],
                           }
        self.loss_meters = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'), 
        } 
        self.alignmenter = metric.alignment(0.3, 2)
        self.srgr_calculator = metric.SRGR(4, 47)
        self.l1_calculator = metric.L1div()
        # model para    
        self.pre_frames = args.pre_frames 
        self.pose_fps = 15
        if args.e_name is not None:
            eval_model_module = __import__(f"model.{args.eval_model}", fromlist=["something"])
            self.eval_model = getattr(eval_model_module, args.e_name)(args)
            other_tools.load_checkpoints(self.eval_model, args.root_path+args.e_path, args.e_name)   
            self.eval_model = self.eval_model.cuda()    
            self.eval_model.eval()
          
    def test(self, epoch, test_loader, sample_fn, model, args, guidance_param = 5, frames_perclip = 34):
        align = 0 
        n_pre_poses = 4
        total_length = 0
        for its, batch_data in enumerate(test_loader):
            print(f'{its}/{len(test_loader)}')
            _, in_audio, in_facial, in_word, vid, emo, sem, aux_info = batch_data
            tar_pose = aux_info['rot6d']
            B, T, C = tar_pose.shape
            tar_pose = tar_pose.cuda().float()
            in_audio = in_audio.cuda().float()
            in_id = vid.cuda().long() 
            in_word = in_word.cuda().long() 
            # in_emo = emo.cuda().long()
            # in_sem = sem.cuda()
            # in_facial = None
            pre_seq = torch.zeros(B, 47,model.nfeats, frames_perclip)
            pre_seq[:,:,:,:n_pre_poses] = tar_pose.reshape(B,T,model.njoints,model.nfeats).permute(0,2,3,1)[:,:,:,:n_pre_poses]
            pre_seq = pre_seq.cuda()

  
            input_audio = in_audio.cuda().float()
        
            cond = {'y': {'mask': torch.ones(B,frames_perclip).cuda().bool()
                    , 'lengths': torch.ones(B,frames_perclip).cuda()*frames_perclip,
                     'vid_indices': in_id, 'audio_input':input_audio,  'text_padded':in_word, 'emo': emo.cuda().long(),
                        'origin_x': pre_seq.clone().cuda()}}
            
            cond['y']['scale'] = torch.ones(B, device=dist_util.dev()) * guidance_param

            decoded_motions = forward_mdm(cond, model).permute(0, 3, 1, 2).reshape(tar_pose.shape)


            decoded_motions = decoded_motions.detach()
            # decoded_motions = tar_pose
            latent_out = self.eval_model(decoded_motions).detach()
            latent_ori = self.eval_model(tar_pose).detach()
            if its == 0:
                latent_out_all = [latent_out.cpu().numpy()]
                latent_ori_all = [latent_ori.cpu().numpy()]
            else:
                latent_out_all.append(latent_out.cpu().numpy())
                latent_ori_all.append(latent_ori.cpu().numpy())

            target_euler = rot_utils.matrix_to_euler_angles(rot_utils.rotation_6d_to_matrix(tar_pose.reshape(-1, 34, 47,6)), "XYZ").flatten(2)/(np.pi)*180
            pred_euler = rot_utils.matrix_to_euler_angles(rot_utils.rotation_6d_to_matrix(decoded_motions.reshape(-1, 34, 47,6)), "XYZ").flatten(2)/(np.pi)*180

            total_length += target_euler.shape[0]

            t_start = 0
            t_end = 500

            # import pdb;pdb.set_trace()
            decoded_motions = decoded_motions.detach().cpu().numpy()
            for i in range(B):
                onset_raw, onset_bt, onset_bt_rms = self.alignmenter.load_audio(in_audio[i].cpu().numpy().reshape(-1), t_start, t_end, True)
                beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.alignmenter.load_pose(pred_euler[i].detach().cpu().numpy(), t_start, t_end, 15, True)
                align += self.alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, self.pose_fps)
        align_avg = align/total_length
        print(f"align score: {align_avg}")
        print('expname',args.expname, epoch)
        print('guidance_param',guidance_param)
        diversity = data_tools.FIDCalculator.get_diversity(latent_out_all)
        latent_out_all = np.concatenate(latent_out_all, axis=0)
        latent_ori_all = np.concatenate(latent_ori_all, axis=0)
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        print(f"fid score: {fid}")
        print(f'diversity{diversity}')
        return fid, align_avg, diversity

def forward_mdm(cond, model, frames_perclip = 34):
    bs = cond['y']['origin_x'].shape[0]
    sample = sample_fn(
        model,
        (bs, model.njoints, model.nfeats, frames_perclip),
        clip_denoised=False,
        model_kwargs=cond,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    return sample


def get_mdm():
    args = generate_args()
    use_ddim = True
    if use_ddim:
        timestep_respacing = 'ddim100'
    else:
        timestep_respacing = ''

    train_data = __import__(f"dataloaders.beat", fromlist=["something"]).CustomDataset(args, "train")  
    lang_model = train_data.lang_model
    args.lang_model= lang_model
    model, diffusion = create_model_and_diffusion(args, timestep_respacing)
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    if use_ddim:
        sample_fn = diffusion.ddim_sample_loop
    else:
        sample_fn = diffusion.p_sample_loop

    return args,model, sample_fn

            
if __name__ == "__main__":
    fixseed(233)
    model_type = 'mdm'
    args, model, sample_fn = get_mdm()
    ckpt_path = args.model_path
    epoch = int(ckpt_path.split('/')[-1].split('.')[0][5:])//110
    args.expname = ckpt_path.split('/')[-2]
    args.model_type = model_type
    trainer = BaseTrainer(args)
    model = model.cuda()
    model.eval()

    test_data = __import__(f"dataloaders.beat", fromlist=["something"]).CustomDataset(args, "finaltest")  
    print(len(test_data))
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=256,  
        shuffle=True,  
        num_workers=16,
        drop_last=False,
    )
    results_list = []
    with torch.no_grad():
        t = trainer.test(epoch, test_loader, sample_fn, model, args, guidance_param= 1)
        results_list.append(t)
        t = trainer.test(epoch, test_loader, sample_fn, model, args, guidance_param= 1.5)
        results_list.append(t)
    for item in results_list:
        print(item)


