import torch
import lmdb
import pyarrow as pa
import numpy as np
import matplotlib.pyplot as plt
import cv2, math, os
import ffmpeg
import shutil, random
import soundfile as sf
from model.motionclip import get_SAG
import clip as myclip
import argparse, librosa
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from dataloaders import rot_utils
from utils import other_tools, metric
from mdm_utils.model_util import load_model_wo_clip, create_model_and_diffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from mdm_utils import dist_util


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


device = torch.device("cuda:0")
cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def load_model(ckpt_path, cfg):
    model, clip_model = get_SAG(cfg)
    # import pdb;pdb.set_trace()
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt, strict = True)
    model.to(device)
    model.eval()
    print("load success")
    return model, clip_model


frames = 34
skeleton_resampling_fps = 15
batchify = False


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

    def infer_from_testloader(self, test_loader, motionclip_model = None, frames_perclip = 34, mdm_model = None, sample_fn = None, guidance_param=5, skipsteps = 950):
        align = 0 
        total_length = 0
        for batch_iter, batch in enumerate(test_loader):
            print(f"{batch_iter}/{len(test_loader)}")
            tar_pose, in_audio, in_facial, in_word, vid, emo, sem, aux_info = batch
        
            tar_pose = aux_info['rot6d']
            B, T, C = tar_pose.shape
            tar_pose = tar_pose.cuda().float()
            in_audio = in_audio.cuda().float()
            in_id = vid.cuda().long() 
            in_word = in_word.cuda().long() 
            in_sem = sem.cuda()
            in_facial = None

            n_pre_poses = 4

            batch = {
                "x": tar_pose.reshape(tar_pose.shape[0],tar_pose.shape[1],-1,6).permute(0,2,3,1),
                'mask':torch.ones(tar_pose.shape[0],tar_pose.shape[1]).to(device).bool(),
                "clip_text":aux_info["sentence"],
            }

            texts = myclip.tokenize(batch['clip_text']).cuda()
            features = clip_model.encode_text(texts)
            batch['z'] = features
            decoded_motions = motionclip_model.decoder(batch)['output']

            pre_seq = torch.zeros(B, 47,6, frames_perclip)
            pre_seq[:,:,:,:n_pre_poses] = tar_pose.reshape(B,T,47,6).permute(0,2,3,1)[:,:,:,:n_pre_poses]
            pre_seq = pre_seq.cuda()

            input_audio = in_audio.cuda().float()
        
            cond = {'y': {'mask': torch.ones(B,frames_perclip).cuda().bool()
                    , 'lengths': torch.ones(B,frames_perclip).cuda()*frames_perclip,
                     'vid_indices': in_id, 'audio_input':input_audio,  'text_padded':in_word, 'emo': emo.cuda().long(),
                        'origin_x': pre_seq.clone().cuda()}}
            cond['y']['scale'] = torch.ones(B, device=dist_util.dev()) * guidance_param


            sample = sample_fn(
                mdm_model,
                (B, 47, 6, frames_perclip),
                clip_denoised=False,
                model_kwargs=cond,
                skip_timesteps=skipsteps,  # 0 is the default value - i.e. don't skip any step
                init_image=decoded_motions,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            decoded_motions = sample.permute(0, 3, 1, 2).reshape(tar_pose.shape)

            decoded_motions = decoded_motions.detach()
            latent_out = self.eval_model(decoded_motions).detach()
            latent_ori = self.eval_model(tar_pose).detach()
            if batch_iter == 0:
                latent_out_all = [latent_out.cpu().numpy()]
                latent_ori_all = [latent_ori.cpu().numpy()]
            else:
                latent_out_all.append(latent_out.cpu().numpy())
                latent_ori_all.append(latent_ori.cpu().numpy())


            target_euler = rot_utils.matrix_to_euler_angles(rot_utils.rotation_6d_to_matrix(tar_pose.reshape(-1, 34, 47,6)), "XYZ").flatten(2)/(np.pi)*180
            pred_euler = rot_utils.matrix_to_euler_angles(rot_utils.rotation_6d_to_matrix(decoded_motions.reshape(-1, 34, 47,6)), "XYZ").flatten(2)/(np.pi)*180

            np_cat_results = pred_euler.reshape(-1, pred_euler.shape[-1]).detach().cpu().numpy()
            np_cat_targets = target_euler.reshape(-1, target_euler.shape[-1]).cpu().numpy()
            np_cat_sem = in_sem.flatten().cpu().numpy()


            # target_euler = 
            _ = self.srgr_calculator.run(np_cat_results, np_cat_targets, np_cat_sem)

            total_length += target_euler.shape[0]

            t_start = 0
            t_end = 500

            decoded_motions = decoded_motions.detach().cpu().numpy()
            for i in range(B):
                onset_raw, onset_bt, onset_bt_rms = self.alignmenter.load_audio(in_audio[i].cpu().numpy().reshape(-1), t_start, t_end, True)
                beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.alignmenter.load_pose(pred_euler[i].detach().cpu().numpy(), t_start, t_end, 15, True)
                align += self.alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, self.pose_fps)
        align_avg = align/total_length
        print(f"align score: {align_avg}")
        print('guidance_param',guidance_param)
        srgr = self.srgr_calculator.avg()
        print(f"srgr score: {srgr}")
        diversity = data_tools.FIDCalculator.get_diversity(latent_out_all)
        latent_out_all = np.concatenate(latent_out_all, axis=0)
        latent_ori_all = np.concatenate(latent_ori_all, axis=0)
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        print(f"fid score: {fid}")
        print(f'diversity{diversity}') 
        return  fid, align_avg, diversity, srgr
            
        





if __name__ == "__main__":
    fixseed(233)
    from mdm_utils.parser_util import generate_args
    args = generate_args(use_motionclip=True)
    motionclip_ckpt_path = args.sag_path
    args.use_style_cache = args.use_style
    args.use_style = False
    motionclip_model, clip_model = load_model(motionclip_ckpt_path, args)
    args.use_style = args.use_style_cache


    test_data = __import__(f"dataloaders.beat", fromlist=["something"]).CustomDataset(args, "finaltest")  

    lang_model = test_data.lang_model
    args.lang_model= lang_model
    print(len(test_data))
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=256,  
        shuffle=True,  
        num_workers=11,
        drop_last=False,
    )
    exp = motionclip_ckpt_path.split('/')[-2]


    use_ddim = True
    if use_ddim:
        timestep_respacing = 'ddim100'
    else:
        timestep_respacing = ''
    mdm_model, diffusion = create_model_and_diffusion(args, timestep_respacing)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(mdm_model, state_dict)

    mdm_model = ClassifierFreeSampleModel(mdm_model)   # wrapping model with the classifier-free sampler
    mdm_model.to(dist_util.dev())
    mdm_model.eval()  # disable random masking

    if use_ddim:
        sample_fn = diffusion.ddim_sample_loop
    else:
        sample_fn = diffusion.p_sample_loop

    trainer = BaseTrainer(args)
    skipsteps = 80
    results_list = []
    t = trainer.infer_from_testloader(test_loader = test_loader, motionclip_model= motionclip_model, mdm_model=mdm_model, sample_fn=sample_fn,guidance_param=1, skipsteps = skipsteps)
    results_list.append(t)
    t=trainer.infer_from_testloader(test_loader = test_loader, motionclip_model= motionclip_model, mdm_model=mdm_model, sample_fn=sample_fn,guidance_param=1.5, skipsteps = skipsteps)
    results_list.append(t)


    for item in results_list:
        print(item)
