import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from .motionclip import get_motionclip
from torch.optim import Adam
# from .draw_util import my_mkdirs, vis_results, img2vid
from torch.utils.tensorboard import SummaryWriter
import shutil, tqdm, clip
import matplotlib, math, librosa
import torch.nn.functional as F

# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from utils import other_tools
from dataloaders import data_tools
from utils import keypoints_utils as kp_utils



def my_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class TrainerMotionClip:
    def __init__(self, args, output_dir, train_data_loader, test_data_loader, ckpt_path = None):
        self.device = torch.device("cuda:0")
        self.model, self.clip_model = get_motionclip(args)

        if args.use_bert:
            self.tokenizer, self.clip_model = get_bert()
            self.clip_model.my_tokenizer = self.tokenizer

        betas=(args.adam_beta1, args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=args.lr, betas=betas, weight_decay=args.adam_weight_decay)

        if ckpt_path:
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt)
            print("load success")
        self.model.to(self.device)


        print("Creating Dataloader")
        self.train_data = train_data_loader
        self.test_data = test_data_loader

        self.log_freq = 20
        self.global_step = 0

        exp = args.exp
        self.log_dir = f"{output_dir}/log/{exp}"
        self.ckpt_dir = f"{output_dir}/ckpt/{exp}"
        self.vis_dir = f"{output_dir}/vis/{exp}"
        my_mkdirs(self.ckpt_dir)
        my_mkdirs(self.vis_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir, max_queue=1)
        self.exp = exp
        # self.embed_space_evaluator = EmbeddingSpaceEvaluator()
        self.args = args
        self.cfg = args
        # self.speaker_model = self.train_data.dataset.speaker_model
        eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
        self.eval_model = getattr(eval_model_module, args.e_name)(args)
        # self.eval_model = torch.nn.DataParallel(self.eval_model, args.gpus).cuda()
        other_tools.load_checkpoints(self.eval_model, args.root_path+args.e_path, args.e_name)
        self.eval_model = self.eval_model.cuda()
        self.eval_model.eval()
    def train(self, epoch):
        return self.iteration(epoch, self.train_data, str_code = 'train')

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, str_code = 'test')

    def iteration(self, epoch, data_loader, str_code="train"):
        if str_code=="train":
            self.model.train()
        else:
            self.model.eval()  

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_cos_sim = 0.0
        avg_cos_loss = 0.0
        avg_mse = 0.0
        avg_vel_mse = 0.0
        num_samples = 0
        a2m_score_sum = 0
        m2a_score_sum = 0
        num_audio_beats = 0
        num_motion_beats = 0
        
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # print(f'{i}/{len(data_iter)}')
            tar_pose, in_audio, in_facial, in_word, vid, emo, sem, aux_info = data
            tar_pose = aux_info['rot6d'].to(self.device)
            rep_dim = 6
            if self.args.njoints == 9:
                sub_pose = kp_utils.upper141_To_body27(tar_pose, rep_dim=rep_dim)
            else:
                sub_pose = tar_pose
            # vec_seq = vec_seq.to(self.device)


            sub_pose = sub_pose.to(self.device)

            batch = {
                "x": sub_pose.reshape(sub_pose.shape[0],sub_pose.shape[1],-1,rep_dim).permute(0,2,3,1),
                'mask':torch.ones(sub_pose.shape[0],sub_pose.shape[1]).to(self.device).bool(),
                "clip_text":aux_info["sentence"],
            }

           
            if str_code == "test":
                batch['vid_indices'] = vid.to(self.device).long()
                batch['teacher_forcing'] = False
                batch['use_reparam'] = False
                if self.args.use_bert:
                    with torch.no_grad():
                        inputs = self.clip_model.my_tokenizer(batch['clip_text'], return_tensors = 'pt', padding = True).to('cuda')
                        outputs = self.clip_model(**inputs)
                        features = outputs.last_hidden_state
                        if self.cfg.use_bert_feature == "sos":
                            features = features[:,0,:]
                        if self.cfg.use_bert_feature == "eos":
                            seq_len = inputs['attention_mask'].sum(-1)
                            features = features[torch.arange(features.shape[0]),seq_len-1,:]
                else: #clip
                    texts = texts = clip.tokenize(batch['clip_text']).cuda()
                    features = self.clip_model.encode_text(texts)
                batch['z'] = features
                ret = self.model.decoder(batch)
                ret["output_xyz"] = ret["output"]
            else:
                vid_indices = vid.to(self.device).long()
                batch['vid_indices'] = vid_indices
                batch['use_reparam'] = self.args.use_reparam
                batch['teacher_forcing'] = True
                ret = self.model(batch)

            decoded_motions = ret['output_xyz'].permute(0,3,1,2).reshape(sub_pose.shape[0],sub_pose.shape[1],-1)
            if self.args.njoints == 9:
                decoded_motions = kp_utils.body27_To_upper141(decoded_motions)
            tar_pose_wohand = kp_utils.zeros_hand_upper141(tar_pose)
            if str_code == "test":
                latent_out = self.eval_model(decoded_motions)
                latent_ori = self.eval_model(tar_pose)
                latent_ori_wohand = self.eval_model(tar_pose_wohand)
                if i == 0:
                    latent_out_all = latent_out.cpu().numpy()
                    latent_ori_all = latent_ori.cpu().numpy()
                    latent_ori_wohand_all = latent_ori_wohand.cpu().numpy()
                else:
                    latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()], axis=0)
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()], axis=0)
                    latent_ori_wohand_all = np.concatenate([latent_ori_wohand_all, latent_ori_wohand.cpu().numpy()], axis=0)

         
         
            loss, loss_dict = self.model.compute_loss(ret, self.clip_model)
            loss_dict['norm_scale'] = ret['feature_scale']/ret['z_scale']
            
        
            # 3. backward and optimization only in train
            # loss = 0
            # for key in loss_dict.keys():
            #     loss += loss_dict[key]
            # import pdb;pdb.set_trace()
            # self.model.promptLearner.ctx
            # print("init", self.optim.param_groups[0]['params'][-1])

            if str_code=="train":
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                # self.scheduler.step()
            # import pdb;pdb.set_trace()
            avg_loss += loss.item()
            avg_cos_sim += loss_dict["cos_sim"].item() * tar_pose.shape[0]
            avg_cos_loss += loss_dict["clip_loss"].item() * tar_pose.shape[0]
            avg_mse += loss_dict["xyz_loss"].item() * tar_pose.shape[0]
            avg_vel_mse += loss_dict["vel_loss"].item() * tar_pose.shape[0]

            num_samples += tar_pose.shape[0]
            self.global_step +=1
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
                "cos loss": loss_dict["clip_loss"].item(),
                "cos sim": loss_dict["cos_sim"].item()/tar_pose.shape[0]
                            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                if str_code=="train":
                    for key in loss_dict:
                        self.writer.add_scalar(f"Train/{key}", loss_dict[key].detach().cpu().numpy(), self.global_step/self.log_freq)
                    # self.writer.add_scalar("LR", self.scheduler.get_lr()[0], self.global_step/self.log_freq)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / num_samples)
        if str_code == "test":
            print(avg_loss,num_samples, avg_loss/num_samples)
            self.writer.add_scalar(f"Test/SUM", avg_loss/num_samples, epoch)
            self.writer.add_scalar(f"Test/cos_sim", avg_cos_sim/num_samples, epoch)
            self.writer.add_scalar(f"Test/cos_loss", avg_cos_loss/num_samples, epoch)
            self.writer.add_scalar(f"Test/xyz_mse", avg_mse/num_samples, epoch)
            self.writer.add_scalar(f"Test/vel_mse", avg_vel_mse/num_samples, epoch)
        
            fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
            fid_wohand = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_wohand_all)

            print("fid", fid)
            self.writer.add_scalar(f"Test/FGD", fid, epoch)
            self.writer.add_scalar(f"Test/FGD_wohand", fid_wohand, epoch)


        return avg_loss



    def save(self, epoch):
        output_path = self.ckpt_dir + "/model_ep%d.pth" % epoch
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
