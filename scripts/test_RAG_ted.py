import torch
from model.ted_evaluator import EmbeddingSpaceEvaluator
import pyarrow as pa
import numpy as np
import math
import librosa
import shutil
import soundfile as sf
from train_utils.ted_loader import build_dataloader
import torch.nn.functional as F

from mdm_utils.fixseed import fixseed
import os, random
import numpy as np
import torch
from mdm_utils.parser_util import generate_args
from mdm_utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm_utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
import pickle as pkl
device = torch.device("cuda:0")
mean_dir_vec = np.array([ 0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916])

angle_pair = [
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8)
]
change_angle = [0.0034540758933871984, 0.007043459918349981, 0.003493624273687601, 0.007205077446997166]

thres=0.03
sigma=0.1

embed_space_evaluator = EmbeddingSpaceEvaluator()


def infer_from_testloader(sample_fn, model, args, test_dataloader, guidance_param = 2.5, speaker_model = None, eta = 0):
    beat_align_score_sum = 0
    num_beats = 0
    motion_beats_sum= 0

    for i, data in enumerate(test_dataloader):
        # print(f"{i}/{len(test_dataloader)}")
        beat, word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info = data
        B, T = vec_seq.shape[0],vec_seq.shape[1]

        vec_seq = vec_seq.to(device)
        spectrogram = spectrogram.to(device).float()

        inpainting_mask = torch.zeros(B,9,3,34).to(device).bool()
        inpainting_mask[...,:4] = 1
        n_frames = vec_seq.shape[1]
        vids = aux_info['vid']
        # vid_indices = torch.randint(0,1370, (B,))
        vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(B)]
        vid_indices = torch.LongTensor(vid_indices).to(device)
        # import pdb;pdb.set_trace()

        audio_input = audio.to(device).float()

        
        cond = {'y': {'mask': torch.ones(B,34).to(device).bool(), 'beat':beat.to(device)
                    , 'lengths': torch.ones(B,34).to(device)*34, 'text': aux_info['sentence'],
                    'audio_input':audio_input, 'text_padded':text_padded.to(device), 'vid_indices':vid_indices, 
                        'origin_x': vec_seq.clone().reshape(B, T ,9,3).permute(0,2,3,1).to(device)}}

         

        cond['y']['scale'] = torch.ones(B, device=dist_util.dev()) * guidance_param
        sample = sample_fn(
            model,
            (B, model.njoints, model.nfeats, 34),
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False
        )        
      
        aligned_motions = sample.permute(0,3,1,2).reshape(vec_seq.shape[0],vec_seq.shape[1],-1)

        embed_space_evaluator.push_samples(aligned_motions, vec_seq)
        batch_size = aligned_motions.shape[0]
        beat_vec = aligned_motions + torch.Tensor(mean_dir_vec).squeeze().cuda()
        beat_vec = beat_vec.reshape(beat_vec.shape[0], beat_vec.shape[1], -1, 3)
        beat_vec = F.normalize(beat_vec, dim = -1)
        all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)

        for idx, pair in enumerate(angle_pair):
            vec1 = all_vec[:, pair[0]]
            vec2 = all_vec[:, pair[1]]
            inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
            inner_product = torch.clamp(inner_product, -1, 1, out=None)
            angle = torch.acos(inner_product) / math.pi
            angle_time = angle.reshape(batch_size, -1)
            if idx == 0:
                angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
            else:
                angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
        angle_diff = torch.cat((torch.zeros(batch_size, 1).cuda(), angle_diff), dim = -1)

        for b in range(batch_size):
            motion_beat_time = []
            for t in range(2, 33):
                if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                    if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres or angle_diff[b][t + 1] - angle_diff[b][t] >= thres):
                        motion_beat_time.append(float(t) / 15.0)
            audio_t = audio[b].numpy()
            audio_beat_time = librosa.onset.onset_detect(y=audio_t, sr=16000, units='time')

            motion_beats_sum +=len(motion_beat_time)
            if (len(motion_beat_time) == 0):
                continue
            sum = 0
            for audio_beat in audio_beat_time:
                sum += np.power(math.e, -np.min(np.power((audio_beat - motion_beat_time), 2)) / (2 * sigma * sigma))

            beat_align_score_sum += sum
            num_beats += len(audio_beat_time)


    beat_score = beat_align_score_sum/num_beats
    beat_score_no_skip = beat_align_score_sum/num_beats

    frechet_dist, feat_dist = embed_space_evaluator.get_scores()
    ha2g_diversity = embed_space_evaluator.get_diversity_scores()
    print('no_of_samples',embed_space_evaluator.get_no_of_samples())
    print("guidance_param", guidance_param)
    print('beat_score',beat_score)
    print('motion_beats_sum', motion_beats_sum)
    print('beat_score_no_skip',beat_score_no_skip)
    print('frechet_dist',frechet_dist)
    print('ha2g_diversity', ha2g_diversity)
    print('diffusionstep',args.diffusion_steps)
    print('eta', eta)
    embed_space_evaluator.reset()
    return frechet_dist, beat_score, ha2g_diversity


if __name__== "__main__":
    use_ddim = True
    fixseed(233)

    args = generate_args()
    args.batch_size = 512
    args.num_workers = 11
    if use_ddim:
        timestep_respacing = 'ddim100'
    else:
        timestep_respacing = ''
    

    # test_data_loader = build_dataloader('val', args, shuffle= True)
    test_data_loader = build_dataloader('test', args, shuffle= True)

    train_data_loader = build_dataloader('train', args, shuffle= False)
    speaker_model = train_data_loader.dataset.speaker_model

    lang_model = test_data_loader.dataset.lang_model
    args.lang_model = lang_model

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

    with torch.no_grad():
        result_list = []

        t = infer_from_testloader(sample_fn, model, args, test_data_loader, guidance_param = 1, speaker_model=speaker_model, eta = 0)
        result_list.append(t)
        print(t)

        t = infer_from_testloader(sample_fn, model, args, test_data_loader, guidance_param = 1.5, speaker_model=speaker_model, eta = 0)
        result_list.append(t)
        print(t)

        t = infer_from_testloader(sample_fn, model, args, test_data_loader, guidance_param = 2, speaker_model=speaker_model, eta = 0)
        result_list.append(t)
        print(t)

        for tup in result_list:
            print(tup)


