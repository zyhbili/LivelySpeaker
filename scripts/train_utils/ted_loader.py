from torch.utils.data import DataLoader
import sys
import numpy as np
[sys.path.append(i) for i in ['.', '..']]
from data_loader.lmdb_data_loader import *
from torch.utils.data import DataLoader
from utils.vocab_utils import build_vocab
from tqdm import tqdm
import os

import torch
torch.manual_seed(233)
import random
random.seed(233)
np.random.seed(233)

mean_dir_vec= np.array([ 0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916])
mean_pose = np.array([ 0.0000306,  0.0004946,  0.0008437,  0.0033759, -0.2051629, -0.0143453,  0.0031566, -0.3054764,  0.0411491,  0.0029072, -0.4254303, -0.001311 , -0.1458413, -0.1505532, -0.0138192, -0.2835603,  0.0670333,  0.0107002, -0.2280813,  0.112117 , 0.2087789,  0.1523502, -0.1521499, -0.0161503,  0.291909 , 0.0644232,  0.0040145,  0.2452035,  0.1115339,  0.2051307])


data_base = './datasets/ted_dataset'
fasttext_dir = './datasets/ted_dataset/fasttext'


vocab_cache_path = os.path.join(data_base, 'vocab_cache.pkl')
lang_model = build_vocab('words', [None], vocab_cache_path, f'{fasttext_dir}/crawl-300d-2M-subword.bin', 300)


def build_dataloader(mode, args, shuffle = False):
    assert mode in ["train","val", "test"]
    prefix = "lmdb"
    if mode == 'train':
        dataset = SpeechMotionDataset(f'{data_base}/lmdb_{mode}',
                                n_poses=34,
                                subdivision_stride=10,
                                pose_resampling_fps=15,
                                mean_dir_vec=mean_dir_vec,
                                mean_pose= mean_pose,
                                remove_word_timing=False, args=args) 
        
    else:
        dataset = SpeechMotionDataset(f'{data_base}/{prefix}_{mode}',
                                    n_poses=34,
                                    subdivision_stride=10,
                                    pose_resampling_fps=15,
                                    mean_dir_vec=mean_dir_vec,
                                    mean_pose= mean_pose,
                                    remove_word_timing=False, args=args) 
    
    dataset.set_lang_model(lang_model)
    dataset.prefix = prefix
    print(mode, f"len {len(dataset)}")
    train_dataloader = DataLoader(dataset, shuffle=shuffle,batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=word_seq_collate_fn)
    return train_dataloader


