import lmdb
import torch
import librosa, pyarrow
import numpy as np
import rot_utils

def extract_melspectrogram(y, sr=16000):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec

mean_pose = np.load('/p300/wangchy/zhiyh/BEAT_dataset/bvh_mean.npy')
std_pose = np.load('/p300/wangchy/zhiyh/BEAT_dataset/bvh_std.npy')  

def build_data_with_beat(mode = "train"):
    preloaded_dir = f"/p300/wangchy/zhiyh/BEAT_dataset/{mode}/bvh_rot_2_4_6_8_cache"
    lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
    with lmdb_env.begin() as txn:
        n_samples = txn.stat()['entries']
    map_size = 1024 * 50  # in MB
    map_size <<= 20  # in B
    out_lmdb_dir = f"/p300/wangchy/zhiyh/BEAT_dataset/{mode}/my6d_bvh_rot_2_4_6_8_cache"
    dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
    with dst_lmdb_env.begin(write=True) as dst_txn:
        for idx in range(n_samples):
            print(f"{mode}, {idx}/{n_samples}")
            with lmdb_env.begin(write=False) as txn:
                key = "{:005}".format(idx).encode("ascii")
                sample = txn.get(key)

                sample = pyarrow.deserialize(sample)

                tar_pose, in_audio, in_facial, in_word, vid, emo, sem = sample

                mel = extract_melspectrogram(in_audio)

                pose_euler = (tar_pose*std_pose)+mean_pose
                
                pose_euler = (torch.Tensor(pose_euler.reshape(-1, 3))/180)*np.pi
            
                rot_matrix = rot_utils.euler_angles_to_matrix(pose_euler, "XYZ")
                rot6d = rot_utils.matrix_to_rotation_6d(rot_matrix)

                rot6d = rot6d.reshape(tar_pose.shape[0],-1).numpy()

                aux_info = {
                    'mel': mel,
                    'rot6d': rot6d,
                }
                v = [tar_pose, in_audio, in_facial, in_word, vid, emo, sem, aux_info]
                # # save
                k = "{:005}".format(idx).encode("ascii")
                v = pyarrow.serialize(v).to_buffer()
                dst_txn.put(k, v)


build_data_with_beat("train")
# build_data_with_beat("val")
# build_data_with_beat("test")

