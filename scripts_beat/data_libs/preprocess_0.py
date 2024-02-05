import json, librosa
import os
import sys
import pandas as pd
import pickle
import math
import shutil
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
joint_list = {
    "beat_joints" : {
        'Hips':         [6,6],
        'Spine':        [3,9],
        'Spine1':       [3,12],
        'Spine2':       [3,15],
        'Spine3':       [3,18],
        'Neck':         [3,21],
        'Neck1':        [3,24],
        'Head':         [3,27],
        'HeadEnd':      [3,30],

        'RShoulder':    [3,33], 
        'RArm':         [3,36],
        'RArm1':        [3,39],
        'RHand':        [3,42],    
        'RHandM1':      [3,45],
        'RHandM2':      [3,48],
        'RHandM3':      [3,51],
        'RHandM4':      [3,54],

        'RHandR':       [3,57],
        'RHandR1':      [3,60],
        'RHandR2':      [3,63],
        'RHandR3':      [3,66],
        'RHandR4':      [3,69],

        'RHandP':       [3,72],
        'RHandP1':      [3,75],
        'RHandP2':      [3,78],
        'RHandP3':      [3,81],
        'RHandP4':      [3,84],

        'RHandI':       [3,87],
        'RHandI1':      [3,90],
        'RHandI2':      [3,93],
        'RHandI3':      [3,96],
        'RHandI4':      [3,99],

        'RHandT1':      [3,102],
        'RHandT2':      [3,105],
        'RHandT3':      [3,108],
        'RHandT4':      [3,111],

        'LShoulder':    [3,114], 
        'LArm':         [3,117],
        'LArm1':        [3,120],
        'LHand':        [3,123],    
        'LHandM1':      [3,126],
        'LHandM2':      [3,129],
        'LHandM3':      [3,132],
        'LHandM4':      [3,135],

        'LHandR':       [3,138],
        'LHandR1':      [3,141],
        'LHandR2':      [3,144],
        'LHandR3':      [3,147],
        'LHandR4':      [3,150],

        'LHandP':       [3,153],
        'LHandP1':      [3,156],
        'LHandP2':      [3,159],
        'LHandP3':      [3,162],
        'LHandP4':      [3,165],

        'LHandI':       [3,168],
        'LHandI1':      [3,171],
        'LHandI2':      [3,174],
        'LHandI3':      [3,177],
        'LHandI4':      [3,180],

        'LHandT1':      [3,183],
        'LHandT2':      [3,186],
        'LHandT3':      [3,189],
        'LHandT4':      [3,192],

        'RUpLeg':       [3,195],
        'RLeg':         [3,198],
        'RFoot':        [3,201],
        'RFootF':       [3,204],
        'RToeBase':     [3,207],
        'RToeBaseEnd':  [3,210],

        'LUpLeg':       [3,213],
        'LLeg':         [3,216],
        'LFoot':        [3,219],
        'LFootF':       [3,222],
        'LToeBase':     [3,225],
        'LToeBaseEnd':  [3,228],
        },
    
    "beat_141" : {
            'Spine':       3 , #0
            'Neck':        3 , #3
            'Neck1':       3 , #6
            'RShoulder':   3 , #9
            'RArm':        3 , #12
            'RArm1':       3 , #15
            'RHand':       3 , #18   
            'RHandM1':     3 , #21
            'RHandM2':     3 , #24
            'RHandM3':     3 , #27
            'RHandR':      3 , #30
            'RHandR1':     3 , #33
            'RHandR2':     3 , #36
            'RHandR3':     3 , #39
            'RHandP':      3 , #42
            'RHandP1':     3 , #45
            'RHandP2':     3 , #48
            'RHandP3':     3 , #51
            'RHandI':      3 , #54
            'RHandI1':     3 , #57
            'RHandI2':     3 , #60
            'RHandI3':     3 , #63
            'RHandT1':     3 , #66
            'RHandT2':     3 , #69
            'RHandT3':     3 , #72
            'LShoulder':   3 , #75
            'LArm':        3 , #78
            'LArm1':       3 , #81
            'LHand':       3 , #84
            'LHandM1':     3 , #87
            'LHandM2':     3 , #90
            'LHandM3':     3 , #93
            'LHandR':      3 , #96
            'LHandR1':     3 , #99
            'LHandR2':     3 , #102
            'LHandR3':     3 , #105
            'LHandP':      3 , #108
            'LHandP1':     3 , #111
            'LHandP2':     3 , #114
            'LHandP3':     3 , #117
            'LHandI':      3 , #120
            'LHandI1':     3 , #123
            'LHandI2':     3 , #126
            'LHandI3':     3 , #129
            'LHandT1':     3 , #132
            'LHandT2':     3 , #135
            'LHandT3':     3 , #138
        },
    
    "beat_27" : {
            'Spine':       3 ,
            'Neck':        3 ,
            'Neck1':       3 ,
            'RShoulder':   3 , 
            'RArm':        3 ,
            'RArm1':       3 ,
            'LShoulder':   3 , 
            'LArm':        3 ,
            'LArm1':       3 ,     
        },
}
#calculate mean and build cache for data. 
target_fps = 15
ori_list = joint_list["beat_joints"]
target_list = joint_list["beat_141"]
ori_data_path = "/your/path/to/BEAT/data_all/beat_english_v0.2.0"
#wave cache from a = librosa.load(sr=16000) and np.save(a)
ori_data_path_npy = "/your/path/to/BEAT/data_all/beat_english_v0.2.0"
ori_data_path_ann = "/your/path/to/BEAT/data_all/beat_english_v0.2.0"
cache_path = f"/your/path/to/BEAT/beat_english_{target_fps}_141_origin/"
reduce_factor_json = int(60/target_fps)
reduce_factor_bvh = int(120/target_fps)
print(f"target_fps: {target_fps}, reduce json {reduce_factor_json}, reduce bvh {reduce_factor_bvh}")
speakers = sorted(os.listdir(ori_data_path),key=str,)
npy_s_v = []
npy_s_k = []
json_s_v = []
bvh_s_v = []

load_type = "train"
if not os.path.exists(f"{cache_path}"): 
    os.mkdir(cache_path)
if not os.path.exists(f"{cache_path}{load_type}/"): 
    os.mkdir(f"{cache_path}{load_type}/")
    os.mkdir(f"{cache_path}{load_type}/wave16k/")
    os.mkdir(f"{cache_path}{load_type}/bvh_rot/")
    os.mkdir(f"{cache_path}{load_type}/bvh_full/")
    os.mkdir(f"{cache_path}{load_type}/bvh_rot_vis/")
    os.mkdir(f"{cache_path}{load_type}/bvh_full_vis/")

    os.mkdir(f"{cache_path}{load_type}/facial52/")
    os.mkdir(f"{cache_path}{load_type}/text/")
    os.mkdir(f"{cache_path}{load_type}/emo/")
    os.mkdir(f"{cache_path}{load_type}/sem/")     

tmp_speakers = [2,4,6,8]
front_speakers = [1,5,7,9,10,12,13,15,16,17,18,19,21,22,23,24,26,28,29,30] # root oriention faced front, may have been uniformed in latest BEAT
back_speaker = [2,3,4,6,8,11,14,20,25,27] # root oriention faced back, may have been uniformed in latest BEAT
# for speaker in range(1, 31):#replace to 1, 31 for all speakers
for speaker in tmp_speakers:#replace to 1, 31 for all speakers
    print('speaker:', speaker)
    all_data = list(filter(lambda x: x.endswith("wav"), os.listdir(f'{ori_data_path_npy}/{speaker}')))
    npy_all = []
    json_all = []
    bvh_all = []   
    for ii, file in tqdm(enumerate(all_data)):
        file = file[:-4]
        wav_file, _ = librosa.load(f"{ori_data_path_npy}/{file.split('_')[0]}/{file}.wav", sr=16000)
        np.save(f"{cache_path}{load_type}/wave16k/{file}.npy", wav_file)
        # shutil.copy(f"{ori_data_path_npy}/{file.split('_')[0]}/{file}.npy", f"{cache_path}{load_type}/wave16k/{file}.npy")
        try:shutil.copy(f"{ori_data_path}/{file.split('_')[0]}/{file}.TextGrid", f"{cache_path}{load_type}/text/{file}.TextGrid")
        except: print(f"{file}.TextGrid")
        try: shutil.copy(f"{ori_data_path_ann}/{file.split('_')[0]}/{file}.txt", f"{cache_path}{load_type}/sem/{file}.txt")
        except: print(f"{file}.txt")
        try: shutil.copy(f"{ori_data_path_ann}/{file.split('_')[0]}/{file}.csv", f"{cache_path}{load_type}/emo/{file}.csv")
        except: print(f"{file}.csv")
        # npy_all.extend(list(wav_file))

        with open(f"{ori_data_path}/{file.split('_')[0]}/{file}.json", "r", encoding='utf-8') as json_file_raw:
            json_file = json.load(json_file_raw)
            with open(f"{cache_path}{load_type}/facial52/{file}.json", "w") as reduced_json:
                counter = 0
                new_frames_list = []
                for json_data in json_file["frames"]:
                    if counter % reduce_factor_json == 0:
                        # json_all.append(json_data["weights"])
                        new_frames_list.append(json_data)
                    counter += 1
                json_new = {"names":json_file["names"], "frames": new_frames_list}
                json.dump(json_new, reduced_json)

            with open(f"{ori_data_path}/{file.split('_')[0]}/{file}.bvh", "r") as bvh_file:
                with open(f"{cache_path}{load_type}/bvh_rot/{file}.bvh", "w") as reduced_raw_bvh:
                    with open(f"{cache_path}{load_type}/bvh_full_vis/{file}.bvh", "w") as reduced_full_bvh:
                        with open(f"{cache_path}{load_type}/bvh_rot_vis/{file}.bvh", "w") as reduced_trainable_bvh:
                            with open(f"{cache_path}{load_type}/bvh_full/{file}.bvh", "w") as my_raw_bvh:
                                for i, line_data in enumerate(bvh_file.readlines()):
                                    if i < 432: 
                                        reduced_full_bvh.write(line_data)
                                        reduced_trainable_bvh.write(line_data)
                                    if i >= 432:
                                        data = np.fromstring(line_data, dtype=float, sep=' ')
                                        if i % reduce_factor_bvh == 0:
                                            # bvh_all.append(data)
                                            reduced_full_bvh.write(line_data)
                                            my_raw_bvh.write(line_data)
                                            trainable_rotation = np.zeros_like(data)
                                            for k, v in target_list.items():
                                                trainable_rotation[ori_list[k][1]-v:ori_list[k][1]] = data[ori_list[k][1]-v:ori_list[k][1]]

                                            trainable_line_data = np.array2string(trainable_rotation, max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                                            reduced_trainable_bvh.write(trainable_line_data[1:-1]+"\n")
                                            data_rotation = np.zeros((1))   
                                            for k, v in target_list.items():
                                                data_rotation = np.concatenate((data_rotation, data[ori_list[k][1]-v:ori_list[k][1]]))                             
                                                raw_line_data = np.array2string(data_rotation[1:], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                                            reduced_raw_bvh.write(raw_line_data[1:-1]+"\n")


