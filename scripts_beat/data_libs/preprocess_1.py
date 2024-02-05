import json
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

front_speakers = [1,5,7,9,10,12,13,15,16,17,18,19,21,22,23,24,26,28,29,30]
back_speaker = [2,3,4,6,8,11,14,20,25,27]

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
            'Spine':       3 ,
            'Neck':        3 ,
            'Neck1':       3 ,
            'RShoulder':   3 , 
            'RArm':        3 ,
            'RArm1':       3 ,
            'RHand':       3 ,    
            'RHandM1':     3 ,
            'RHandM2':     3 ,
            'RHandM3':     3 ,
            'RHandR':      3 ,
            'RHandR1':     3 ,
            'RHandR2':     3 ,
            'RHandR3':     3 ,
            'RHandP':      3 ,
            'RHandP1':     3 ,
            'RHandP2':     3 ,
            'RHandP3':     3 ,
            'RHandI':      3 ,
            'RHandI1':     3 ,
            'RHandI2':     3 ,
            'RHandI3':     3 ,
            'RHandT1':     3 ,
            'RHandT2':     3 ,
            'RHandT3':     3 ,
            'LShoulder':   3 , 
            'LArm':        3 ,
            'LArm1':       3 ,
            'LHand':       3 ,    
            'LHandM1':     3 ,
            'LHandM2':     3 ,
            'LHandM3':     3 ,
            'LHandR':      3 ,
            'LHandR1':     3 ,
            'LHandR2':     3 ,
            'LHandR3':     3 ,
            'LHandP':      3 ,
            'LHandP1':     3 ,
            'LHandP2':     3 ,
            'LHandP3':     3 ,
            'LHandI':      3 ,
            'LHandI1':     3 ,
            'LHandI2':     3 ,
            'LHandI3':     3 ,
            'LHandT1':     3 ,
            'LHandT2':     3 ,
            'LHandT3':     3 ,
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

split_rule_english = {
    # 4h speakers x 10
    "1, 2, 3, 4, 6, 7, 8, 9, 11, 21":{
        # 48+40+100=188mins each
        "train": [
            "0_9_9", "0_10_10", "0_11_11", "0_12_12", "0_13_13", "0_14_14", "0_15_15", "0_16_16", \
            "0_17_17", "0_18_18", "0_19_19", "0_20_20", "0_21_21", "0_22_22", "0_23_23", "0_24_24", \
            "0_25_25", "0_26_26", "0_27_27", "0_28_28", "0_29_29", "0_30_30", "0_31_31", "0_32_32", \
            "0_33_33", "0_34_34", "0_35_35", "0_36_36", "0_37_37", "0_38_38", "0_39_39", "0_40_40", \
            "0_41_41", "0_42_42", "0_43_43", "0_44_44", "0_45_45", "0_46_46", "0_47_47", "0_48_48", \
            "0_49_49", "0_50_50", "0_51_51", "0_52_52", "0_53_53", "0_54_54", "0_55_55", "0_56_56", \
            
            "0_66_66", "0_67_67", "0_68_68", "0_69_69", "0_70_70", "0_71_71",  \
            "0_74_74", "0_75_75", "0_76_76", "0_77_77", "0_78_78", "0_79_79",  \
            "0_82_82", "0_83_83", "0_84_84", "0_85_85",  \
            "0_88_88", "0_89_89", "0_90_90", "0_91_91", "0_92_92", "0_93_93",  \
            "0_96_96", "0_97_97", "0_98_98", "0_99_99", "0_100_100", "0_101_101",  \
            "0_104_104", "0_105_105", "0_106_106", "0_107_107", "0_108_108", "0_109_109",  \
            "0_112_112", "0_113_113", "0_114_114", "0_115_115", "0_116_116", "0_117_117",  \
            
            "1_2_2", "1_3_3", "1_4_4", "1_5_5", "1_6_6", "1_7_7", "1_8_8", "1_9_9", "1_10_10", "1_11_11",
        ],
        # 8+7+10=25mins each
        "val": [
            "0_57_57", "0_58_58", "0_59_59", "0_60_60", "0_61_61", "0_62_62", "0_63_63", "0_64_64", \
            "0_72_72", "0_80_80", "0_86_86", "0_94_94", "0_102_102", "0_110_110", "0_118_118", \
            "1_12_12",
        ],
        # 8+7+10=25mins each
        "test": [
           "0_1_1", "0_2_2", "0_3_3", "0_4_4", "0_5_5", "0_6_6", "0_7_7", "0_8_8", \
           "0_65_65", "0_73_73", "0_81_81", "0_87_87", "0_95_95", "0_103_103", "0_111_111", \
           "1_1_1",
        ],
    },
    
    
    # 1h speakers x 20
    "5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30":{
        # 8+7+20=35mins each
        "train": [
            "0_9_9", "0_10_10", "0_11_11", "0_12_12", "0_13_13", "0_14_14", "0_15_15", "0_16_16", \
            "0_66_66", "0_74_74", "0_82_82", "0_88_88", "0_96_96", "0_104_104", "0_112_112", "0_118_118", \
            "1_2_2", "1_3_3", 
            "1_0_0", "1_4_4", # for speaker 29 only
        ],
        # 4+3.5+5 = 12.5mins each
        # 0_65_a and 0_65_b denote the frist and second half of sequence 0_65_65
        "val": [
            "0_5_5", "0_6_6", "0_7_7", "0_8_8",  \
            "0_65_b", "0_73_b", "0_81_b", "0_87_b", "0_95_b", "0_103_b", "0_111_b", \
            "1_1_b",
        ],
        # 4+3.5+5 = 12.5mins each
        "test": [
           "0_1_1", "0_2_2", "0_3_3", "0_4_4", \
           "0_65_a", "0_73_a", "0_81_a", "0_87_a", "0_95_a", "0_103_a", "0_111_a", \
           "1_1_a",
        ],
    },
}



def cut_sequence(source_path, save_path_a, save_path_b, file_id, fps = 15, sr = 16000, tmp="/home/home/ma-user/work/datasets/beat_tmp/"):
    cut_point = 30 if file_id.split("_")[0] == "0" else 300 #in seconds
    if source_path.endswith(".npy"):
        data = np.load(source_path)
        data_a = data[:sr*cut_point]
        data_b = data[sr*cut_point:]
        np.save(save_path_a, data_a)
        np.save(save_path_b, data_b)
        
    elif source_path.endswith(".bvh"):
        copy_lines = 431 if "full" in source_path or "vis" in source_path else 0
        with open(source_path, "r") as data:
            with open(save_path_a, "w") as data_a:
                with open(save_path_b, "w") as data_b:
                    for i, line_data in enumerate(data.readlines()):
                        if i < copy_lines:
                            data_a.write(line_data)
                            data_b.write(line_data)
                        elif i < cut_point * fps:
                            data_a.write(line_data)
                        else:
                            data_b.write(line_data)
    
    elif source_path.endswith(".json"):
        with open(source_path, "r", encoding='utf-8') as data:
            json_file = json.load(data)
            with open(save_path_a, "w") as data_a:
                with open(save_path_b, "w") as data_b:
                    new_frames_a = []
                    new_frames_b = []
                    for json_data in json_file["frames"]:
                        if json_data["time"] < cut_point:
                            new_frames_a.append(json_data)
                        else:
                            new_frame = json_data.copy()
                            new_frame["time"]-=cut_point
                            new_frames_b.append(new_frame)
                    json_new_a = {"names":json_file["names"], "frames": new_frames_a}
                    json_new_b = {"names":json_file["names"], "frames": new_frames_b}
                    json.dump(json_new_a, data_a)
                    json.dump(json_new_b, data_b) 
        
    else:
        # processing in the dataloader
        shutil.copy(source_path, save_path_a)
        shutil.copy(source_path, save_path_b)
    
    #shutil.move(source_path, tmp)                        
# spilt data
speaker_names = [
    "wayne", "scott", "solomon", "lawrence", "stewart", "carla", "sophie", "catherine", "miranda", "kieks", \
    "nidal", "zhao", "lu", "zhang", "carlos", "jorge", "itoi", "daiki", "jaime", "li", \
    "ayana", "luqi", "hailing", "kexin", "goto", "reamey", "yingqing", "tiffnay", "hanieh", "katya",
]
default_path = "/your/path/to/BEAT/beat_english_15_141_origin/train"
four_hour_speakers = "1, 2, 3, 4, 6, 7, 8, 9, 11, 21".split(", ")
one_hour_speakers = "5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30".split(", ")
folders = os.listdir(default_path)
folders = list(filter(lambda x: not x.endswith('_cache'), folders))
if not os.path.exists(default_path.replace("train", "val")): os.mkdir(default_path.replace("train", "val"))
if not os.path.exists(default_path.replace("train", "test")): os.mkdir(default_path.replace("train", "test"))
endwith = []
for folder in folders:        
    if not os.path.exists(default_path.replace("train", "val")+"/"+folder): os.mkdir(default_path.replace("train", "val")+"/"+folder)
    if not os.path.exists(default_path.replace("train", "test")+"/"+folder): os.mkdir(default_path.replace("train", "test")+"/"+folder)
    endwith.append(os.listdir(default_path+"/"+folder)[500].split(".")[-1])
tmp_speakers = [2, 4, 6, 8]
# front_speakers = [1,5,7,9,10,12,13,15,16,17,18,19,21,22,23,24,26,28,29,30]
# for speaker_id in tqdm(range(1,31)):

errors = []
for speaker_id in tmp_speakers:
    print('process',speaker_id)
    val = split_rule_english["5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30"]["val"] if str(speaker_id) in one_hour_speakers else split_rule_english["1, 2, 3, 4, 6, 7, 8, 9, 11, 21"]["val"]
    test = split_rule_english["5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30"]["test"] if str(speaker_id) in one_hour_speakers else split_rule_english["1, 2, 3, 4, 6, 7, 8, 9, 11, 21"]["test"]
    for file_id in val:
        for ide, folder in enumerate(folders):
            if "b" in file_id:
                cut_sequence(
                    source_path=f"{default_path}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id.split('_')[0]}_{file_id.split('_')[1]}_{file_id.split('_')[1]}.{endwith[ide]}",
                    save_path_a=f"{default_path.replace('train', 'test')}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id.split('_')[0]}_{file_id.split('_')[1]}_a.{endwith[ide]}",
                    save_path_b=f"{default_path.replace('train', 'val')}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id.split('_')[0]}_{file_id.split('_')[1]}_b.{endwith[ide]}",
                    file_id = file_id,
                        )
            else:
                if os.path.exists(f"{default_path.replace('train', 'val')}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id}.{endwith[ide]}"):
                    print(f'found')
                    continue
                else:
                    try:
                        shutil.move(f"{default_path}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id}.{endwith[ide]}", f"{default_path.replace('train', 'val')}/{folder}/")
                    except:
                        print(f"{default_path}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id}.{endwith[ide]}")
                        errors.append(f"{default_path}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id}.{endwith[ide]}")
    for file_id in test:
        for ide, folder in enumerate(folders):
            if "a" in file_id:
                pass
            else:
                #pass
                if os.path.exists(f"{default_path.replace('train', 'test')}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id}.{endwith[ide]}"):
                    continue
                else:
                    try:
                        shutil.move(f"{default_path}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id}.{endwith[ide]}", f"{default_path.replace('train', 'test')}/{folder}/")
                    except:
                        print(f"{default_path}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id}.{endwith[ide]}")
                        errors.append(f"{default_path}/{folder}/{speaker_id}_{speaker_names[speaker_id-1]}_{file_id}.{endwith[ide]}")

print(errors)