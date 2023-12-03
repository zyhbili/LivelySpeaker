import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import pyarrow
# from sklearn.preprocessing import normalize
import librosa 
# import scipy.io.wavfile
# from scipy import signal
# from .build_vocab import Vocab

def make_unique_key(speakers_list):
    if len(speakers_list) == 1:
        return str(speakers_list[0])
    return '_'.join(sorted(speakers_list, key = lambda x:int(x)))

def my_exists(path_list):
    flag = 1
    for path in path_list:
        if not os.path.exists(path):
            print(f'{path} not exists continue')
            flag = 0
            break
    return flag

def extract_melspectrogram(y, sr=16000):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec



class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=False):
        self.loader_type = loader_type
        self.new_cache = args.new_cache
        self.pose_length = args.pose_length #34
        self.stride = args.stride #10
        self.pose_fps = args.pose_fps #15
        self.pose_dims = args.pose_dims # 141
        self.mean_pose = np.load(args.root_path+args.mean_pose_path+"/bvh_mean.npy")
        self.std_pose = np.load(args.root_path+args.std_pose_path+"/bvh_std.npy")

        self.audio_norm = args.audio_norm
        if self.audio_norm:
            self.mean_audio = np.load(args.mean_audio_path+f"{args.audio_rep}/mean.npy")
            self.std_audio = np.load(args.std_audio_path+f"{args.audio_rep}/std.npy")
        self.loader_type = loader_type
        self.audio_rep = args.audio_rep
        self.pose_rep = args.pose_rep
        self.facial_rep = args.facial_rep
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.audio_fps = args.audio_fps
        self.speaker_id = args.speaker_id
        
        self.disable_filtering = args.disable_filtering
        self.clean_first_seconds = args.clean_first_seconds
        self.clean_final_seconds = args.clean_final_seconds


        
        if loader_type == "train":
            self.data_dir = args.root_path + args.train_data_path
        elif loader_type == "val":
            self.data_dir = args.root_path + args.val_data_path
        else:
            self.data_dir = args.root_path + args.test_data_path
        
        if self.word_rep is not "None":
            with open(args.vocab_path, 'rb') as f:
                self.lang_model = pickle.load(f)

        # import pdb;pdb.set_trace()
        # self.cache_speakers = [2,4,6,8]
        # self.cache_speakers = list(map(lambda x:int(x),args.speakers))
        self.cache_speakers = args.speakers
        self.use_sem =  args.use_sem
        # import pdb;pdb.set_trace()
        if loader_type == "finaltest":
            self.data_dir = self.data_dir.replace("test", 'finaltest')
            preloaded_dir = self.data_dir +f"my6d_{self.pose_rep}_{make_unique_key(self.cache_speakers)}_cache"
            self.preloaded_dir = preloaded_dir
        else:
            # preloaded_dir = self.data_dir +f"{self.pose_rep}_{make_unique_key(self.cache_speakers)}_cache"
            if args.use_sem:
                preloaded_dir = self.data_dir +f"my6d_{self.pose_rep}_sem_{make_unique_key(self.cache_speakers)}_cache"
            else:
                preloaded_dir = self.data_dir +f"my6d_{self.pose_rep}_{make_unique_key(self.cache_speakers)}_cache"

        self.cache_speakers = list(map(lambda x:int(x),args.speakers))
        # self.build_cache(preloaded_dir)
        logger.info(f"Audio bit rate: {self.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        # pose_length_extended = int(round(self.pose_length))
        logger.info("loading the dataset cache...")
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]
        # import pdb;pdb.set_trace()
         
            
    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        # preloaded_dir = self.data_dir + f"{self.pose_rep}_cache"
        # pose_length_extended = int(round(self.pose_length))
        logger.info("Creating the dataset cache...")

        if self.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)

        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.disable_filtering, 
                self.clean_first_seconds, self.clean_final_seconds,
                is_test=False)
        

        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]
        
    
    def __len__(self):
        #print("in_dataset:", self.n_samples)
        return self.n_samples
    
    def cache_generation_env(self, test_lmdb):
        self.n_out_samples = 0
        out_lmdb_dir = self.preloaded_dir 
        map_size = int(1024 * 1024 * 2048 * (self.audio_fps/16000)**3 * 4 *10)  # in 1024 MB
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        n_filtered_out = defaultdict(int)
        for idx in range(64):
            # if idx =
            batch = test_lmdb.__getitem__(idx)
            # tar_pose, in_audio, in_facial, in_word, vid, emo, sem, aux_info = batch
            pose_each_file, audio_each_file, facial_each_file, word_each_file, vid_each_file\
            , emo_each_file, sem_each_file, aux_info = batch

            pose_each_file = pose_each_file.numpy()
            facial_each_file = facial_each_file.numpy()

            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_each_file, facial_each_file, word_each_file,
                vid_each_file, emo_each_file, sem_each_file,
                self.disable_filtering, 
                self.clean_first_seconds, self.clean_final_seconds, is_test=False
                ) 
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                
            
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
        pass

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0

        # t = glob.glob(os.path.join(self.data_dir, f"{self.audio_rep}") + "/*.npy")
        # for a in t:
        #     try:
        #         print(int(a.split('/')[-1].split('_')[0]))
        #     except:
        #         print(a)
        #         import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
 
        # tt = list(filter(lambda x: int(x.split('/')[-1].split('_')[0]) in self.cache_speakers , t))
        # import pdb;pdb.set_trace()

        audio_files = sorted(filter(lambda x: int(x.split('/')[-1].split('_')[0]) in self.cache_speakers ,glob.glob(os.path.join(self.data_dir, f"{self.audio_rep}",) + "/*.npy")), key=str,)
        
        # create db for samples
        map_size = int(1024 * 1024 * 2048 * (self.audio_fps/16000)**3 * 4 *10)  # in 1024 MB
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        n_filtered_out = defaultdict(int)

        for audio_file in audio_files:
            # print(audio_file)
            audio_each_file = []
            pose_each_file = [] 
            facial_each_file = []
            word_each_file = []
            vid_each_file = []
            emo_each_file = []
            sem_each_file = []
            
            id_audio = audio_file.split('/')[-1][:-4]

            pose_file_path = audio_file[:-4].replace(self.audio_rep, self.pose_rep) + '.bvh'
            facial_file_path = audio_file[:-4].replace(self.audio_rep, self.facial_rep) + '.json'
            word_file_path = audio_file[:-4].replace(self.audio_rep, self.word_rep) + '.TextGrid'
            sem_file_path = audio_file[:-4].replace(self.audio_rep, self.sem_rep) + '.txt'
            emo_file_path = audio_file[:-4].replace(self.audio_rep, self.emo_rep) + '.csv'

            if my_exists([pose_file_path, facial_file_path, word_file_path, sem_file_path, emo_file_path]):
                pass
            else:
                continue
            
            if self.speaker_id:
                vid_each_file.append(id_audio)
                audio_each_file = np.load(audio_file)
            with open(pose_file_path, "r") as pose_data:
                for j, line in enumerate(pose_data.readlines()):
                    # print(line)
                    data = np.fromstring(line, dtype=float, sep=" ") # 1*27 e.g., 27 rotation 
                    pose_each_file.append(data)
            pose_each_file = np.array(pose_each_file) # n frames * 27

            
            with open(facial_file_path, 'r') as facial_data_file:
                facial_data = json.load(facial_data_file)
                # facial_factor = math.ceil(1/((facial_data['frames'][20]['time'] - facial_data['frames'][10]['time'])/10))//self.pose_fps
                #print(facial_data['frames'][20]['time'] - facial_data['frames'][10]['time']) 
                facial_factor = 1
                for j, frame_data in enumerate(facial_data['frames']):
                    # 60FPS to 15FPS
                    if j % facial_factor == 0:
                        facial_each_file.append(frame_data['weights']) 
            facial_each_file = np.array(facial_each_file)




            tgrid = tg.TextGrid.fromFile(word_file_path)
            for i in range(pose_each_file.shape[0]):
                found_flag = False
                current_time = i/self.pose_fps
                for word in tgrid[0]:
                    word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                    if word_s<=current_time and current_time<=word_e:
                        if word_n == "":
                            #TODO now don't have eos and sos token
                            word_each_file.append(self.lang_model.PAD_token)
                        else:    
                            word_each_file.append(self.lang_model.get_word_index(word_n))
                        found_flag = True
                        break
                    else: continue   
                if not found_flag: word_each_file.append(self.lang_model.UNK_token)
            word_each_file = np.array(word_each_file)
    
 
            emo_all = pd.read_csv(emo_file_path, 
                sep=',', 
                names=["name", "start", "end", "duration", "score"])
            for i in range(pose_each_file.shape[0]):
                found_flag = False
                for j, (start, end, score) in enumerate(zip(emo_all['start'],emo_all['end'], emo_all['score'])):
                    
                    current_time = i/self.pose_fps
                    if start<=current_time and current_time<=end: 
                        emo_each_file.append(score)
                        found_flag=True
                        break
                    else: continue 
                if not found_flag: emo_each_file.append(0)
            emo_each_file = np.array(emo_each_file)
                #print(emo_each_file)
    

            try:
                sem_all = pd.read_csv(sem_file_path, 
                    sep='\t', 
                    names=["name", "start", "end", "duration", "score", 'word'])
            except:
                sem_all = pd.read_csv(sem_file_path, 
                    sep='\t', 
                    names=["name", "start", "end", "duration", "score"])
            for i in range(pose_each_file.shape[0]):
                found_flag = False
                for j, (start, end, score) in enumerate(zip(sem_all['start'],sem_all['end'], sem_all['score'])):
                    current_time = i/self.pose_fps
                    if start<=current_time and current_time<=end: 
                        sem_each_file.append(score)
                        found_flag=True
                        break
                    else: continue 
                if not found_flag: sem_each_file.append(0.)
            sem_each_file = np.array(sem_each_file)

            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_each_file, facial_each_file, word_each_file,
                vid_each_file, emo_each_file, sem_each_file,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                ) 
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                                
            
        # print stats
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
    
    def _sample_from_clip(
        self, dst_lmdb_env, audio_each_file, pose_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        # logger.info(f"alignment: {alignment}")
        audio_start = 0 #int(alignment[0] * self.audio_fps)
        pose_start = 0 #int(alignment[1] * self.pose_fps)
#         print(audio_each_file)
#         print(pose_each_file)
        logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        audio_each_file = audio_each_file[pose_start:]
        pose_each_file = pose_each_file[pose_start:]
        logger.info(f"after: {audio_each_file.shape} {pose_each_file.shape}")
        round_seconds_skeleton = pose_each_file.shape[0] // self.pose_fps  # assume 1500 frames / 15 fps = 100 s
        round_seconds_audio = len(audio_each_file) // self.audio_fps # assume 16,000,00 / 16,000 = 100 s
        if facial_each_file != []:
            round_seconds_facial = facial_each_file.shape[0] // self.pose_fps
            logger.info(f"audio: {round_seconds_skeleton}s, pose: {round_seconds_audio}s, facial: {round_seconds_facial}s")
            round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
            max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
            if round_seconds_skeleton != max_round: 
                logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
        else:
            logger.info(f"audio: {round_seconds_skeleton}s, pose: {round_seconds_audio}s")
            round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
            max_round = max(round_seconds_audio, round_seconds_skeleton)
            if round_seconds_skeleton != max_round: 
                logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.audio_fps * clip_s_t, clip_e_t * self.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.pose_fps, clip_e_t * self.pose_fps # [150,90*15]

        if is_test:# stride = length for test
            self.pose_length = clip_e_f_pose - clip_s_f_pose
            self.stride = self.pose_length
        audio_short_length = math.floor(self.pose_length / self.pose_fps * self.audio_fps)
        num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - self.pose_length) / self.stride) + 1
        """
        for audio sr = 16000, fps = 15, pose_length = 34, 
        audio short length = 36266.7 -> 36266
        this error is fine.
        """
        logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")
        logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {self.pose_length}")
        logger.info(f"{num_subdivision} clips is expected with stride {self.stride}") 

        n_filtered_out = defaultdict(int)
        sample_pose_list = []
        sample_audio_list = []
        sample_facial_list = []
        sample_word_list = []
        sample_vid_list = []
        sample_emo_list = []
        sample_sem_list = []
        
        for i in range(num_subdivision): # cut into around 2s chip, (self npose)
            start_idx = clip_s_f_pose + i * self.stride
            fin_idx = start_idx + self.pose_length # 34
            audio_start = clip_s_f_audio + math.floor(i * self.stride * self.audio_fps / self.pose_fps)
            audio_end = audio_start + audio_short_length
            # print(start_idx, fin_idx, audio_start, audio_end)
            sample_pose = pose_each_file[start_idx:fin_idx]
            # print(sample_pose.shape)
 
            if audio_end > clip_e_f_audio:  # correct size mismatch between poses and audio
                n_padding = audio_end - clip_e_f_audio
                logger.warning(f"padding audio for length {n_padding}")
                padded_data = np.pad(audio_each_file, (0, n_padding), mode="symmetric")
                sample_audio = padded_data[audio_start:audio_end]
            else:
                sample_audio = audio_each_file[audio_start:audio_end]

            
            if facial_each_file != []: 
                sample_facial = facial_each_file[start_idx:fin_idx]
                # print(sample_facial.shape)
                # print(sample_pose.shape)
                if sample_pose.shape[0] != sample_facial.shape[0]:
                    logger.warning(f"skip {sample_pose.shape}, {sample_facial.shape}")
                    continue
            else: 
                sample_facial = np.array([-1])
                
            sample_vid = vid_each_file
            start_time = start_idx/self.pose_fps
            end_time = fin_idx/self.pose_fps
            sample_word = []
            if word_each_file != []:
                sample_word = word_each_file[start_idx:fin_idx]
            else: sample_word = np.array([-1])    
            #print(sample_word)
            
            if len(set(sample_word))<4:
                continue
            if emo_each_file != []:
                sample_emo = emo_each_file[start_idx:fin_idx]
                #print(sample_emo)
            else:                   
                sample_emo = np.array([-1])                      
 
            if sem_each_file != []:
                sample_sem = sem_each_file[start_idx:fin_idx]

                if self.use_sem and (sample_sem<0.4).all():
                    continue
                #print(sample_sem)
            else:   
                sample_sem = np.array([-1])
            
                                  
            if sample_audio.any() != None:
                # filtering motion skeleton data
                sample_pose, filtering_message = MotionPreprocessor(sample_pose, self.mean_pose).get()
                is_correct_motion = (sample_pose != [])
                if is_correct_motion or disable_filtering:
                    sample_pose_list.append(sample_pose)
                    sample_audio_list.append(sample_audio)
                    sample_facial_list.append(sample_facial)
                    sample_word_list.append(sample_word)
                    sample_vid_list.append(sample_vid)
                    sample_emo_list.append(sample_emo)
                    sample_sem_list.append(sample_sem)
                else:
                    n_filtered_out[filtering_message] += 1
        if len(sample_pose_list) > 0:
            with dst_lmdb_env.begin(write=True) as txn:
                for pose, audio, facial, word, vid, emo, sem in zip(sample_pose_list,
                                                    sample_audio_list,
                                                    sample_facial_list,
                                                    sample_word_list,
                                                    sample_vid_list,
                                                    sample_emo_list,
                                                    sample_sem_list, 
                                                    ):
                    # normalized_pose = self.normalize_pose(pose, self.mean_pose, self.std_pose)
                    normalized_pose = pose
                    # import pdb;pdb.set_trace()

                    # # save
                    # import pdb;pdb.set_trace()
                    k = "{:005}".format(self.n_out_samples).encode("ascii")
                    v = [normalized_pose, audio, facial, word, vid, emo, sem]
                    # print(v)
                    if v is None:
                        import pdb;pdb.set_trace()

                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1
        return n_filtered_out


    @staticmethod
    def normalize_pose(dir_vec, mean_pose, std_pose=None):
        return (dir_vec - mean_pose) / std_pose 

    @staticmethod
    def unnormalize_data(normalized_data, data_mean, data_std, dimensions_to_ignore):
        """
        this method is from https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
        """
        T = normalized_data.shape[0]
        D = data_mean.shape[0]

        origData = np.zeros((T, D), dtype=np.float32)
        dimensions_to_use = []
        for i in range(D):
            if i in dimensions_to_ignore:
                continue
            dimensions_to_use.append(i)
        dimensions_to_use = np.array(dimensions_to_use)

        origData[:, dimensions_to_use] = normalized_data

        # potentially inefficient, but only done once per experiment
        stdMat = data_std.reshape((1, D))
        stdMat = np.repeat(stdMat, T, axis=0)
        meanMat = data_mean.reshape((1, D))
        meanMat = np.repeat(meanMat, T, axis=0)
        origData = np.multiply(origData, stdMat) + meanMat

        return origData
   
    
    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            
            # tar_pose, in_audio, in_facial, in_word, vid, emo, sem = sample
            tar_pose, in_audio, in_facial, in_word, vid, emo, sem, aux_info = sample

            # mel = extract_melspectrogram(in_audio)
            if isinstance(vid, int):
                pass
            else:
                vid = int(vid[0].split("_")[0])

            # emo = torch.from_numpy(emo).int()
            # sem = torch.from_numpy(sem).float() 
            # in_audio = torch.from_numpy(in_audio).float() 
            # in_word = torch.from_numpy(in_word).int()  
        
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                in_facial = torch.from_numpy(in_facial).float()
                            
            else:
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()

            text = []
            last_word = 'zhiyh'
            for word_token in in_word:
                word = self.lang_model.index2word[int(word_token)]
                if word in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
                    continue

                if word == last_word:
                    continue
                else:
                    last_word = word
                    # word = word.replace("'","'")
                    # print(word)
                    text.append(word)

            # aux_info = {}
            sentence = ' '.join(text)
            prompt = "A person is talking: " #1

            # aux_info['sentence'] = prompt + '"' +sentence + '"'
            aux_info['sentence'] = sentence

            # aux_info['rot6d'] may need normalize by subtracting mean


            return tar_pose, in_audio, in_facial, in_word, vid, emo, sem, aux_info

         
class MotionPreprocessor:
    def __init__(self, skeletons, mean_pose):
        self.skeletons = skeletons
        self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        # if self.skeletons != []:
        #     if self.check_pose_diff():
        #         self.skeletons = []
        #         self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
        diff = np.abs(self.skeletons - self.mean_pose) # 186*1
        diff = np.mean(diff)

        # th = 0.017
        th = 0.02 #0.02  # exclude 3594
        if diff < th:
            if verbose:
                print("skip - check_pose_diff {:.5f}".format(diff))
            return True
#         th = 3.5 #0.02  # exclude 3594
#         if 3.5 < diff < 5:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
        else:
            if verbose:
                print("pass - check_pose_diff {:.5f}".format(diff))
            return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False
