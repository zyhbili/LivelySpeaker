import logging
import os
import pickle

import lmdb
import pyarrow

from model.vocab import Vocab
import numpy as np
import torch

def build_vocab(name, dataset_list, cache_path, word_vec_path=None, feat_dim=None):
    logging.info('  building a language model...')
    if not os.path.exists(cache_path):
        lang_model = Vocab(name)
        for dataset in dataset_list:
            logging.info('    indexing words from {}'.format(dataset.lmdb_dir))
            index_words(lang_model, dataset.lmdb_dir)

        if word_vec_path is not None:
            lang_model.load_word_vectors(word_vec_path, feat_dim)

        with open(cache_path, 'wb') as f:
            pickle.dump(lang_model, f)
    else:
        logging.info('    loaded from {}'.format(cache_path))
        with open(cache_path, 'rb') as f:
            lang_model = pickle.load(f)
        if word_vec_path is None:
            lang_model.word_embedding_weights = None
        elif lang_model.word_embedding_weights.shape[0] != lang_model.n_words:
            logging.warning('    failed to load word embedding weights. check this')
            assert False
        # else:
        #     lang_model.load_word_vectors(word_vec_path, feat_dim)

    return lang_model


def index_words(lang_model, lmdb_dir):
    lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    txn = lmdb_env.begin(write=False)
    cursor = txn.cursor()

    for key, buf in cursor:
        video = pyarrow.deserialize(buf)

        for clip in video['clips']:
            for word_info in clip['words']:
                word = word_info[0]
                lang_model.index_word(word)

    lmdb_env.close()
    logging.info('    indexed %d words' % lang_model.n_words)

    # filtering vocab
    # MIN_COUNT = 3
    # lang_model.trim(MIN_COUNT)

def extend_word_seq(lang, words, n_frames, start_time, end_time=None):
    frame_duration = (end_time - start_time) / n_frames
    extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
    # if self.remove_word_timing:
    #     n_words = 0
    #     for word in words:
    #         idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
    #         if idx < n_frames:
    #             n_words += 1
    #     space = int(n_frames / (n_words + 1))
    #     for i in range(n_words):
    #         idx = (i+1) * space
    #         extended_word_indices[idx] = lang.get_word_index(words[i][0])
    # else:
    # prev_idx = 0
    for word in words:
        idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
        if idx < n_frames:
            extended_word_indices[idx] = lang.get_word_index(word[0])
            # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
            # prev_idx = idx
    return torch.Tensor(extended_word_indices).long()


def fetch_raw_word(words, clip_s_t, skeleton_resampling_fps, frames, n_iter):
    ret_woprompt = []
    ret = []

    tmp_ = []
    prompt = "A person is talking: " #1
    # prompt = "is talking: " #1

    for cur_iter in range(n_iter):
        start_frame = cur_iter * frames
        end_frame =  (cur_iter+1)* frames
        s_t = clip_s_t + start_frame  /skeleton_resampling_fps
        e_t = clip_s_t + end_frame  /skeleton_resampling_fps
        for word in words:
            if word[1]> s_t and (word[2]< e_t or word[1]< e_t and word[2]>e_t):
                tmp_.append(word[0])

        text = ' '.join(tmp_)
        tmp_ = []
        ret_woprompt.append(text)
        text = prompt + '"' +text + '"'
        ret.append(text)

    # n_iter = ret[:n_iter]
    return ret, ret_woprompt


