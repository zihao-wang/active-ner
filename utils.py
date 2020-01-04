import re
import unicodedata
import os
import json
import pickle
from collections import OrderedDict
import numpy as np

SOS_token = 0
EOS_token = 1


class TokenMap:
    def __init__(self, vec_dim=300, init_token=None):
        """
        vocab_path for pickle file
        """
        self.vec_dim = vec_dim
        self.token2id = {"<SOS>": 0}
        self.id2token = ["<SOS>"]
        self.id2vec = [np.zeros((1, self.vec_dim))]
        self.add_token("<pad>")
        if init_token:
            for token in init_token:
                self.add_token(token)

    def __len__(self):
        return len(self.token2id)

    def add_token_seq(self, seq):
        if seq:
            return [self.add_token("<SOS>")] + [self.add_token(token) for token in seq]
        else:
            return []

    def add_token(self, token):
        if token not in self.token2id:
            new_id = len(self.id2token)
            self.token2id[token] = new_id
            self.id2token.append(token)
            self.id2vec.append(np.random.randn(1, self.vec_dim))
            return new_id
        else:
            return self.token2id[token]
    
    def load_token_map(self, load_dir):
        self.default_dir = load_dir
        if not (os.path.exists(load_dir) and 
                os.path.exists(os.path.join(load_dir, 'token2id.json')) and 
                os.path.exists(os.path.join(load_dir, 'id2token.json')) and 
                os.path.exists(os.path.join(load_dir, 'id2vec.npy'))):
            print("load failed, no load_dir found, exit function")
            return
        with open(os.path.join(load_dir, 'token2id.json'), mode='rt') as f:
            self.token2id = json.load(f)
        with open(os.path.join(load_dir, 'id2token.json'), mode='rt') as f:
            self.id2token = json.load(f)
        id2vec_mat = np.load(os.path.join(load_dir, 'id2vec.npy'))
        self.id2vec = np.split(id2vec_mat, len(self.id2token), axis=0)

    def save_token_map(self, save_dir=None):
        if save_dir is None:
            save_dir = self.default_dir
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'token2id.json'), mode='wt') as f:
            json.dump(self.token2id, f)
        with open(os.path.join(save_dir, 'id2token.json'), mode='wt') as f:
            json.dump(self.id2token, f)
        np.save(os.path.join(save_dir, 'id2vec.npy'), np.concatenate(self.id2vec, axis=0))

    def update_id2vec(self, emb_array):
        self.id2vec = np.split(emb_array, len(emb_array), axis=0)
