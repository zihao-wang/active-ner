import argparse
import os
import json

import numpy as np
import torch

from utils import TokenMap
from models import Encoder, Attention, Decoder, Seq2Seq
from routine import train, inf

parser = argparse.ArgumentParser()
parser.add_argument("--inf", default="in.data")
parser.add_argument("--outf", default="out.data.new")
parser.add_argument("--logpfx", default="log")
parser.add_argument("--num_epoch", default=1)
parser.add_argument("--batch_size", default=8)
parser.add_argument("--model_save_dir", default="saved_models")
parser.add_argument("--model_load_dir", default="saved_models")

"""input data sample
{
    "id":"1",
    "content":",患者2015.11.27因“卵巢癌”于我院全麻上行经腹子宫+两侧附件切除+大网膜切除+部分肠壁转移病灶切除术。",
    "tags":[
        {"_id":"5dd3f9e63c561e02aa0080bd","length":15,"symbol":"O"},
        {"_id":"5dd3f9e63c561e02aa0080bc","length":3,"symbol":"A"},
        {"_id":"5dd3f9e63c561e02aa0080bb","length":8,"symbol":"O"},
        {"_id":"5dd3f9e63c561e02aa0080ba","length":29,"symbol":"E"},
        {"_id":"5dd3f9e63c561e02aa0080b9","length":1,"symbol":"O"}
    ]
}
"""

device=torch.device('cuda:0')


# TODO: handle the sample parser
def load_from_infile(infile, wordmap, tagmap):
    with open(infile, mode='rt') as f:
        data = [parse_sample(raw_sample, wordmap, tagmap) for raw_sample in json.load(f)]
    return data


def save_to_outdir(outdir, data_dict):
    os.makedirs(outdir, exist_ok=True)
    for fn, data in data_dict.items():
        with open(os.path.join(outdir, fn), mode='wt') as f:
            json.dump(data, f)


def parse_sample(data_sample, word_map: TokenMap, tag_map: TokenMap):
    sample_id = data_sample['id']
    content = word_map.add_token_seq(data_sample['content'])
    tag_tmp = []
    # TODO: this may not be suitable for tags with only one length
    for tag in data_sample.get('tags', []):
        s = tag['symbol']
        if s == 'O':
            tag_tmp += tag["length"] * ['O']
        else:
            tag_tmp += ['b-' + s] + (tag["length"]-1) * ['i-' + s]
    tags = tag_map.add_token_seq(tag_tmp)

    assert len(tags) == 0 or len(tags) == len(content)
    return [sample_id, content, tags]



"""output data sample
{
    "id": "5114", 
    "confidence": 0, 
    "tags": [
        {"length": 8, "symbol": "O"}, 
        {"length": 3, "symbol": "A"}, 
        {"length": 10, "symbol": "O"}, 
        {"length": 14, "symbol": "E"}, 
        {"length": 36, "symbol": "O"}]}
    ]
"""

# TODO: parser the predicted tags into the given format



def analysis_data(data):
    length_list = [len(content) for _, content,  _ in data]
    length_dist = np.quantile(length_list, np.linspace(0, 1, 100))
    print(length_dist)


def split_data(data):
    train_data, inf_data = [], []    
    for sid, content, tags in data:
        if tags:
            train_data.append([sid, content, tags])
        else:
            inf_data.append([sid, content, [-1]])
    return train_data, inf_data


if __name__ == "__main__":
    # parse args
    params = parser.parse_args()
    print("1 argument parsed")
    ######################### prepare model and data #########################
    # init map and model
    word_map = TokenMap(vec_dim=300)
    tag_map = TokenMap(vec_dim=396)
    print("2 meta init")
    # if there is saved meta, then load
    meta_path = os.path.join(params.model_load_dir, 'meta')
    if meta_path and os.path.exists(meta_path):
        print("2 load saved meta")
        word_map.load_token_map(os.path.join(meta_path, 'word'))
        tag_map.load_token_map(os.path.join(meta_path, 'tag'))
    else:
        print("2 no saved meta to load")
    # load & parse data
    data = load_from_infile(params.inf, word_map, tag_map)
    train_data, inf_data = split_data(data)
    print(train_data)
    print(inf_data)
    print("word map size:", len(word_map))
    print("tag map size:", len(tag_map))
    print("3 data prepared")
    model = Seq2Seq(vocab=word_map, tags=tag_map)
    print("4 model init")
    # if there is saved model, then load
    if params.model_load_dir and os.path.exists(os.path.join(params.model_save_dir, 'model.pt')):
        print("4 load saved model")
        model.load_state_dict(torch.load(os.path.join(params.model_load_dir, 'model.pt')))
    else:
        print("4 no saved model to load")
    ######################### prepare model and data #########################

    ######################### train the model #########################
    # TODO:
    if train_data:
        train(data=train_data, model=model, params=params)
    ######################### train the model #########################

    ######################### save model & meta #########################
    word_emb, tag_emb = model.get_new_emb()
    word_map.update_id2vec(word_emb)
    tag_map.update_id2vec(tag_emb)
    torch.save(model.state_dict(), os.path.join(params.model_save_dir, 'model.pt'))
    word_map.save_token_map(os.path.join(params.model_save_dir, 'meta', 'word'))
    tag_map.save_token_map(os.path.join(params.model_save_dir, 'meta', 'tag'))
    ######################### save model & meta #########################

    ######################### inference #########################
    # TODO:
    data_dicts = {}
    if train_data:
        data_dicts['train.out'] = inf(data=train_data, model=model, params=params)
    if inf_data:
        data_dicts['inf.out'] = inf(data=inf_data, model=model, params=params)
    print(data_dicts)
    save_to_outdir('./', data_dicts)
    ######################### inference #########################
