import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import random


def parse_anstags(anstags):
    tag_list = []
    state = anstags[0]
    l = 1
    break_flag = False
    for tag in anstags[1:]:
        if "-" in tag:
            _, symb = tag.split('-')
            break_flag = True
        else:
            symb = tag

        if break_flag or state != symb:
            tag_list.append({"length": l, "symbol": state})
            state = symb
            l = 1
            break_flag = False
        else:
            l += 1
    tag_list.append({"length": l, "symbol": state})
    return tag_list


def pack_batch(batch_samples, device):
    sids, cont_list, tags_list = [], [], []
    for sid, content, tags in batch_samples:
        sids.append(sid)
        cont_list.append(content)
        tags_list.append(tags)
    l_list = [len(c) for c in cont_list]
    L = max(l_list)
    cont_tensor = torch.cat(
        [torch.LongTensor(c+[0]*(L-len(c))).view(-1, 1) for c in cont_list], 
        dim=1)
    tags_tensor = torch.cat(
        [torch.LongTensor(c+[0]*(L-len(c))).view(-1, 1) for c in tags_list],
        dim=1)
    return sids, cont_tensor.to(device), tags_tensor.to(device), l_list

def train(data, model, params, device='cuda:1'):
    model.to(device)
    model.train()
    loss = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for e in range(params.num_epoch):
        cum_lo = 0
        cum_acc = 0
        cum_conf = 0
        random.shuffle(data)
        for i in range(0, len(data), params.batch_size):
            batch_samples = data[i: i+params.batch_size]
            sids, cont_tensor, tags_tensor, l_list = pack_batch(batch_samples, device)
            opt.zero_grad()
            output = model(cont_tensor, tags_tensor,
                (1-i*params.batch_size/len(data) * 0.4 + 0.3 * (1-e/params.num_epoch)))
            logits = output[1:, :, :].permute(1, 2, 0)
            target = tags_tensor[1:, :].permute(1, 0)
            lo = loss(logits, target)
            lo.backward()

            loval = lo.cpu().item()
            cum_lo += loval

            for il, l in enumerate(l_list):
                anstags = logits[il:il+1, :, :l].max(1)[1]
                cum_acc += torch.mean((anstags==target[il, :l]).float()).cpu().item()

                conf = F.log_softmax(logits[il:il+1, :, :l], dim=1).max(1)[0]
                cum_conf += conf.cpu().mean().item()
                opt.step()
            print("step {} : loss {:.8f}, acc {:.8f}, conf {:.8f}".format(
                    i, cum_lo/(i+1), cum_acc/(i+1), cum_conf/(i+1)), 
                  end='\r')
        print("\nepoch # {} : loss {:.4f}: acc {:.4f}: conf {:.4f}".format(
            e, cum_lo/len(data), cum_acc/len(data), cum_conf/len(data)
            ))

# TODO: inference with data and model
def inf(data, model, params, device='cuda:1'):
    model.to(device)
    model.eval()
    state_dicts = []
    for i in range(0, len(data)):
        batch_samples = data[i: i+1]
        sids, cont_tensor, tags_tensor, l_list = pack_batch(batch_samples, device)
        print(sids, cont_tensor.shape)
        output = model.generate(cont_tensor)
        logits = output[1:, :, :].permute(1, 2, 0)
        anstags = logits.max(1)[1][0]
        tag_str = [model.tags.id2token[tagid] for tagid in anstags.cpu().numpy().tolist()]
        state_dicts.append(
            {'id': sids[0],
             'confidence': F.log_softmax(logits, dim=1).max(1)[0].mean().cpu().item(),
             'tags': parse_anstags(tag_str)}
        )
    return state_dicts
