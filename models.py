from __future__ import unicode_literals, print_function, division
from io import open
import string
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import TokenMap

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


emb_dim = 300
hid_dim = 32
dropout = 0.5

class Encoder(nn.Module):
    def __init__(self,
                 vocab,  # size of the encoding vocab
                 emb_dim: int=emb_dim,    # embedding size
                 enc_hid_dim: int=hid_dim,
                 dec_hid_dim: int=hid_dim,
                 dropout: float=dropout):
        super().__init__()
        input_dim = len(vocab)
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding.weights = torch.tensor(np.concatenate(vocab.id2vec, axis=0), dtype=float)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self,
                src: Tensor): # src: [seq_len, batch]
        embedded = self.dropout(self.embedding(src))    # embedded [seq_len, batch, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs: [seq_len, batch, nu_directions * enc_hid_dim]
        # hidden: [num_layers * num_directions, batch, enc_hid_dim]
        hidden = torch.tanh(
                    self.fc(    # [batch, dec_hid_dim]
                        torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))  # [batch, enc_hid_dim * 2]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int=hid_dim,
                 dec_hid_dim: int=hid_dim,
                 attn_dim: int=hid_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(
                    self.attn(
                        torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
    
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)
    

class Decoder(nn.Module):
    def __init__(self,
                 tags: int,
                 emb_dim: int = emb_dim,
                 enc_hid_dim: int = hid_dim,
                 dec_hid_dim: int = hid_dim,
                 dropout: int = dropout,
                 attention: nn.Module = None):
        super().__init__()
        output_dim = len(tags)
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.out.weight.data = torch.tensor(np.concatenate(tags.id2vec, axis=0), dtype=torch.float)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor):

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, vocab, tags):
        super().__init__()

        self.encoder = Encoder(vocab=vocab)
        self.decoder = Decoder(tags=tags, attention=Attention())
        self.vocab = vocab
        self.tags = tags
        self.device = device

    # annealing teacher forcing
    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs
    
    def get_new_emb(self):
        return self.encoder.embedding.weights.cpu().numpy(), self.decoder.out.weight.detach().cpu().numpy()
    

    def generate(self, src):
        batch_size = src.shape[1]
        max_len = src.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> token
        output = src[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            output = output.max(1)[1]

        return outputs
