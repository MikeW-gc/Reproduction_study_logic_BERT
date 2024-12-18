# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import evaluate as eva
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    Encoder is a multi-layer transformer with BiGRU
    """
    def __init__(self, encoder, logic, src_embed, pos_emb, dropout):
        super(Model, self).__init__()
        self.encoder = encoder
        self.logic = logic
        self.src_embed = src_embed
        self.pos_emb = pos_emb
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, label_map, entity_map, epoch, src, pos_ind, src_mask, train_ent, \
                entity=False, rel=False, train=False, train_rel=None):        
        x, out_chunk, out_rel = self.encode(label_map, entity_map, epoch, src, src_mask, train_ent, \
                                            pos_ind, entity, rel, train, train_rel)
        return out_chunk, out_rel
    
    def encode(self, label_map, entity_map, epoch, src, src_mask, train_ent, pos_ind, entity, rel, train, train_rel):
        X = torch.cat((self.dropout(self.src_embed(src)), self.pos_emb[pos_ind]), dim=2)
        return self.encoder(label_map, entity_map, epoch, X, src_mask, train_ent, entity, rel, train, train_rel)


        
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    
class Encoder(nn.Module):
    def __init__(self, layer, N, d_in, d_h):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.N = N
        self.norm = LayerNorm(layer.size)
        self.gru = nn.GRU(d_in, d_h, batch_first=True, dropout=0.1, bidirectional=True)
        
    def forward(self, label_map, entity_map, epoch, X, mask, train_ent, entity, rel, train, train_rel):
        x, _ = self.gru(X)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, out_chunk, out_rel = layer(label_map, entity_map, epoch, x, mask, train_ent, entity, rel, train, train_rel)
        return x, out_chunk, out_rel
        
        
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #return x + self.dropout(sublayer(self.norm(x)))
        return x + sublayer(x)
        
        
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, classifier, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.classifier = classifier
        self.size = size

    def forward(self, label_map, entity_map, epoch, x, mask, train_ent, entity, rel, train, train_rel):
        h = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        h = self.sublayer[1](h, self.feed_forward)
        attn = self.self_attn.attn
        h, out_chunk, out_relvec = self.classifier(label_map, entity_map, epoch, h, attn, mask, train_ent, entity, rel, train, train_rel)

        return h, out_chunk, out_relvec
        

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn
    
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        # d_k is the output dimension for each head
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
        
        
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
        
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, emb):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, _weight=emb)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)
                
        
class PositionalEncodingEmb(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncodingEmb, self).__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.position = nn.Parameter(torch.empty(max_len + 1, d_model))
        nn.init.xavier_uniform_(self.position)
        
    def forward(self, x):
        if x.size(1) < self.max_len:
            p = self.position[torch.arange(x.size(1))]
            p = p.unsqueeze(0)
            p = p.expand(x.size(0), p.size(1), p.size(2))
        else:
            i = torch.cat((torch.arange(self.max_len), -1 * torch.ones((x.size(1) - self.max_len), dtype=torch.long)))
            p = self.position[i]
            p = p.unsqueeze(0)
            p = p.expand(x.size(0), p.size(1), p.size(2))
        return self.dropout(x + p)


class Logic_module(nn.Module):
    def __init__(self, nhead):
        super(Logic_module, self).__init__()
        self.linear_con = nn.Linear(1, 1)
        self.linear_dis = nn.Linear(1, 1)
        self.nhead = nhead
        self.weights = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, 15)), requires_grad=True)
    
    def forward(self, label_map, label, output_chunk, output_rel, mask):
        mask = mask.squeeze()
        output_chunk = output_chunk.view(mask.size(0), mask.size(1), -1)
        idx_map = self.fact_node(output_chunk, label_map)
        loss = self.rule_node(label, output_chunk, output_rel, idx_map, mask)
        return loss
    # output has size batch_size x seq_length x nclass
    def fact_node(self, output_chunk, label_map):
        idx_map = {}
        # batch_size x seq_size
        idxs = torch.argmax(output_chunk, dim=2)
        for key in label_map.keys():
            # list with len=batch_size
            idx_map[key] = [torch.nonzero((i == key)).squeeze() for i in idxs]
        
        return idx_map
        
    def rule_node(self, label, output_chunk, output_rel, idx_map, mask):

        loss = 0.0
        loss_ent = 0.0
        count = 0
        weights = torch.sigmoid(self.weights)

        # relation rules
        for key in output_rel.keys():
            rel_vec = output_rel[key]
            rel = torch.argmax(rel_vec).item()
            (l, s, e, batch_idx) = key
            out_s = output_chunk[batch_idx][list(s)]
            out_e = output_chunk[batch_idx][list(e)]
            
            # rule 1: R(livein) & E(location) -> E(people) 
            # rule 2: R(livein) & E(people) -> E(location)
            if rel == label.index('Live_In-->'):
                if torch.argmax(out_e[0]) == 7:
                    loss += weights[0,0] * ((out_s[0, 3] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_e[0, 7] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
                if torch.argmax(out_s[0]) == 3:
                    loss += weights[0,1] * ((out_e[0, 7] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_s[0, 3] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
            elif rel == label.index('Live_In<--'):
                if torch.argmax(out_s[0]) == 7:
                    loss += weights[0,0] * ((out_e[0, 3] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_s[0, 7] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
                if torch.argmax(out_e[0]) == 3:
                    loss += weights[0,1] * ((out_s[0, 7] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_e[0, 3] - 2).view(-1, 1)))) ** 2)
                    count += 1
            
                
            # rule 3: R(orgbasedin) & E(loc) -> E(org) 
            # rule 4: R(orgbasedin) & E(org) -> E(loc)
            elif rel == label.index('OrgBased_In-->'):                
                if torch.argmax(out_e[0]) == 7:
                    loss += weights[0,2] * ((out_s[0, 5] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_e[0, 7] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
                if torch.argmax(out_s[0]) == 5:
                    loss += weights[0,3] * ((out_e[0, 7] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_s[0, 5] - 2).view(-1, 1)))) ** 2)
                    count += 1
                    
            elif rel == label.index('OrgBased_In<--'):                
                if torch.argmax(out_s[0]) == 7:
                    loss += weights[0,2] * ((out_e[0, 5] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_s[0, 7] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
                if torch.argmax(out_e[0]) == 5:
                    loss += weights[0,3] * ((out_s[0, 7] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_e[0, 5] - 2).view(-1, 1)))) ** 2)
                    count += 1
            
            # rule 5: R(locatedin) & E(loc) -> E(loc)
            elif rel == label.index('Located_In-->') or rel == label.index('Located_In<--'):                
                if torch.argmax(out_e[0]) == 7:
                    loss += weights[0,4] * ((out_s[0, 7] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_e[0, 7] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
                if torch.argmax(out_s[0]) == 7:
                    loss += weights[0,4] * ((out_e[0, 7] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_s[0, 7] - 2).view(-1, 1)))) ** 2)
                    count += 1
                                 
            
            # rule 6: R(kill) & E(per) -> E(per)
            elif rel == label.index('Kill-->') or rel == label.index('Kill<--'):                
                if torch.argmax(out_e[0]) == 3:
                    loss += weights[0,5] * ((out_s[0, 3] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_e[0, 3] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
                if torch.argmax(out_s[0]) == 3:
                    loss += weights[0,5] * ((out_e[0, 3] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_s[0, 3] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
            # rule 7: R(workfor) & E(org) -> E(per)
            # rule 8: R(workfor) & E(per) -> E(org)
            elif rel == label.index('Work_For-->'):                
                if torch.argmax(out_e[0]) == 5:
                    loss += weights[0,6] * ((out_s[0, 3] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_e[0, 5] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
                if torch.argmax(out_s[0]) == 3:
                    loss += weights[0,7] * ((out_e[0, 5] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_s[0, 3] - 2).view(-1, 1)))) ** 2)
                    count += 1
                    
            elif rel == label.index('Work_For<--'):                
                if torch.argmax(out_s[0]) == 5:
                    loss += weights[0,6] * ((out_e[0, 3] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_s[0, 5] - 2).view(-1, 1)))) ** 2)
                    count += 1
                
                if torch.argmax(out_e[0]) == 3:
                    loss += weights[0,7] * ((out_s[0, 5] -\
                        torch.sigmoid(self.linear_con((rel_vec[rel] + \
                        out_e[0, 3] - 2).view(-1, 1)))) ** 2)
                    count += 1


        if count > 0:
            loss = loss / count + loss_ent / mask.size(0)
        return loss  
            

# The decoder is also composed of a stack of N=6 identical layers.
class Classifier(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, sch_k, d_in, d_h, d_e, h, dchunk_out, drel_out, vocab, dropout):
        super(Classifier, self).__init__()
        self.sch_k = sch_k
        self.vocab = vocab
        self.d_in = d_in
        self.d_h = d_h
        self.h = h
        self.dchunk_out = dchunk_out
        self.lstm = nn.LSTM(3 * d_in, d_h, batch_first=True, dropout=0.5, bidirectional=True)
        self.gru = nn.GRU(3 * d_in, d_h, batch_first=True, dropout=0.1, bidirectional=True)
        #self.middle = nn.Linear(2 * d_h + d_e, d_h)
        self.chunk_out = nn.Linear(2 * d_h + d_e, dchunk_out)        
        self.softmax = nn.Softmax(dim=1)
        self.middle_rel = nn.Linear(d_h * 4 + d_e * 2 + h * 2, d_h)
        self.rel_out = nn.Linear(d_h, drel_out)
        self.drel_out = drel_out
        self.pad = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, d_in)), requires_grad=True)
        self.label_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(dchunk_out + 1, d_e)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, label_map, entity_map, epoch, X, attn, src_mask, train_ent, entity=False, rel=False, train=False, train_rel=None):
        
        # transformer with label GRU            
        X = self.dropout(X)    
        X_pad = torch.cat((self.pad.repeat(X.size(0),1,1), X, self.pad.repeat(X.size(0),1,1)), dim=1)
        l = X_pad[:, :-2, :]
        m = X_pad[:, 1:-1, :]
        r = X_pad[:, 2:, :]
        
        # transformer with Elman GRU
        output_h, _ = self.gru(torch.cat((l,m,r), dim=2))
        #output_h, _ = self.gru(X)
        # concatenate label embedding
        output_batch = torch.Tensor().to(device)
        output_h_ = output_h.permute(1,0,2)
        hi = torch.cat((output_h_[0,:,:], self.label_emb[-1].repeat(output_h_.size(1), 1)), dim=1)
        output_chunki = self.softmax(self.chunk_out(hi)).view(hi.size(0), 1, -1)
        output_batch = torch.cat((output_batch, output_chunki), dim=1)
        chunki = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1)
        chunk_batch = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1).view(-1, 1)
        for h in output_h_[1:,:,:]:
            hi = torch.cat((h, self.label_emb[chunki]), dim=1)
            output_chunki = self.softmax(self.chunk_out(hi)).view(hi.size(0), 1, -1)
            output_batch = torch.cat((output_batch, output_chunki), dim=1)
            chunki = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1)
            chunk_batch = torch.cat((chunk_batch, chunki.view(-1, 1)), dim=1)
        output_chunk = output_batch.view(-1, self.dchunk_out)
        
        if entity and not rel:
            return output_h, output_chunk
        else:
            # X has size batch_size x seq_length x emb_size
            # attn has size batch_size x nhead x seq_length x seq_length
            attn = attn.permute(0, 2, 3, 1)
            src_mask = src_mask.view(src_mask.size(0), src_mask.size(-1))
            out_rel = {}
            output_hl = torch.cat((output_h, self.label_emb[chunk_batch]), dim=2)
            out_batch = []
            for mask, out in zip(src_mask, output_batch):
                out = out[:mask.sum(), :]
                out_ind = torch.argmax(out, dim=1)
                out_batch.append(list(out_ind))
            # sample candidate entities    
            if train:
                sch_sample_ent = utils.schedule_sample(self.sch_k, out_batch, train_ent, epoch)
            else:
                sch_sample_ent = out_batch
            # collect candidate entity pairs
            if train:
                candi_ent_idxs, labels, idx2batch = utils.generate_candidate_entity_pair(sch_sample_ent, train_rel, label_map, entity_map, True)
            else:
                candi_ent_idxs, labels, idx2batch = utils.generate_candidate_entity_pair(sch_sample_ent, train_rel, label_map, entity_map, False)
                
            for idx, ent_pair, label in zip(range(len(candi_ent_idxs)), candi_ent_idxs, labels):
                (s, e) = ent_pair
                batch_idx = idx2batch[idx]   
                # first entity
                data_row1 = output_hl[batch_idx][list(s)].view(-1, output_hl.size(-1))
                data_row1 = torch.mean(data_row1, dim=0)
                # second entity                                        
                data_row2 = output_hl[batch_idx][list(e)].view(-1, output_hl.size(-1))
                data_row2 = torch.mean(data_row2, dim=0)
                            
                att1 = torch.cat(([attn[batch_idx, ind1, ind2, :] for ind1 in s \
                                for ind2 in e]), dim=0).view(-1, attn.size(-1))
                att2 = torch.cat(([attn[batch_idx, ind2, ind1, :] for ind2 in e \
                                for ind1 in s]), dim=0).view(-1, attn.size(-1))                                
                att1 = torch.mean(att1, dim=0)
                att2 = torch.mean(att2, dim=0)
                            
                rel_input = torch.cat((data_row1, data_row2, att1, att2)).view(1, -1)
                rel_input = self.dropout(F.tanh(self.middle_rel(rel_input)))
                rel_vec = self.softmax(self.rel_out(rel_input))[0]
                out_rel[(label, s, e, batch_idx)] = rel_vec 
                
            return output_h, output_chunk, out_rel
        
    

# full model
def make_model(vocab, pos, emb, sch_k=1.0, N=2, d_in=350, d_h=100, d_e=25, d_p=50, dchunk_out=9,
               drel_out=11, d_model=200, d_emb=300, d_ff=200, h=10, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(pos, d_p)), requires_grad=True)
    position = PositionalEncodingEmb(d_emb, dropout)
    logic_module = Logic_module(nhead=10)
    model = Model(
        Encoder(EncoderLayer(Classifier(sch_k, d_model, d_h, d_e, h, dchunk_out, \
                    drel_out, vocab, 0.1), d_model, c(attn), c(ff), dropout), N, 350, 100),
        logic_module,
        nn.Sequential(Embeddings(d_emb, vocab, emb), c(position)),
        pos_emb, dropout)
    
    return model
    


# class Batch:
#     "Object for holding a batch of data with mask during training."
#     def __init__(self, src, pos, trg=None, trg_rel=None, pad=0):
#         self.src = src
#         self.src_mask = (src != pad).unsqueeze(-2)
#         self.trg = trg
#         self.trg_rel = trg_rel
#         self.pos = pos
#         self.ntokens = torch.tensor(self.src != pad, dtype=torch.float).data.sum()

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, pos, trg=None, trg_rel=None, pad=0):
        self.src = src.to(device)
        self.pos = pos.to(device)
        self.src_mask = (self.src != pad).unsqueeze(-2).to(device)
        self.trg = trg.to(device) if trg is not None else None
        # If trg_rel is a list of tuples, it might not be a tensor. If you need it on GPU,
        # you’ll have to handle that conversion similarly. For now, if it's just used as is, leave it.
        self.trg_rel = trg_rel  
        self.ntokens = (self.src != pad).float().sum().to(device)

    

def run_logic_epoch(label_map, entity_map, label_rel, epoch, data_iter, model, loss_compute, entity=False, rel=False, train=False):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        out_chunk, out_rel = model.forward(label_map, entity_map, epoch, batch.src.to(device), batch.pos.to(device), \
                batch.src_mask.to(device), batch.trg.to(device), entity, rel, train, batch.trg_rel)
        loss_logic = 0
        loss = loss_compute(out_chunk, out_rel, loss_logic, batch.trg.to(device), \
            batch.trg_rel, batch.ntokens.to(device), batch.src_mask.to(device))    
        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                (i, loss, tokens / elapsed))
        start = time.time()
        tokens = 0
            
    return total_loss

def predict(label_map, entity_map, data_iter, model, entity=False, rel=False):
    labels_all = []
    labels_rel_all = []
    train=False
    
    with torch.no_grad():        
        if entity and not rel:
            for batch in data_iter:
                out = model.forward(label_map, entity_map, 0, batch.src.to(device), batch.pos.to(device), \
                        batch.src_mask.to(device), batch.trg.to(device), entity, rel)
                label = torch.argmax(out, dim=1)
                labels_all.append(label)
            return labels_all
            
        else:
            for batch in data_iter:
                out_chunk, out_rel = model.forward(label_map, entity_map, 0, batch.src.to(device), batch.pos.to(device), \
                    batch.src_mask.to(device), batch.trg.to(device), entity, rel, train, batch.trg_rel)
                label = torch.argmax(out_chunk, dim=1)
                labels_all.append(label)
                
                for key in out_rel.keys():
                    out_rel[key] = torch.argmax(out_rel[key]).item()
                labels_rel_all.append(out_rel)
            
            return labels_all, labels_rel_all    
    
                
def data_batch(idxs_src, labels_src, labels_rel_src, pos_src, batchlen):
    assert len(idxs_src) == len(labels_src)
    batches = [idxs_src[x : x + batchlen] for x in range(0, len(idxs_src), batchlen)]
    label_batches = [labels_src[x : x + batchlen] for x in range(0, len(labels_src), batchlen)]    
    label_rel_batches = [labels_rel_src[x : x + batchlen] for x in range(0, len(labels_rel_src), batchlen)]
    pos_batches = [pos_src[x : x + batchlen] for x in range(0, len(idxs_src), batchlen)]
    for batch, label, label_rel, pos in zip(batches, label_batches, label_rel_batches, pos_batches):
        # compute length of longest sentence in batch
        batch_max_len = max([len(s) for s in batch])
        # prepare a numpy array with the data, initializing the data with 'PAD' 
        # and all labels with -1; initializing labels to -1 differentiates tokens 
        # with tags from 'PAD' tokens
        batch_data = 0 * np.ones((len(batch), batch_max_len))
        batch_labels = 0 * np.ones((len(batch), batch_max_len))
        batch_rel_labels = label_rel
        batch_pos = 0 * np.ones((len(batch), batch_max_len))

        # copy the data to the numpy array
        for j in range(len(batch)):
            cur_len = len(batch[j])
            batch_data[j][:cur_len] = batch[j]
            batch_labels[j][:cur_len] = label[j]
            batch_pos[j][:cur_len] = pos[j]
        # since all data are indices, we convert them to torch LongTensors        
        batch_data, batch_labels, batch_pos = torch.LongTensor(batch_data), \
            torch.LongTensor(batch_labels), torch.LongTensor(batch_pos)
        # convert Tensors to Variables
        yield Batch(batch_data, batch_pos, batch_labels, batch_rel_labels, 0)
                


        
class logicLoss:
    def __init__(self, opt=None):
        self.opt = opt
    def __call__(self, x, x_rel, logic_loss, y, y_rel, norm, mask):
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.log(torch.gather(x.contiguous(), dim=1, index=y.contiguous().view(-1, 1)))
        # losses: (batch, max_len)
        losses = losses_flat.view(*y.size())
        # mask: (batch, max_len)
        mask = mask.squeeze()
        losses_rel = 0.0
        
        for row_x in x_rel.keys():    
            (true_label, s, e, b_idx) = row_x
            rel_vec = x_rel[row_x]
            losses_rel += -torch.log(rel_vec[true_label])

        norm = mask.float().sum()

        loss = (((losses) * mask.float()).sum() + losses_rel) / norm + 0.5 * logic_loss
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss
        


criterion = nn.NLLLoss()
emb = pickle.load(open("data/embedding300_glove", "rb"))


def save_checkpoint(model, optimizer, epoch, path="checkpoints/checkpoint.pth"):
    """
    Saves the model and optimizer state along with the current epoch.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")


def load_checkpoint(model, optimizer, path="checkpoints/checkpoint.pth"):
    """
    Optionally load a checkpoint before training (if resuming training).
    """
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {path}, starting at epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0

os.makedirs("checkpoints", exist_ok=True)

idxs_train = pickle.load(open("data/idx.train", "rb"))
idxs_test = pickle.load(open("data/idx.test", "rb"))
labels_train, labels_type_train = pickle.load(open("data/labels_chunk.train", "rb"))
labels_test, labels_type_test = pickle.load(open("data/labels_chunk.test", "rb"))
labels_rel_train = pickle.load(open("data/labels_2rel.train", "rb"))\

for i in range(min(5, len(labels_rel_train))):
    print(f"Print statement: Sentence {i} relations: {labels_rel_train[i]}")

labels_rel_test = pickle.load(open("data/labels_2rel.test", "rb"))
pos_train, pos_test = pickle.load(open("data/pos.tag", "rb"))
pos_dic = ['pad']
posind_train, posind_test = [], []
for pos_line in pos_train:
    pos_ind = []
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)
        pos_ind.append(pos_dic.index(pos))
    posind_train.append(pos_ind)
for pos_line in pos_test:
    pos_ind = []
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)
        pos_ind.append(pos_dic.index(pos))
    posind_test.append(pos_ind)

emb = torch.tensor(emb, dtype=torch.float).to(device)
model = make_model(emb.shape[0], len(pos_dic), emb, sch_k=1.0, N=2)
model = model.to(device)

f_out = open("result/transformer-without-logic.txt", "w")

optimizer = optim.Adadelta(model.parameters())

label_map = {0:'O', 1:'B-Other', 2:'I-Other', 3:'B-Peop', 4:'I-Peop', \
        5:'B-Org', 6:'I-Org', 7:'B-Loc', 8:'I-Loc'}
entity_map = {'Other':0, 'Peop':1, 'Org':2, 'Loc':3}
label_rel_map = {0:'O', 1:'Live_In<--', 2:'OrgBased_In-->', \
    3:'Located_In-->', 4:'OrgBased_In<--', 5:'Located_In<--', \
    6:'Live_In-->', 7:'Work_For-->', 8:'Work_For<--', 9:'Kill-->', 10:'Kill<--'}
label_rel = ['O', 'Live_In<--', 'OrgBased_In-->', \
'Located_In-->', 'OrgBased_In<--', 'Located_In<--', \
'Live_In-->', 'Work_For-->', 'Work_For<--', 'Kill-->', 'Kill<--']

rel_map = {'Live_In':0, 'OrgBased_In':1, 'Located_In':2, 'Work_For':3, \
                    'Kill':4}
rel_names = ['Live_In', 'OrgBased_In', 'Located_In', 'Work_For', 'Kill']


start_epoch = load_checkpoint(model, optimizer, path="checkpoints/checkpoint.pth")
num_epochs = 150

for epoch in range(start_epoch, num_epochs):    
    model.train()
    loss = run_logic_epoch(label_map, entity_map, label_rel, epoch, \
        data_batch(idxs_train, labels_train, labels_rel_train, posind_train, 25), model, 
        logicLoss(optimizer), entity=True, rel=True, train=True)
    
    # save a checkpoint
    save_checkpoint(model, optimizer, epoch, path=f"checkpoints/checkpoint_epoch_{epoch}.pth")
    
    model.eval()
    labels_predict, labels_rel_predict = predict(label_map, entity_map, data_batch(idxs_test, labels_test, labels_rel_test, \
                posind_test, 1), model, entity=True, rel=True)
        
    labels_test_map = [label_map[item] for sub in labels_test for item in sub]
    labels_predict_map = [label_map[t.item()] for sub in labels_predict for t in sub]

    print(epoch)  
    print(eva.f1_score(labels_test_map, labels_predict_map))
    report = eva.classification_report(labels_test_map, labels_predict_map)
    p, r, f1, report_rel = eva.report_relation_trec(labels_rel_test, labels_rel_predict, 6)
    print(report)
    print(report_rel)
    print(loss)
    f_out.write("epoch: " + str(epoch) + "\n")
    f_out.write("performance on entity extraction:" + "\n")
    f_out.write("precision: " + str(eva.precision_score(labels_test_map, labels_predict_map)))
    f_out.write("\t" + "recall: " + str(eva.recall_score(labels_test_map, labels_predict_map)))
    f_out.write("\t" + "f1: " + str(eva.f1_score(labels_test_map, labels_predict_map)))
    f_out.write("\n" + report)
    f_out.write("\n")
    
    f_out.write("performance on relation extraction:" + "\n")
    f_out.write("precision: " + str(p))
    f_out.write("\t" + "recall: " + str(r))
    f_out.write("\t" + "f1: " + str(f1))

    f_out.write('\n' + report_rel)
    f_out.write("\n")  

f_out.close()
 
    
