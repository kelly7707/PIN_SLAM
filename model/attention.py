#!/usr/bin/env python3
# @file      attention.py
# @author    Jiawei Zhou

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad

from utils.config import Config


class Attention(nn.Module):
    '''MLP for key/value/query + multi-attention + MLP for single sdf prediction'''
    def __init__(self, config: Config, hidden_dim, hidden_level, out_dim, is_time_conditioned = False): 
        
        super().__init__()
    
        self.out_dim = out_dim #1

        bias_on = config.mlp_bias_on #True

        self.use_leaky_relu = False 

        self.num_bands = config.pos_encoding_band
        self.dimensionality = config.pos_input_dim # 3

        if config.use_gaussian_pe:
            position_dim = config.pos_input_dim + 2 * config.pos_encoding_band
        else: # default
            position_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1) #  3 = 3*(0+1)
            
        feature_dim = config.feature_dim # 8
        input_layer_count = feature_dim + position_dim # 8+3
        query_input_layer_dim = position_dim # 3
        
        if is_time_conditioned:
            input_layer_count += 1

        # ---- predict V/K/Q, assume they have the same output dim N (hidden_dim)
        # -- Shared MLP layers
        shared_layers = []
        kv_dim = hidden_dim #hidden_dim
        for i in range(hidden_level):
            if i == 0:
                shared_layers.extend([
                    nn.Linear(input_layer_count, kv_dim, bias_on),
                    nn.LayerNorm(kv_dim), # LN1
                    # nn.ReLU(inplace=True)
                    nn.Tanh()
                    # nn.GELU()
                ])
            else:
                shared_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim, bias_on),
                    # nn.LayerNorm(hidden_dim), 
                    nn.ReLU(inplace=True)
                ]) #TODO: different hidden_dim
        self.shared_layers = nn.Sequential(*shared_layers)

        # Separate linear layers for value, key, and query
        self.value_layer = nn.Linear(kv_dim, kv_dim, bias_on) #hidden_dim
        self.key_layer = nn.Linear(kv_dim, kv_dim, bias_on)

        # self.value_layer = nn.Sequential(
        #     nn.Linear(kv_dim, kv_dim, bias_on),
        #     nn.LayerNorm(kv_dim), # LN2
        #     nn.ReLU(inplace=True)
        #     # nn.Tanh()
        # )
        # self.key_layer = nn.Sequential(
        #     nn.Linear(kv_dim, kv_dim, bias_on),
        #     nn.LayerNorm(kv_dim), # LN3
        #     nn.ReLU(inplace=True)
        #     # nn.Tanh()
        # )

        # -- Query MLP layers
        q_layers = []
        q_dim = hidden_dim
        for i in range(hidden_level):
            if i == 0:
                q_layers.extend([
                    nn.Linear(query_input_layer_dim, q_dim, bias_on),
                    nn.LayerNorm(q_dim), # LN4
                    # nn.ReLU(inplace=True),
                    nn.Tanh()
                    # nn.GELU()
                ])
            # TODO: multiple layers
            
        self.query_layer = nn.Sequential(*q_layers)
        # # self.layers = nn.ModuleList(layers) # hidden_level = 1 (11, 64)
        # self.query_layer = nn.ModuleList(q_layers)


        # -- MHA
        embed_dim = q_dim #hidden_dim # q_dim 
        num_heads = 1 #4 # TODO: config & multiple heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, kdim=kv_dim, vdim=kv_dim, add_bias_kv=True) # ,dropout=0.2)
        
        # LayerNorm after multihead attention
        # self.attn_norm = nn.LayerNorm(embed_dim, elementwise_affine=False) # TODO LN5
        # # TODO: relu


        # -- output layer
        self.lout = nn.Linear(embed_dim, out_dim, bias_on) # (64, 1)

        if config.main_loss_type == 'bce': # default
            self.sdf_scale = config.logistic_gaussian_ratio*config.sigma_sigmoid_m
        else: # l1, l2 or zhong loss
            self.sdf_scale = 1.

        self.to(config.device)
        # torch.cuda.empty_cache()

    def forward(self, feature, query_features=None):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sdf(feature, query_features)
        return output

    # predict the sdf (opposite sign to the actual sdf)
    # unit is already m
    def sdf(self, features, query_features=None):
        assert features.shape[0] == query_features.shape[0]
        # query_features = F.layer_norm(query_features, [query_features.shape[1]])

        shared_output = self.shared_layers(features)
        value = self.value_layer(shared_output)
        key = self.key_layer(shared_output)

        # value = self.value_layer(features)
        # key = self.key_layer(features) #.clone().detach()

        query = self.query_layer(query_features.clone().detach()).unsqueeze(1)  #.clone().detach()
        
        attn_output = self.multihead_attn(query, key, value, need_weights=False) # ignore attn_output_weights

        # # Apply LayerNorm after multihead attention
        # attn_output = self.attn_norm(attn_output[0])
        # out = self.lout(attn_output).squeeze(1)

        out = self.lout(attn_output[0]).squeeze(1)
        out *= self.sdf_scale
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out
    
    

    # predict the occupancy probability
    def occupancy(self, features):
        out = torch.sigmoid(self.sdf(features)/-self.sdf_scale)  # to [0, 1]
        return out
