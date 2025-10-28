import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint

from typing import Optional, List
from torch import Tensor

from .transformer_decoder import TransformerDecoder, TransformerDecoderWoSelfAttenLayer

class TQN_Model(nn.Module):
    def __init__(self, 
            # embed_dim: int = 768, 
            # class_num: int = 2, 
            cfg = None
            ):
        super().__init__()
        embed_dim = cfg.MODEL.FUSION_DIM if cfg is not None else 512
        class_num = cfg.MODEL.FUSION_CLASS_NUM if cfg is not None else 1
        decoder_number_layer = cfg.MODEL.FUSION_DECODER_NUM if cfg is not None else 4

        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        decoder_layer = TransformerDecoderWoSelfAttenLayer(self.d_model, 4, 1024,
                                        0.1, 'relu',normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, decoder_number_layer, self.decoder_norm,
                                return_intermediate=False)
        self.dropout_feas = nn.Dropout(0.1)

        self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, class_num)
        )

        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(self, image_features, text_features, pos=None, return_atten = False, inside_repeat=True):
        #image_features (batch_size,patch_num,dim)
        #text_features (query_num,dim)
        batch_size = image_features.shape[0]
        image_features = image_features.transpose(0,1)  #(patch_num,batch_size,dim)
        if inside_repeat:
            text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1) # (query_num,batch_size,dim)
        image_features = self.decoder_norm(image_features)
        text_features = self.decoder_norm(text_features)
        features,atten_map = self.decoder(text_features, image_features, 
                memory_key_padding_mask=None, pos=pos, query_pos=None) 
        features = self.dropout_feas(features).transpose(0,1)  #b,embed_dim
        out = self.mlp_head(features)  #(batch_size, query_num)
        if return_atten:
            return out, atten_map
        else:
            return out


