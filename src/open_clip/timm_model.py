""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

try:
    import timm
    try:
        # new timm imports >= 0.8.1
        from timm.layers import RotAttentionPool2d
        from timm.layers import AttentionPool2d as AbsAttentionPool2d
        from timm.layers import Mlp, to_2tuple
    except ImportError as e:
        # fallback, try old timm imports < 0.8.1
        from timm.models.layers.attention_pool2d import RotAttentionPool2d
        from timm.models.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
        from timm.models.layers import Mlp, to_2tuple
except ImportError:
    timm = None

from .utils import freeze_batch_norm_2d


class TimmModel(nn.Module):
    """ timm model adapter
    """

    def __init__(
            self,
            model_name,
            embed_dim,
            image_size=224,
            pool='avg',
            proj='linear',
            proj_bias=False,
            drop=0.,
            drop_path=None,
            patch_drop=None,
            pretrained=False,
            return_tokens=True,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")
        self.image_size = to_2tuple(image_size)
        self.return_tokens = return_tokens

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None:
            timm_kwargs['patch_drop_rate'] = patch_drop
        
        self.trunk = timm.create_model(
                        model_name,
                        pretrained=pretrained,
                        **timm_kwargs,
                    )
    
        # 분류 헤드 제거 및 풀링 비활성화
        self.trunk.reset_classifier(0, global_pool='')
        
        # 모델의 특성 차원 가져오기
        self.feature_dim = self.trunk.num_features

        head_layers = OrderedDict()

        # 프로젝션 레이어만 설정 (풀링 제거)
        if proj == 'linear':
            head_layers['proj'] = nn.Linear(self.feature_dim, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(self.feature_dim, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))
        elif proj == 'none':
            pass  # 프로젝션 없음

        self.head = nn.Sequential(head_layers)
        self.embed_dim = embed_dim
        self.pool = pool  # 풀링 설정 저장


    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def forward(self, x):
        
        # 원본 이미지를 trunk에 통과시킴
        if hasattr(self.trunk, 'forward_features'):
            # ViT 모델의 경우 forward_features 메서드 사용
            features = self.trunk.forward_features(x)
        else:
            # 일반 모델의 경우 feature extractor 부분만 실행
            features = self.trunk.forward_stem(x)
            features = self.trunk.forward_tokens(features)
        
            # 토큰 시퀀스 준비
        tokens = features  # 원본 토큰 시퀀스 저장
        
        # 풀링된 피처 계산 (항상 계산)
        if self.pool == 'avg':
            # 평균 풀링
            pooled_features = features.mean(dim=1)  # [batch, feat_dim]
        elif self.pool == 'cls':
            # CLS 토큰 (첫 번째 토큰)
            pooled_features = features[:, 0]  # [batch, feat_dim]
        else:
            # 기본값: 평균 풀링
            pooled_features = features.mean(dim=1)  # [batch, feat_dim]
        
        # 프로젝션 적용 (풀링된 피처에)
        projected_features = self.head(pooled_features)  # [batch, embed_dim]
        
        # 토큰 시퀀스 반환 모드
        if self.return_tokens:
            # 토큰에도 프로젝션 적용 (각 토큰에 개별적으로)
            if len(self.head) > 0:
                # [batch, seq, feat_dim] -> [batch, seq, embed_dim]
                B, N, C = tokens.shape
                tokens_flat = tokens.view(-1, C)  # [batch*seq, feat_dim]
                tokens_projected = self.head(tokens_flat)  # [batch*seq, embed_dim]
                tokens_projected = tokens_projected.view(B, N, -1)  # [batch, seq, embed_dim]
                tokens = tokens_projected
            
            # 프로젝션된 피처와 토큰 시퀀스 모두 반환
            return projected_features, tokens  # ([batch, embed_dim], [batch, seq, embed_dim]) 형태로 반환
        
        # 기존 동작 (프로젝션된 피처만 반환)
        else:
            return projected_features  # [batch, embed_dim]

        
