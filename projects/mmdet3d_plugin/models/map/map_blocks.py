from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

from ..blocks import linear_relu_ln


@POSITIONAL_ENCODING.register_module()
class SparsePoint3DEncoder(BaseModule):
    def __init__(
        self, 
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
    ):
        super(SparsePoint3DEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.input_dims = num_sample * coords_dim
        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(self.input_dims)

    def forward(self, anchor: torch.Tensor):
        pos_feat = self.pos_fc(anchor)  
        return pos_feat


@PLUGIN_LAYERS.register_module()
class SparsePoint3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
    ):
        super(SparsePoint3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.output_dim = num_sample * coords_dim
        self.num_cls = num_cls

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor, # [B, K, C]
        anchor: torch.Tensor, # [B, K, N_pt*2]
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        output = self.layers(instance_feature + anchor_embed) # [B, K, N_pt*2]
        output = output + anchor
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)  ## NOTE anchor embed?
        else:
            cls = None
        qt = None
        return output, cls, qt


@PLUGIN_LAYERS.register_module()
class SparsePoint3DKeyPointsGenerator(BaseModule): 
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        num_learnable_pts: int = 0, # 3
        fix_height: Tuple = (0,), # (0, 0.5, -0.5, 1, -1)
        ground_height: int = 0, # -1.84023
    ):
        super(SparsePoint3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        if self.num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, self.num_pts * 2)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor, # [B, K, N_pt*2]
        instance_feature=None, # [B, K, C]
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        assert self.num_learnable_pts > 0, 'No learnable pts'
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1) # [B, K, N_pt, 2]
        offset = (
            self.learnable_fc(instance_feature) # [B, K, N_pt*N_fixh*N_L*2], offset in xy plane
            .reshape(bs, num_anchor, self.num_sample, len(self.fix_height), self.num_learnable_pts, 2)
        )        
        key_points = offset + key_points[..., None, None, :] # [B, K, N_pt, N_fixh, N_L, 2]
        key_points = torch.cat(
            [
                key_points,
                key_points.new_full(key_points.shape[:-1]+(1,), fill_value=self.ground_height),
            ],
            dim=-1,
        ) # [B, K, N_pt, N_fixh, N_L, 3]
        fix_height = key_points.new_tensor(self.fix_height) # [N_fixh]
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:,None]], dim=-1) # [N_fixh, 3]
        key_points = key_points + height_offset[None, None, None, :, None]
        key_points = key_points.flatten(2, 4) # [B, K, N_pt*N_fixh*N_L, 3]
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        for i, t_time in enumerate(temp_timestamps):
            temp_key_points = key_points
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    # @staticmethod
    def anchor_projection(
        self,
        anchor, # [B, K_temp, N_pt*2]
        T_src2dst_list, # [[B,4,4], ]
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            dst_anchor = anchor.clone()
            bs, num_anchor, _ = anchor.shape
            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(1, 2) # [B, K_temp*N_pt, 2]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            ) # [B, 1, 4, 4]

            dst_anchor = (
                torch.matmul(
                    T_src2dst[..., :2, :2], dst_anchor[..., None]
                ).squeeze(dim=-1) # [B, K_temp*N_pt, 2]
                + T_src2dst[..., :2, 3] # not model 3d transformation
            ) # static object, not use velocity and time_intervals

            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(2, 3) # [B, K_temp, N_pt*2]
            dst_anchors.append(dst_anchor)
        return dst_anchors