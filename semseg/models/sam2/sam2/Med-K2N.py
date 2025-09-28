# -*- coding: utf-8 -*-
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors.torch import save_file
from icecream import ic
from modeling.sam2_base import SAM2Base
import torch.nn.init as init
import random
from mamba_ssm import Mamba
import pandas as pd
import numpy as np

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

import warnings
from collections import defaultdict

MEDK2N_CONFIG = {
    'num_modalities': 4,
    'embed_dim': 64,
    'feat_dim': 256,
    'quality_threshold': 0.1,
    'enable_quality_feedback': True,
    'progressive_fusion': True,
    'max_memory_history': 8,
    'simplified_architecture': True,
    'enable_causal_constraints': True
}

MEDK2N_MODULE_SPECS = {
    'PreweightNet': {'params': '400K'},
    'ThresholdNet': {'params': '60K'},
    'EffeWeightNet': {'params': '120K'},
    'TaskHeadNet': {'params': '5M'},
    'CausalIdentityModule': {'params': '800K'},
}

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0

        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class BottleneckCNN(nn.Module):
    def __init__(self, config):
        super(BottleneckCNN, self).__init__()
        self.config = config
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = "reflect"

        model = [
            ResnetBlock(
                288,
                padding_type=padding_type,
                norm_layer=norm_layer,
                use_dropout=False,
                use_bias=use_bias,
            )
        ]
        self.residual_cnn = nn.Sequential(*model)

    def forward(self, x):
        return self.residual_cnn(x)

class CrossFrameFusion(nn.Module):
    def __init__(self, dim=288, num_frames=4, fusion_type='attention'):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.fusion_type = fusion_type

        if fusion_type == 'attention':
            self.temporal_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.Dropout(0.1)
            )
        elif fusion_type == 'conv3d':
            self.conv3d = nn.Conv3d(dim, dim, kernel_size=(3,3,3), padding=(1,1,1))
            self.norm = nn.BatchNorm3d(dim)
        else:
            self.fusion_type = 'mean'

    def forward(self, frame_features):
        if len(frame_features) == 1:
            return frame_features[0]

        if self.fusion_type == 'mean':
            return torch.stack(frame_features, dim=0).mean(dim=0)

        elif self.fusion_type == 'attention':
            B, C, H, W = frame_features[0].shape
            seq_features = []
            for feat in frame_features:
                seq_features.append(feat.permute(0, 2, 3, 1).reshape(B, H*W, C))
            seq_input = torch.cat(seq_features, dim=1)

            normed = self.norm1(seq_input)
            attn_out, _ = self.temporal_attn(normed, normed, normed)
            attn_out = attn_out + seq_input

            normed2 = self.norm2(attn_out)
            ffn_out = self.ffn(normed2)
            output = ffn_out + attn_out

            output = output.view(B, len(frame_features), H, W, C)
            output = output.mean(dim=1)
            output = output.permute(0, 3, 1, 2)
            return output

        elif self.fusion_type == 'conv3d':
            stacked = torch.stack(frame_features, dim=2)
            fused = self.conv3d(stacked)
            fused = self.norm(fused)
            fused = F.relu(fused)
            fused = fused.mean(dim=2)
            return fused

class MultiHeadGenerator(nn.Module):
    def __init__(self, input_dim=256, output_frames=4, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_frames = output_frames
        self.num_heads = num_heads

        self.shared_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.encoder_residual_proj = nn.Conv2d(64, 256, kernel_size=1)

        self.shared_upsampler = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.modality_heads = nn.ModuleList()
        class SEModule(nn.Module):
            def __init__(self, channels, reduction=16):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channels, channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels // reduction, channels, bias=False),
                    nn.Sigmoid()
                )
            def forward(self, x):
                b, c, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1)
                return x * y

        for i in range(output_frames):
            class ModalityHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm2d(32)
                    self.act1 = nn.ReLU(inplace=True)
                    self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm2d(32)
                    self.act2 = nn.ReLU(inplace=True)
                    self.se2 = SEModule(32)
                    self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
                    self.bn3 = nn.BatchNorm2d(16)
                    self.act3 = nn.ReLU(inplace=True)
                    self.conv4 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
                def forward(self, x):
                    out1 = self.act1(self.bn1(self.conv1(x)))
                    out2 = self.act2(self.bn2(self.conv2(out1)))
                    out2_se = self.se2(out2)
                    out3 = self.act3(self.bn3(self.conv3(out2_se + out1)))
                    out4 = torch.sigmoid(self.conv4(out3))
                    return out4
            self.modality_heads.append(ModalityHead())

        self.temporal_refiner = nn.Sequential(
            nn.Conv2d(3 * output_frames, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3 * output_frames, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.input_adapters = nn.ModuleDict({
            '32': self._create_adapter(32),
            '64': self._create_adapter(64),
            '128': self._create_adapter(128),
            '256': self._create_adapter(256),
        })
        self.fallback_adapter = None

    def _create_adapter(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

    def forward(self, x, head_indices: Optional[list] = None):
        input_channels = x.shape[1]
        adapter_key = str(input_channels)

        if adapter_key in self.input_adapters:
            adapted_features = self.input_adapters[adapter_key](x)
        else:
            if self.fallback_adapter is None or self.fallback_adapter[0].in_channels != input_channels:
                self.fallback_adapter = self._create_adapter(input_channels).to(x.device)
            adapted_features = self.fallback_adapter(x)

        encoded_features = self.shared_encoder(adapted_features)
        res = self.encoder_residual_proj(adapted_features)
        encoded_features = encoded_features + res
        encoded_features = F.layer_norm(encoded_features, encoded_features.shape[1:])

        upsampled_features = self.shared_upsampler(encoded_features)
        upsampled_features = F.layer_norm(upsampled_features, upsampled_features.shape[1:])

        generated_frames = []
        if head_indices is None:
            heads_to_run = enumerate(self.modality_heads)
        else:
            valid_idx = [i for i in head_indices if 0 <= int(i) < len(self.modality_heads)]
            heads_to_run = ((i, self.modality_heads[i]) for i in valid_idx)

        for _, head in heads_to_run:
            frame = head(upsampled_features)
            generated_frames.append(frame)

        if len(generated_frames) > 1:
            concatenated = torch.cat(generated_frames, dim=1)
            refined_concat = self.temporal_refiner(concatenated)

            refined_frames = []
            for i in range(len(generated_frames)):
                start_ch = i * 3
                end_ch = (i + 1) * 3
                refined_frames.append(refined_concat[:, start_ch:end_ch, :, :])
            return refined_frames

        return generated_frames

def apply_lora_to_sam2_encoder(model, target_modules=None, r=16, lora_alpha=32, lora_dropout=0.1):
    if not PEFT_AVAILABLE:
        return model

    if target_modules is None:
        target_modules = [
            "layers.0", "layers.1", "layers.2", "layers.3",
            "attn.qkv", "attn.q_proj", "attn.k_proj", "attn.v_proj",
            "mlp.lin1", "mlp.lin2",
            "proj"
        ]

    peft_config = LoraConfig(
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    if hasattr(model, 'image_encoder'):
        peft_wrapped = get_peft_model(model.image_encoder, peft_config)
        underlying = getattr(peft_wrapped, 'base_model', peft_wrapped)
        class _PEFTInputAdapter(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def forward(self, *args, **kwargs):
                if 'input_ids' in kwargs and (args is None or len(args) == 0):
                    x = kwargs.pop('input_ids')
                    return self.wrapped(x, *args, **kwargs)
                return self.wrapped(*args, **kwargs)
        model.image_encoder = _PEFTInputAdapter(underlying)
    else:
        pass

    return model

class MedK2N_PreweightNet(nn.Module):
    def __init__(self, feat_dim=256, embed_dim=128, num_modalities=4, base_channels=256, 
                 enhance_channels=256, enable_history_tracking=True, enable_compatibility_assessment=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        self.multi_scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dim, 256, kernel_size=3, padding=d, dilation=d),
                nn.GroupNorm(32, 256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            ) for d in [1, 2, 4, 8]
        ])

        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )

        self.quality_memory = nn.LSTM(
            input_size=32, hidden_size=64, num_layers=2, 
            batch_first=True, dropout=0.1, bidirectional=True
        )

        self.compatibility_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.weight_predictor = nn.Sequential(
            nn.Linear(256*4 + 128 + 1 + embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, base_features, enhance_features, prev_outputs, 
                quality_history, task_compatibility, modality_embed):
        B = base_features.size(0)
        device = base_features.device

        base_multiscale = []
        enhance_multiscale = []

        for extractor in self.multi_scale_extractors:
            base_feat = extractor(base_features).flatten(1)
            enhance_feat = extractor(enhance_features).flatten(1)
            base_multiscale.append(base_feat)
            enhance_multiscale.append(enhance_feat)

        base_stack = torch.stack(base_multiscale, dim=1)
        enhance_stack = torch.stack(enhance_multiscale, dim=1)

        base_enhanced, _ = self.cross_modal_attn(
            query=base_stack, key=enhance_stack, value=enhance_stack
        )
        enhance_enhanced, _ = self.cross_modal_attn(
            query=enhance_stack, key=base_stack, value=base_stack  
        )

        base_global = base_enhanced.mean(dim=1)
        enhance_global = enhance_enhanced.mean(dim=1)

        if quality_history.dim() == 2:
            quality_history = quality_history.unsqueeze(1)
        quality_encoded, _ = self.quality_memory(quality_history)
        quality_feat = quality_encoded.mean(dim=1)

        compat_input = torch.cat([modality_embed, modality_embed], dim=-1)
        compatibility = self.compatibility_net(compat_input)

        combined_features = torch.cat([
            *base_multiscale, quality_feat, compatibility, modality_embed
        ], dim=1)
        weight = self.weight_predictor(combined_features)
        return weight

class MedK2N_ThresholdNet(nn.Module):
    def __init__(self, embed_dim=128, modality_dim=128, task_dim=128, 
                 compatibility_dim=64, performance_history_dim=32):
        super().__init__()

        self.context_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim*2, nhead=8, dim_feedforward=512, 
            dropout=0.1, activation='gelu', batch_first=True
        )

        self.performance_analyzer = nn.LSTM(
            input_size=32, hidden_size=64, num_layers=2,
            batch_first=True, dropout=0.1, bidirectional=True
        )

        self.threshold_predictor = nn.Sequential(
            nn.Linear(embed_dim*2 + 128 + 128 + 1 + 1, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(), 
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.adaptive_scaler = nn.Sequential(
            nn.Linear(embed_dim*2, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(), 
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, mod_embed, task_embed, compat_matrix, 
                perf_tracker, context_embedding, current_weight, quality_gain):
        B = mod_embed.size(0)

        context_input = torch.cat([mod_embed, task_embed], dim=-1).unsqueeze(1)
        context_enhanced = self.context_transformer(context_input).squeeze(1)

        if perf_tracker.dim() == 2:
            perf_tracker = perf_tracker.unsqueeze(1)
        perf_encoded, _ = self.performance_analyzer(perf_tracker)
        perf_feat = perf_encoded.mean(dim=1)

        combined = torch.cat([
            context_enhanced, perf_feat, context_embedding, 
            current_weight, quality_gain
        ], dim=1)

        base_threshold = torch.sigmoid(self.threshold_predictor(combined))

        scale_factor = self.adaptive_scaler(context_enhanced)
        adaptive_threshold = base_threshold * scale_factor

        final_threshold = torch.clamp(adaptive_threshold, 0.05, 0.95)

        return final_threshold

class MedK2N_EffeWeightNet(nn.Module):
    def __init__(self, embed_dim=128, base_channels=256, aux_channels=256, 
                 modality_embed_dim=128, task_embed_dim=128, spatial_resolution=(64, 64)):
        super().__init__()

        self.base_processor = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

        self.threshold_processor = nn.Sequential(
            nn.Linear(1, 64), 
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

        self.context_processor = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

        self.history_processor = nn.Sequential(
            nn.Linear(16, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

        self.uncertainty_processor = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )

        self.quality_processor = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )

        self.intelligent_fusion = nn.Sequential(
            nn.Linear(32+32+64+32+16+16, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.confidence_estimator = nn.Sequential(
            nn.Linear(32+32+64+32+16+16, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.weight_regulator = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(), 
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, w_global, tau, task_context, modality_context,
                hist_performance, uncertainty_score, quality_indicator):
        base_feat = self.base_processor(w_global)
        thresh_feat = self.threshold_processor(tau)
        context_feat = self.context_processor(torch.cat([task_context, modality_context], dim=1))
        hist_feat = self.history_processor(hist_performance)
        uncert_feat = self.uncertainty_processor(uncertainty_score)
        qual_feat = self.quality_processor(quality_indicator)

        all_features = torch.cat([
            base_feat, thresh_feat, context_feat,
            hist_feat, uncert_feat, qual_feat
        ], dim=1)

        raw_weight = self.intelligent_fusion(all_features)
        effective_weight = torch.sigmoid(raw_weight)

        confidence = self.confidence_estimator(all_features)

        regulation_input = torch.cat([w_global, tau, effective_weight], dim=1)
        regulation_factor = self.weight_regulator(regulation_input)

        final_weight = effective_weight * regulation_factor

        components = {
            'base_weight': w_global,
            'threshold': tau,
            'context_influence': context_feat.norm(dim=1, keepdim=True),
            'history_adjustment': hist_feat.norm(dim=1, keepdim=True),
            'uncertainty_factor': uncertainty_score,
            'quality_factor': quality_indicator,
            'confidence': confidence,
            'regulation_factor': regulation_factor,
            'final_weight': final_weight
        }

        return final_weight, components

class MedK2N_ResFusionNet(nn.Module):
    def __init__(self, feat_dim=256, input_channels=256, fusion_dim=128, num_residual_blocks=3):
        super().__init__()

        self.align_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=k, padding=k//2),
                nn.GroupNorm(32, feat_dim),
                nn.GELU()
            ) for k in [1, 3, 5]
        ])

        self.align_base = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=k, padding=k//2),
                nn.GroupNorm(32, feat_dim),
                nn.GELU()
            ) for k in [1, 3, 5]
        ])

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feat_dim*2, feat_dim//4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feat_dim//4, feat_dim//8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim//8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.progressive_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dim*2, feat_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, feat_dim),
                nn.GELU(),
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, feat_dim)
            ) for _ in range(3)
        ])

        self.global_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim//8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feat_dim//8, feat_dim//4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feat_dim//4, feat_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, feat_dim),
            nn.GELU()
        )

    def forward(self, enhance_features, base_features, effective_weight):
        B, C, H, W = enhance_features.shape

        enhance_aligned = []
        base_aligned = []

        for align_e, align_b in zip(self.align_enhance, self.align_base):
            enhance_aligned.append(align_e(enhance_features))
            base_aligned.append(align_b(base_features))

        enhance_final = torch.stack(enhance_aligned, dim=0).mean(dim=0)
        base_final = torch.stack(base_aligned, dim=0).mean(dim=0)

        weight_spatial = effective_weight.view(B, 1, 1, 1).expand(B, 1, H, W)

        spatial_concat = torch.cat([enhance_final, base_final], dim=1)
        spatial_attn = self.spatial_attention(spatial_concat)

        weight_combined = weight_spatial * spatial_attn
        weighted_enhance = enhance_final * weight_combined

        fused = weighted_enhance
        for fusion_layer in self.progressive_fusion:
            concat_feat = torch.cat([fused, base_final], dim=1)
            residual = fusion_layer(concat_feat)
            fused = F.gelu(fused + residual)

        global_weight = self.global_modulator(fused)
        modulated = fused * global_weight

        output = self.final_conv(modulated + base_final)

        return output

class MedK2N_TaskHeadNet(nn.Module):
    def __init__(self, input_dim=512, context_dim=128, task_type="medical", 
                 input_channels=256, aux_channels=256, num_output_channels=3, 
                 enable_attention=True):
        super().__init__()
        self.task_type = task_type

        self.task_condition = nn.Sequential(
            nn.Linear(context_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        self.quality_generator = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128), 
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )

        self.upsample_stages = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.GroupNorm(2, 16),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.GroupNorm(2, 16),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.GroupNorm(1, 8),
                nn.GELU(),
                nn.Conv2d(8, 3, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        ])

        self.quality_assessor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.improvement_predictor = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, features, task_context, prev_quality=None, quality_threshold=0.1):
        B, C, H, W = features.shape

        task_feat = self.task_condition(task_context)
        task_spatial = task_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        conditioned_features = features + task_spatial

        feat_flat = conditioned_features.view(B, C, -1).permute(0, 2, 1)
        attended_feat, _ = self.self_attention(feat_flat, feat_flat, feat_flat)
        attended_feat = attended_feat.permute(0, 2, 1).view(B, C, H, W)

        quality_features = self.quality_generator(attended_feat)

        current_quality = self.quality_assessor(quality_features)

        upsampled = quality_features
        for stage in self.upsample_stages:
            upsampled = stage(upsampled)

        if prev_quality is not None:
            quality_input = torch.cat([prev_quality, current_quality], dim=1)
            quality_improvement = self.improvement_predictor(quality_input)
        else:
            quality_improvement = torch.zeros_like(current_quality)

        feedback = {
            'feature_energy': quality_features.norm(dim=1).mean(),
            'quality_score': current_quality.mean(),
            'improvement': quality_improvement.mean(), 
            'meets_threshold': (current_quality > quality_threshold).float().mean(),
            'spatial_consistency': self._compute_spatial_consistency(upsampled)
        }

        return {
            'output': upsampled,
            'quality_score': current_quality,
            'quality_improvement': quality_improvement.squeeze(-1),
            'feedback': feedback
        }

    def _compute_spatial_consistency(self, output):
        grad_x = torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])
        grad_y = torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])
        consistency = 1.0 / (1.0 + grad_x.mean() + grad_y.mean())
        return consistency

class MedK2N_QualityFeedbackNet(nn.Module):
    def __init__(self, num_tasks=4, embed_dim=128, max_history_length=10, 
                 enable_quality_control=True, quality_threshold=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.max_history_length = max_history_length

        self.quality_history = []
        self.performance_history = []

        self.quality_evaluators = nn.ModuleDict()
        for i in range(num_tasks):
            self.quality_evaluators[str(i)] = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                nn.GroupNorm(4, 32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        self.quality_comparator = nn.Sequential(
            nn.Linear(num_tasks * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_tasks),
        )

        self.decision_network = nn.Sequential(
            nn.Linear(num_tasks * 2 + 1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_tasks),
            nn.Sigmoid()
        )

        self.tracker_updater = nn.LSTM(
            input_size=2, hidden_size=32, num_layers=2,
            batch_first=True, bidirectional=True
        )

    def forward(self, current_outputs, previous_outputs, effective_weight_matrix):
        B = list(current_outputs.values())[0].size(0)
        device = list(current_outputs.values())[0].device

        current_qualities = []
        for i, (task_name, output) in enumerate(current_outputs.items()):
            evaluator = self.quality_evaluators[str(i)]
            quality = evaluator(output)
            current_qualities.append(quality)
        current_qualities = torch.cat(current_qualities, dim=1)

        if previous_outputs is not None:
            previous_qualities = []
            for i, (task_name, output) in enumerate(previous_outputs.items()):
                evaluator = self.quality_evaluators[str(i)]
                quality = evaluator(output)
                previous_qualities.append(quality)
            previous_qualities = torch.cat(previous_qualities, dim=1)
        else:
            previous_qualities = torch.zeros_like(current_qualities)

        quality_comparison = torch.cat([current_qualities, previous_qualities], dim=1)
        quality_improvements = self.quality_comparator(quality_comparison)

        if isinstance(effective_weight_matrix, dict):
            weight_values = []
            for key, weight in effective_weight_matrix.items():
                if isinstance(weight, dict) and 'weight' in weight:
                    weight_values.append(weight['weight'])
                else:
                    weight_values.append(weight)
            weight_matrix_flat = torch.cat(weight_values, dim=1)
        else:
            weight_matrix_flat = effective_weight_matrix.view(B, -1)

        decision_input = torch.cat([
            current_qualities, weight_matrix_flat, quality_improvements
        ], dim=1)

        accept_decisions = self.decision_network(decision_input)

        tracker_input = torch.stack([current_qualities, quality_improvements], dim=-1)
        tracker_updated, _ = self.tracker_updater(tracker_input)

        feedback_signals = {
            'quality_scores': current_qualities,
            'quality_improvements': quality_improvements,
            'accept_decisions': accept_decisions,
            'reward_penalties': quality_improvements * accept_decisions,
            'updated_tracker': tracker_updated.mean(dim=1),
            'overall_performance': current_qualities.mean(dim=1, keepdim=True)
        }

        return feedback_signals

class MultiScaleFeaturePyramid(nn.Module):
    def __init__(self, feat_dims=[288, 720, 1440, 1440], target_dim=256, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.target_dim = target_dim

        self.feature_projections = nn.ModuleList()
        for i, feat_dim in enumerate(feat_dims):
            self.feature_projections.append(
                nn.Sequential(
                    nn.Conv2d(feat_dim, target_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(target_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.top_down_convs = nn.ModuleList()
        for i in range(num_levels - 1):
            self.top_down_convs.append(
                nn.Sequential(
                    nn.Conv2d(target_dim, target_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(target_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.output_convs = nn.ModuleList()
        for i in range(num_levels):
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(target_dim, target_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(target_dim),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, backbone_fpn_features):
        if not isinstance(backbone_fpn_features, list) or len(backbone_fpn_features) == 0:
            return []

        if len(backbone_fpn_features) == 1:
            feat0 = backbone_fpn_features[0]
            if feat0 is not None and isinstance(feat0, torch.Tensor):
                projected = self.feature_projections[0](feat0)
                refined = self.output_convs[0](projected)
                return [refined]
            else:
                return []

        projected_features = []
        for i, feat in enumerate(backbone_fpn_features[:self.num_levels]):
            if feat is not None and isinstance(feat, torch.Tensor):
                if i < len(self.feature_projections):
                    projected = self.feature_projections[i](feat)
                    projected_features.append(projected)
                else:
                    projected = self.feature_projections[-1](feat)
                    projected_features.append(projected)
            else:
                if i > 0 and len(projected_features) > 0:
                    ref_feat = projected_features[-1]
                    B, C = ref_feat.shape[:2]
                    H, W = ref_feat.shape[2] // 2, ref_feat.shape[3] // 2
                    zero_feat = torch.zeros(B, self.target_dim, H, W, device=ref_feat.device)
                    projected_features.append(zero_feat)

        if len(projected_features) == 0:
            return []

        pyramid_features = [projected_features[-1]]

        for i in range(len(projected_features) - 2, -1, -1):
            high_level_feat = pyramid_features[0]
            low_level_feat = projected_features[i]

            if high_level_feat.shape[-2:] != low_level_feat.shape[-2:]:
                high_level_feat = F.interpolate(
                    high_level_feat, 
                    size=low_level_feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )

            if i < len(self.top_down_convs):
                fused_feat = high_level_feat + low_level_feat
                fused_feat = self.top_down_convs[i](fused_feat)
            else:
                fused_feat = high_level_feat + low_level_feat

            pyramid_features.insert(0, fused_feat)

        refined_features = []
        for i, feat in enumerate(pyramid_features):
            if i < len(self.output_convs):
                refined = self.output_convs[i](feat)
                refined_features.append(refined)
            else:
                refined_features.append(feat)

        return refined_features

class HierarchicalSpiralMamba(nn.Module):
    def __init__(self, config, pyramid_dim=256, num_levels=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.pyramid_dim = pyramid_dim
        self.num_levels = num_levels

        self.level_mambas = nn.ModuleList()
        for i in range(num_levels):
            self.level_mambas.append(
                MambaLayerOnlyspiral(
                    config=config,
                    dim=pyramid_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            )

        self.cross_level_fusion = nn.ModuleList()
        for i in range(num_levels - 1):
            self.cross_level_fusion.append(
                nn.Sequential(
                    nn.Conv2d(pyramid_dim * 2, pyramid_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(pyramid_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.global_aggregation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pyramid_dim * num_levels, pyramid_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(pyramid_dim, pyramid_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, pyramid_features):
        if not pyramid_features or len(pyramid_features) == 0:
            return []

        mamba_features = []
        for i, feat in enumerate(pyramid_features):
            if i < len(self.level_mambas) and feat is not None:
                try:
                    enhanced_feat = self.level_mambas[i](feat)
                    mamba_features.append(enhanced_feat)
                except Exception as e:
                    mamba_features.append(feat)
            else:
                mamba_features.append(feat)

        enhanced_features = []
        for i, feat in enumerate(mamba_features):
            if feat is None:
                enhanced_features.append(None)
                continue

            current_feat = feat

            if i > 0 and i-1 < len(self.cross_level_fusion) and mamba_features[i-1] is not None:
                prev_feat = mamba_features[i-1]

                if prev_feat.shape[-2:] != current_feat.shape[-2:]:
                    prev_feat = F.interpolate(
                        prev_feat, 
                        size=current_feat.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )

                fused_input = torch.cat([current_feat, prev_feat], dim=1)
                fusion_conv = self.cross_level_fusion[i-1]
                current_feat = fusion_conv(fused_input)

            enhanced_features.append(current_feat)

        if len(enhanced_features) > 1:
            global_feats = []
            base_size = enhanced_features[0].shape[-2:]

            for feat in enhanced_features:
                if feat is not None:
                    if feat.shape[-2:] != base_size:
                        feat_resized = F.interpolate(
                            feat, size=base_size, mode='bilinear', align_corners=False
                        )
                    else:
                        feat_resized = feat
                    global_feats.append(feat_resized)

            if global_feats:
                global_concat = torch.cat(global_feats, dim=1)
                global_attention = self.global_aggregation(global_concat)

                for i, feat in enumerate(enhanced_features):
                    if feat is not None:
                        if global_attention.shape[-2:] != feat.shape[-2:]:
                            attention_resized = F.interpolate(
                                global_attention, 
                                size=feat.shape[-2:], 
                                mode='bilinear', 
                                align_corners=False
                            )
                        else:
                            attention_resized = global_attention

                        enhanced_features[i] = feat * attention_resized

        return enhanced_features

class MambaLayerOnlyspiral(nn.Module):
    def __init__(self, config, dim=288, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba1 = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba2 = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.conv1d = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1)
        base_dir = None
        if config is not None:
            base_dir = getattr(config, 'spiral_base_dir', None) or config.get('spiral_base_dir', None) if isinstance(config, dict) else None           
        base_dir = ''
        self.spiral_eye = torch.tensor(np.load(f"-"), dtype=torch.float)
        self.despiral_eye = torch.tensor(np.load(f"-"), dtype=torch.float)
        self.despiral_r_eye = torch.tensor(np.load(f"-"), dtype=torch.float)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim, f"Expect C={self.dim}, got {C}"
        device = x.device
        spiral_eye = self.spiral_eye.to(device)
        despiral_eye = self.despiral_eye.to(device)
        despiral_r_eye = self.despiral_r_eye.to(device)
        x_flat = x.view(B, C, -1)
        seq = torch.einsum('ij,bcj->bci', spiral_eye, x_flat)
        seq = seq.permute(0, 2, 1)
        seq_r = torch.flip(seq, dims=[1])
        norm1 = self.norm(seq)
        out1 = self.mamba1(norm1)
        norm2 = self.norm(seq_r)
        out2 = self.mamba2(norm2)
        out1_bcl = out1.permute(0, 2, 1)
        out2_bcl = out2.permute(0, 2, 1)
        rec1 = torch.einsum('ij,bcj->bci', despiral_eye, out1_bcl).view(B, C, H, W)
        rec2 = torch.einsum('ij,bcj->bci', despiral_r_eye, out2_bcl).view(B, C, H, W)
        fused = torch.cat([rec1, rec2], dim=1)
        fused = self.conv1d(fused)
        return fused

__all__ = [
    'apply_lora_to_sam2_encoder',
    'MedK2N_PreweightNet',
    'MedK2N_ThresholdNet', 
    'MedK2N_EffeWeightNet',
    'MedK2N_ResFusionNet',
    'MedK2N_TaskHeadNet',
    'MedK2N_QualityFeedbackNet',
    'CausalModalityIdentityModule',
    'MultiHeadGenerator',
    'CrossFrameFusion', 
    'BottleneckCNN',
    'ResnetBlock',
    'SimplifiedSAM2'
]

class SimplifiedSAM2(nn.Module):
    def __init__(
        self,
        config,
        base_model: SAM2Base,
        enable_rgb_generation: bool = True,
        num_output_frames: int = 4,
        num_generator_heads: int = 4,
        enable_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules=None,
    ):
        super(SimplifiedSAM2, self).__init__()

        if enable_lora:
            self.student = apply_lora_to_sam2_encoder(
                base_model,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
        else:
            self.student = base_model

        self.config = config
        self.enable_rgb_generation = enable_rgb_generation
        self.num_output_frames = num_output_frames
        self.debug = getattr(config, 'debug', False)

        feat_dim = getattr(config, 'feat_dim', 512)
        embed_dim = getattr(config, 'embed_dim', 128)
        num_modalities = getattr(config, 'num_modalities', 4)

        try:
            self.multi_scale_pyramid = MultiScaleFeaturePyramid(
                feat_dims=[288, 720, 1440, 1440],
                target_dim=256,
                num_levels=4
            )
            if self.debug:
                pass
        except Exception as e:
            self.multi_scale_pyramid = None
            if self.debug:
                pass

        try:
            self.hierarchical_spiral_mamba = HierarchicalSpiralMamba(
                config=config,
                pyramid_dim=256,
                num_levels=4,
                d_state=16,
                d_conv=4,
                expand=2
            )
            if self.debug:
                pass
        except Exception as e:
            self.hierarchical_spiral_mamba = None
            if self.debug:
                pass

        self.preweight_net = MedK2N_PreweightNet(
            base_channels=256,
            enhance_channels=256,
            embed_dim=embed_dim,
            num_modalities=num_modalities,
            enable_history_tracking=True,
            enable_compatibility_assessment=True
        )

        self.threshold_net = MedK2N_ThresholdNet(
            modality_dim=embed_dim,
            task_dim=embed_dim,
            compatibility_dim=64,
            performance_history_dim=32
        )

        self.effe_weight_net = MedK2N_EffeWeightNet(
            base_channels=256,
            aux_channels=256,
            modality_embed_dim=embed_dim,
            task_embed_dim=embed_dim,
            spatial_resolution=(64, 64)
        )

        self.res_fusion_net = MedK2N_ResFusionNet(
            input_channels=256,
            fusion_dim=embed_dim,
            num_residual_blocks=3
        )

        self.task_head_net = MedK2N_TaskHeadNet(
            input_dim=256,
            context_dim=embed_dim,
            task_type="medical",
            input_channels=256,
            aux_channels=256,
            num_output_channels=4,
            enable_attention=True
        )

        quality_threshold = getattr(config, 'quality_threshold', 0.1)
        self.quality_feedback_net = MedK2N_QualityFeedbackNet(
            num_tasks=4,
            embed_dim=embed_dim,
            max_history_length=10,
            enable_quality_control=True,
            quality_threshold=quality_threshold
        )

        if enable_rgb_generation:
            self.multi_head_generator = MultiHeadGenerator(
                input_dim=256,
                output_frames=num_output_frames,
                num_heads=num_generator_heads
            )

        self.cross_frame_fusion = CrossFrameFusion(
            dim=288,
            num_frames=num_output_frames,
            fusion_type='attention'
        )

        self.fpn_pyramid = MultiScaleFeaturePyramid(
            feat_dims=[288, 720, 1440, 1440],
            target_dim=256,
            num_levels=4
        )

        self.hierarchical_spiral_mamba = HierarchicalSpiralMamba(
            config=config,
            pyramid_dim=256,
            num_levels=4,
            d_state=getattr(config, 'd_state', 16),
            d_conv=getattr(config, 'd_conv', 4)
        )

        self.bottleneck = BottleneckCNN(config=config)

        d_model = getattr(config, 'd_model', 256)
        d_state = getattr(config, 'd_state', 16)
        d_conv = getattr(config, 'd_conv', 4)

        self.spiral_mamba = MambaLayerOnlyspiral(
            config=config,
            dim=288,
            d_state=d_state,
            d_conv=d_conv
        )

        self.causal_identity_module = CausalModalityIdentityModule(
            num_modalities=num_modalities,
            feat_dim=feat_dim,
            identity_dim=embed_dim,
            semantic_dim=256
        )

        self.multi_frame_fuse = 'mean'
        self.multi_scale_max_levels = getattr(config, 'multi_scale_levels', 3)

    def _run_model(self, model, batched_input):
        m = len(batched_input)
        image_embeddings, backbone_outs, vision_feats, vision_pos_embeds, feat_sizes, highres_outputs = [], [], [], [], [], []
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        key_frame_idx = 0
        aux_frame_indices = list(range(1, m))

        f_base = None
        f_aux_list = []

        if self.debug and m > 1:
            pass

        for i in range(m):
            img_emb = model.forward_image(batched_input[i])
            image_embeddings.append(img_emb)
            backbone_out_item, vision_feats_item, vision_pos_embeds_item, feat_sizes_item = model._prepare_backbone_features(img_emb)

            try:
                if self.debug:
                    frame_role = 'Key' if i == key_frame_idx else 'Aux'
                    b0 = None
                    if isinstance(backbone_out_item, dict) and 'backbone_fpn' in backbone_out_item and len(backbone_out_item['backbone_fpn'])>0:
                        b0 = backbone_out_item['backbone_fpn'][0]
                    else:
                        pass
            except Exception:
                pass

            fpn_features = []
            if isinstance(backbone_out_item, dict) and 'backbone_fpn' in backbone_out_item:
                backbone_fpn = backbone_out_item['backbone_fpn']

                for level_idx in range(len(backbone_fpn)):
                    feat = backbone_fpn[level_idx] if level_idx < len(backbone_fpn) else None
                    fpn_features.append(feat)

                    if self.debug and feat is not None and isinstance(feat, torch.Tensor):
                        pass

                if len(fpn_features) > 0 and fpn_features[0] is not None:
                    feat0 = fpn_features[0]

                    if isinstance(feat0, torch.Tensor):
                        if i == key_frame_idx:
                            f_base = feat0.clone()
                            f_base_pyramid = [feat.clone() if feat is not None else None for feat in fpn_features]
                            if self.debug:
                                pass
                        else:
                            f_aux_list.append({
                                'frame_idx': i, 
                                'feat': feat0.clone(),
                                'pyramid_feats': [feat.clone() if feat is not None else None for feat in fpn_features]
                            })
                            if self.debug:
                                pass
            else:
                if isinstance(backbone_out_item, dict) and 'backbone_fpn' in backbone_out_item and len(backbone_out_item['backbone_fpn']) > 0:
                    feat0 = backbone_out_item['backbone_fpn'][0]
                    if isinstance(feat0, torch.Tensor):
                        if i == key_frame_idx:
                            f_base = feat0.clone()
                            f_base_pyramid = [feat0.clone()]
                            if self.debug:
                                pass
                        else:
                            f_aux_list.append({
                                'frame_idx': i, 
                                'feat': feat0.clone(),
                                'pyramid_feats': [feat0.clone()]
                            })
                            if self.debug:
                                pass

            backbone_outs.append(backbone_out_item)
            vision_feats.append(vision_feats_item)
            vision_pos_embeds.append(vision_pos_embeds_item)
            feat_sizes.append(feat_sizes_item)

        if f_base is not None and f_base_pyramid is not None and len(f_aux_list) > 0:
            try:
                if self.debug:
                    pass

                aux_pyramids = []
                for aux_data in f_aux_list:
                    if 'pyramid_feats' in aux_data and aux_data['pyramid_feats'] is not None:
                        aux_pyramids.append(aux_data['pyramid_feats'])
                    elif 'feat' in aux_data:
                        aux_pyramids.append([aux_data['feat']])

                if hasattr(self, 'multi_scale_pyramid') and self.multi_scale_pyramid is not None and len(aux_pyramids) > 0:
                    try:
                        pyramid_features = [f_base_pyramid]
                        pyramid_features.extend(aux_pyramids)

                        enhanced_pyramid = self.multi_scale_pyramid(pyramid_features)

                        if self.debug:
                            for level, feat in enumerate(enhanced_pyramid):
                                if feat is not None:
                                    pass

                        if hasattr(self, 'hierarchical_spiral_mamba') and self.hierarchical_spiral_mamba is not None:
                            try:
                                enhanced_pyramid = self.hierarchical_spiral_mamba(enhanced_pyramid)
                                if self.debug:
                                    pass
                            except Exception as e:
                                if self.debug:
                                    pass

                        if len(enhanced_pyramid) > 0 and enhanced_pyramid[0] is not None:
                            current_base_feat = enhanced_pyramid[0]
                            enhanced_success = True
                        else:
                            current_base_feat = f_base.clone()
                            enhanced_success = False

                    except Exception as e:
                        if self.debug:
                            pass
                        current_base_feat = f_base.clone()
                        enhanced_success = False
                else:
                    current_base_feat = f_base.clone()
                    enhanced_success = False
                    if self.debug:
                        pass

                if not enhanced_success:
                    B, C, H, W = current_base_feat.shape
                    device = current_base_feat.device

                    for aux_info in f_aux_list:
                        aux_idx = aux_info['frame_idx'] 
                        f_aux = aux_info['feat']

                        if self.debug:
                            pass

                        try:
                            quality_history = torch.zeros(B, 1, 32, device=device)
                            task_compatibility = torch.ones(B, 1, device=device)
                            mod_idx = aux_idx % 4
                            modality_embed = torch.randn(B, 128, device=device)

                            preweight = self.preweight_net(
                                base_features=current_base_feat,
                                enhance_features=f_aux,
                                prev_outputs=None,
                                quality_history=quality_history,
                                task_compatibility=task_compatibility,
                                modality_embed=modality_embed
                            )

                            if self.debug:
                                pass

                            mod_embed = torch.randn(B, 128, device=device)
                            task_embed = torch.randn(B, 128, device=device)
                            performance_tracker = torch.zeros(B, 1, 32, device=device)

                            threshold = self.threshold_net(
                                mod_embed=mod_embed,
                                task_embed=task_embed,
                                compat_matrix=torch.ones(B, 1, device=device),
                                perf_tracker=performance_tracker,
                                context_embedding=mod_embed + task_embed,
                                current_weight=preweight,
                                quality_gain=torch.zeros(B, 1, device=device)
                            )

                            if self.debug:
                                pass

                            effective_weights = self.effe_weight_net(
                                base_features=current_base_feat,
                                aux_features=f_aux,
                                preweight=preweight,
                                threshold=threshold,
                                modality_embed=mod_embed,
                                task_embed=task_embed
                            )
                            final_weight = effective_weights.get('final_weights', preweight)

                            if self.debug:
                                pass

                            spatial_weight = final_weight.unsqueeze(-1).unsqueeze(-1)

                            fused_feat = current_base_feat * (1 - spatial_weight) + f_aux * spatial_weight
                            current_base_feat = fused_feat

                            if self.debug:
                                pass

                            if isinstance(backbone_outs[aux_idx], dict) and 'backbone_fpn' in backbone_outs[aux_idx]:
                                backbone_outs[aux_idx]['backbone_fpn'][0] = fused_feat.clone()

                        except Exception as e:
                            if self.debug:
                                pass
                            fused_feat = 0.7 * current_base_feat + 0.3 * f_aux
                            current_base_feat = fused_feat

                if isinstance(backbone_outs[key_frame_idx], dict) and 'backbone_fpn' in backbone_outs[key_frame_idx]:
                    backbone_outs[key_frame_idx]['backbone_fpn'][0] = current_base_feat
                    if self.debug:
                        pass

            except Exception as e:
                pass

        elif hasattr(self, 'cross_frame_fusion') and self.cross_frame_fusion is not None:
            frame_fpn0_features = []
            for i in range(m):
                if isinstance(backbone_outs[i], dict) and 'backbone_fpn' in backbone_outs[i] and len(backbone_outs[i]['backbone_fpn']) > 0:
                    feat0 = backbone_outs[i]['backbone_fpn'][0]
                    if isinstance(feat0, torch.Tensor):
                        frame_fpn0_features.append(feat0)

            if len(frame_fpn0_features) > 1:
                try:
                    if self.debug:
                        pass

                    fused_fpn0 = self.cross_frame_fusion(frame_fpn0_features)

                    if self.debug:
                        pass

                    for i in range(m):
                        if isinstance(backbone_outs[i], dict) and 'backbone_fpn' in backbone_outs[i] and len(backbone_outs[i]['backbone_fpn']) > 0:
                            backbone_outs[i]['backbone_fpn'][0] = fused_fpn0.clone()

                        if isinstance(vision_feats[i], (list, tuple)) and len(vision_feats[i]) > 0:
                            new_vf = list(vision_feats[i])
                            if isinstance(new_vf[0], torch.Tensor) and len(new_vf[0].shape) >= 2 and new_vf[0].shape[-2:] == fused_fpn0.shape[-2:]:
                                new_vf[0] = fused_fpn0.clone()
                                vision_feats[i] = tuple(new_vf) if isinstance(vision_feats[i], tuple) else new_vf

                except Exception as e:
                    pass

        for i in range(m):
            try:
                if isinstance(backbone_outs[i], dict) and 'backbone_fpn' in backbone_outs[i] and len(backbone_outs[i]['backbone_fpn']) > 0:
                    feat0 = backbone_outs[i]['backbone_fpn'][0]
                    if isinstance(feat0, torch.Tensor) and feat0.dim() == 4 and feat0.shape[-2:] == (64, 64) and feat0.shape[1] == 288:
                        if self.debug:
                            pass
                        device = feat0.device
                        new_feat0 = self.bottleneck(feat0)
                        new_feat0 = self.spiral_mamba(new_feat0)
                        if self.debug:
                            pass
                        backbone_outs[i]['backbone_fpn'][0] = new_feat0
                if isinstance(vision_feats[i], (list, tuple)):
                    new_vf = []
                    for vf in vision_feats[i]:
                        if isinstance(vf, torch.Tensor) and vf.dim() == 4 and vf.shape[-2:] == (64, 64) and vf.shape[1] == 288:
                            if self.debug:
                                pass
                            nv = self.bottleneck(vf)
                            nv = self.spiral_mamba(nv)
                            if self.debug:
                                pass
                            new_vf.append(nv)
                        else:
                            new_vf.append(vf)
                    vision_feats[i] = tuple(new_vf) if isinstance(vision_feats[i], tuple) else new_vf
            except Exception as e:
                pass

        for frame_idx in range(m):
            is_init_cond_frame = frame_idx == 0
            multi_mask_output = model.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=vision_feats[frame_idx],
                current_vision_pos_embeds=vision_pos_embeds[frame_idx],
                feat_sizes=feat_sizes[frame_idx],
                point_inputs=None,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=m,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
            output_dict["cond_frame_outputs"][frame_idx] = multi_mask_output
            highres_outputs.append(multi_mask_output["high_res_multimasks"])
            try:
                if self.debug:
                    hmask = multi_mask_output["high_res_multimasks"]
            except Exception:
                pass

        return {
            'image_embeddings': image_embeddings,
            'highres_masks': highres_outputs,
            'backbone_outs': backbone_outs,
            'vision_feats': vision_feats,
            'feat_sizes': feat_sizes,
        }

    def _fuse_multi_frame_masks(self, mask_list):
        if len(mask_list) == 1:
            return mask_list[0]
        if self.multi_frame_fuse == 'none':
            return mask_list[0]
        stacked = torch.stack(mask_list, dim=0)
        if self.multi_frame_fuse == 'mean':
            return stacked.mean(0)
        if self.multi_frame_fuse == 'attention':
            with torch.no_grad():
                weights = torch.softmax(torch.randn(stacked.size(0), device=stacked.device), dim=0)
            fused = (stacked * weights.view(-1, 1, 1, 1, 1)).sum(0)
            return fused
        return stacked.mean(0)

    def forward(self, batched_input, multimask_output=None, inference_head_indices=None):
        student_out = self._run_model(self.student, batched_input)

        student_feat = None
        if student_out['backbone_outs'] and len(student_out['backbone_outs']) > 0:
            backbone_out = student_out['backbone_outs'][0]
            if isinstance(backbone_out, dict) and 'backbone_fpn' in backbone_out and len(backbone_out['backbone_fpn']) > 0:
                student_feat = backbone_out['backbone_fpn'][0]

        generated_rgb_frames = None
        if self.enable_rgb_generation and hasattr(self, 'multi_head_generator') and student_feat is not None:
            try:
                generated_rgb_frames = self.multi_head_generator(student_feat, head_indices=inference_head_indices)

                if self.debug and isinstance(generated_rgb_frames, list) and len(generated_rgb_frames) > 0:
                    pass
            except Exception as e:
                generated_rgb_frames = None

        simple_outputs = {
            'student_masks': student_out.get('highres_masks', []),
            'student_features': student_feat,
            'generated_frames': generated_rgb_frames
        }

        return generated_rgb_frames, student_feat, simple_outputs

    def train(self, mode=True):
        super().train(mode)
        self.student.train(mode)
        return self

    def eval(self):
        return self.train(False)

class CausalModalityIdentityModule(nn.Module):
    def __init__(self, num_modalities=4, feat_dim=256, identity_dim=128, semantic_dim=256):
        super(CausalModalityIdentityModule, self).__init__()

        self.num_modalities = num_modalities
        self.feat_dim = feat_dim
        self.identity_dim = identity_dim
        self.semantic_dim = semantic_dim

        self.modality_identity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, identity_dim),
                nn.LayerNorm(identity_dim),
                nn.ReLU(),
                nn.Linear(identity_dim, identity_dim),
                nn.LayerNorm(identity_dim),
                nn.Tanh()
            ) for _ in range(num_modalities)
        ])

        self.causal_consistency_validator = nn.Sequential(
            nn.Linear(identity_dim * 2, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(semantic_dim, semantic_dim // 2),
            nn.LayerNorm(semantic_dim // 2),
            nn.ReLU(),
            nn.Linear(semantic_dim // 2, 1),
            nn.Sigmoid()
        )

        self.identity_alignment_projector = nn.Sequential(
            nn.Linear(feat_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, identity_dim),
            nn.LayerNorm(identity_dim)
        )

        self.semantic_preservation_validator = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, semantic_dim // 4),
            nn.ReLU(),
            nn.Linear(semantic_dim // 4, 1),
            nn.Sigmoid()
        )

        self.modality_prototypes = nn.Parameter(torch.randn(num_modalities, identity_dim))
        nn.init.xavier_uniform_(self.modality_prototypes)

    def forward(self, source_features, target_generated, source_modality_id, target_modality_id, 
                source_image=None, target_image=None):
        batch_size = source_features.shape[0]
        device = source_features.device

        source_identity = self.modality_identity_encoders[source_modality_id[0].item()](source_features)
        target_identity_expected = self.modality_prototypes[target_modality_id[0].item()].unsqueeze(0).expand(batch_size, -1)

        target_generated_features = self.identity_alignment_projector(
            target_generated.view(batch_size, -1)
        )

        causal_input = torch.cat([source_identity, target_identity_expected], dim=1)
        causal_consistency_score = self.causal_consistency_validator(causal_input)

        identity_alignment_loss = F.mse_loss(target_generated_features, target_identity_expected.detach())

        semantic_preservation_score = None
        if source_image is not None:
            source_semantic_score = self.semantic_preservation_validator(source_image)
            target_semantic_score = self.semantic_preservation_validator(target_generated)
            semantic_preservation_score = 1.0 - torch.abs(source_semantic_score - target_semantic_score)

        causal_consistency_loss = 1.0 - causal_consistency_score.mean()

        semantic_preservation_loss = None
        if semantic_preservation_score is not None:
            semantic_preservation_loss = 1.0 - semantic_preservation_score.mean()

        return {
            'causal_consistency_loss': causal_consistency_loss,
            'identity_alignment_loss': identity_alignment_loss,
            'semantic_preservation_loss': semantic_preservation_loss,
            'causal_consistency_score': causal_consistency_score.mean().item(),
            'semantic_preservation_score': semantic_preservation_score.mean().item() if semantic_preservation_score is not None else None,
            'source_identity': source_identity,
            'target_identity_expected': target_identity_expected,
            'target_identity_generated': target_generated_features
        }

class SAM2LoRAConfig:
    def __init__(self):
        self.enable_lora = True
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

        self.enable_rgb_generation = True
        self.num_output_frames = 4
        self.num_generator_heads = 4

        self.multi_scale_levels = 3

        self.feat_dim = 256
        self.embed_dim = 128
        self.num_modalities = 4
        self.quality_threshold = 0.1

        self.d_model = 256
        self.d_state = 16
        self.d_conv = 4

        self.enable_quality_feedback = True
        self.progressive_fusion = True

    def update_from_yaml_config(self, yaml_config):
        model_config = yaml_config.get('MODEL', {})
        medk2n_config = yaml_config.get('MEDK2N', {})

        combined_config = {**model_config, **medk2n_config}

        if 'NUM_MODALITIES' in combined_config:
            self.num_modalities = combined_config['NUM_MODALITIES']
            self.num_generator_heads = self.num_modalities
            self.num_output_frames = self.num_modalities

        if 'NUM_GENERATOR_HEADS' in combined_config:
            self.num_generator_heads = combined_config['NUM_GENERATOR_HEADS']

        if 'NUM_OUTPUT_FRAMES' in combined_config:
            self.num_output_frames = combined_config['NUM_OUTPUT_FRAMES']

        if 'LORA_R' in combined_config:
            self.lora_r = combined_config['LORA_R']

        if 'LORA_ALPHA' in combined_config:
            self.lora_alpha = combined_config['LORA_ALPHA']

        if 'LORA_DROPOUT' in combined_config:
            self.lora_dropout = combined_config['LORA_DROPOUT']

        if 'FEAT_DIM' in combined_config:
            self.feat_dim = combined_config['FEAT_DIM']

        if 'EMBED_DIM' in combined_config:
            self.embed_dim = combined_config['EMBED_DIM']

        if 'QUALITY_THRESHOLD' in combined_config:
            self.quality_threshold = combined_config['QUALITY_THRESHOLD']

        if 'ENABLE_QUALITY_FEEDBACK' in combined_config:
            self.enable_quality_feedback = combined_config['ENABLE_QUALITY_FEEDBACK']

        if 'PROGRESSIVE_FUSION' in combined_config:
            self.progressive_fusion = combined_config['PROGRESSIVE_FUSION']

        if 'D_MODEL' in combined_config:
            self.d_model = combined_config['D_MODEL']

        if 'D_STATE' in combined_config:
            self.d_state = combined_config['D_STATE']

        if 'D_CONV' in combined_config:
            self.d_conv = combined_config['D_CONV']

def apply_lora_to_sam2_encoder(model, target_modules=None, r=16, lora_alpha=32, lora_dropout=0.1):
    if not PEFT_AVAILABLE:
        return model

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

    try:
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=None
        )

        peft_model = get_peft_model(model, lora_config)

        return peft_model

    except Exception as e:
        return model

def create_sam2_lora_model(teacher_model, student_model, config=None):
    if config is None:
        config = SAM2LoRAConfig()

    model = SimplifiedSAM2(
        config=config,
        base_model=student_model,
        enable_rgb_generation=getattr(config, 'enable_rgb_generation', True),
        num_output_frames=getattr(config, 'num_output_frames', 4),
        num_generator_heads=getattr(config, 'num_generator_heads', 4),
        enable_lora=getattr(config, 'enable_lora', True),
        lora_r=getattr(config, 'lora_r', 16),
        lora_alpha=getattr(config, 'lora_alpha', 32),
        lora_dropout=getattr(config, 'lora_dropout', 0.1),
        lora_target_modules=getattr(config, 'lora_target_modules', None)
    )

    return model

__all__ = [
    'SimplifiedSAM2',
    'MedK2N_PreweightNet',
    'MedK2N_ThresholdNet',
    'MedK2N_EffeWeightNet',
    'MedK2N_ResFusionNet',
    'MedK2N_TaskHeadNet',
    'MedK2N_QualityFeedbackNet',
    'CausalModalityIdentityModule',
    'MultiHeadGenerator',
    'ResnetBlock',
    'BottleneckCNN',
    'CrossFrameFusion',
    'MambaLayerOnlyspiral',
    'SAM2LoRAConfig',
    'apply_lora_to_sam2_encoder',
    'create_sam2_lora_model',
    'MEDK2N_CONFIG',
    'MEDK2N_MODULE_SPECS'
]