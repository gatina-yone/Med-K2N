#!/usr/bin/env python3
import os
import string
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .medical_modality_clip_loss import MedicalModalityCLIPLoss, MultiModalityCLIPLoss
    MEDICAL_CLIP_AVAILABLE = True
    if not globals().get('_MEDICAL_CLIP_PRINTED', False):
        print("✓ 医学模态CLIP损失可用")
        globals()['_MEDICAL_CLIP_PRINTED'] = True
except ImportError as e:
    print(f"Warning: 医学模态CLIP损失不可用: {e}")
    MEDICAL_CLIP_AVAILABLE = False


class CLIPLossIntegrator:
    """将CLIP损失集成到现有损失中的工具类（精简版）"""
    def __init__(self, device, modality_mapping=None, dataset_root: Optional[str] = None, dataset_split: str = 'val'):
        self.device = device
        self.modality_mapping = modality_mapping
        if self.modality_mapping is None and dataset_root is not None:
            self.modality_mapping = self._infer_letter_mapping_from_dataset(dataset_root, dataset_split)
            print(f"✓ 自动推断模态映射: {self.modality_mapping}")
        if self.modality_mapping is None:
            self.modality_mapping = {}

        # 尝试从配置文件加载模态描述（可选），优先使用配置
        modality_descriptions = None
        quality_descriptions = None
        try:
            import yaml
            cfg_path = os.path.join('configs', 'modality_descriptions.yaml')
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                modality_descriptions = cfg.get('modality_descriptions')
                quality_descriptions = cfg.get('quality_descriptions')
        except Exception:
            modality_descriptions = None
            quality_descriptions = None

        if MEDICAL_CLIP_AVAILABLE:
            # 将配置注入到 MedicalModalityCLIPLoss（若有），否则使用默认内置描述
            self.clip_loss = MedicalModalityCLIPLoss(modality_descriptions=modality_descriptions, quality_descriptions=quality_descriptions).to(device)
        else:
            self.clip_loss = None

    def integrate_with_existing_loss(self, original_criterion, clip_weight=0.1,
                                     target_modal_letters=None, source_modal_letters=None):
        class EnhancedCriterionWithCLIP(nn.Module):
            def __init__(self, original_criterion, clip_loss, clip_weight, device, letter2modality=None):
                super().__init__()
                self.original_criterion = original_criterion
                self.clip_loss = clip_loss
                self.clip_weight = clip_weight
                self.device = device
                self.letter2modality = letter2modality or {}
                self.target_letters = list(target_modal_letters) if isinstance(target_modal_letters, (list, tuple)) else (
                    [target_modal_letters] if isinstance(target_modal_letters, str) and target_modal_letters else [])
                self.source_letters = list(source_modal_letters) if isinstance(source_modal_letters, (list, tuple)) else (
                    [source_modal_letters] if isinstance(source_modal_letters, str) and source_modal_letters else [])

            def forward(self, pred, target, inputs):
                original_loss, original_losses = self.original_criterion(pred, target, inputs)
                clip_loss_value = torch.tensor(0.0, device=self.device)
                l_causal_value = torch.tensor(0.0, device=self.device)
                l_metric_value = torch.tensor(0.0, device=self.device)
                
                if self.clip_loss is not None and pred is not None:
                    try:
                        generated_images = pred
                        tgt_letter = self.target_letters[-1] if len(self.target_letters) > 0 else None
                        src_letter = self.source_letters[0] if len(self.source_letters) > 0 else None
                        target_name = self.letter2modality.get(tgt_letter, tgt_letter) if tgt_letter is not None else None
                        source_name = self.letter2modality.get(src_letter, src_letter) if src_letter is not None else None
                        if generated_images.dim() == 4 and generated_images.size(1) == 1:
                            generated_images = generated_images.repeat(1, 3, 1, 1)
                        
                        # 计算CLIP损失并获取详细损失字典
                        clip_loss_value, clip_losses_dict = self.clip_loss(
                            generated_images,
                            target_name,
                            source_name,
                            use_quality_loss=True,
                            use_modality_consistency=True
                        )
                        
                        # 提取Med-K2N的L_causal和L_metric损失
                        if isinstance(clip_losses_dict, dict):
                            l_causal_value = clip_losses_dict.get('L_causal', torch.tensor(0.0, device=self.device))
                            l_metric_value = clip_losses_dict.get('L_metric', torch.tensor(0.0, device=self.device))
                        
                    except Exception as e:
                        clip_loss_value = torch.tensor(0.0, device=self.device)
                        l_causal_value = torch.tensor(0.0, device=self.device)
                        l_metric_value = torch.tensor(0.0, device=self.device)

                total_loss = original_loss + self.clip_weight * clip_loss_value
                enhanced_losses = original_losses.copy()
                enhanced_losses['clip'] = clip_loss_value
                # 添加Med-K2N特有的损失项
                enhanced_losses['L_causal'] = l_causal_value
                enhanced_losses['L_metric'] = l_metric_value
                enhanced_losses['total'] = total_loss
                return total_loss, enhanced_losses

        return EnhancedCriterionWithCLIP(
            original_criterion, self.clip_loss, clip_weight, self.device, letter2modality=self.modality_mapping
        )

    def _infer_letter_mapping_from_dataset(self, dataset_root: str, split: str = 'val'):
        modal_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(modal_dir):
            raise FileNotFoundError(f"Dataset split directory not found: {modal_dir}")
        entries = [d for d in os.listdir(modal_dir) if os.path.isdir(os.path.join(modal_dir, d))]
        entries = sorted(entries)
        letters = list(string.ascii_uppercase)
        mapping = {letters[i]: entries[i] for i in range(min(len(entries), len(letters)))}
        return mapping
