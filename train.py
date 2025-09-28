#!/usr/bin/env python3
import sys
import os
import argparse
import yaml
# Fix: Use relative path instead of hardcoded absolute path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'semseg/models/sam2'))
import torch
import torch.nn as nn
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import cast, Optional
# SummaryWriter (TensorBoard) removed; ImageSaver saves visualizations to disk instead.
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.datasets.medicalmri import MedicalMRI
from torch.nn.utils.clip_grad import clip_grad_norm_
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.losses.medical_modality_clip_loss import MedicalModalityCLIPLoss
from semseg.losses.clip_loss_integration import CLIPLossIntegrator
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger
import numpy as np
import random
from semseg.models.sam2.sam2.build_sam import build_sam2 as build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import (
    SimplifiedSAM2,
    SAM2LoRAConfig,
    create_sam2_lora_model,
    MedK2N_PreweightNet,
    MedK2N_ThresholdNet, 
    MedK2N_EffeWeightNet,
    MedK2N_ResFusionNet,
    MedK2N_TaskHeadNet,
    MedK2N_QualityFeedbackNet,
    MEDK2N_CONFIG,
    MEDK2N_MODULE_SPECS
)
import torchmetrics
import matplotlib.pyplot as plt
# cv2 import removed - not used
from PIL import Image
import gc
   
CLIP_AVAILABLE = True

class MedK2N_TrainingManager:
    def __init__(self, config, device, in_feat_channels: Optional[int] = None):
        self.config = config
        self.device = device
        self.medk2n_modules = {}
        self.quality_history = []
        self.performance_tracker = None
        self.in_feat_channels = in_feat_channels
        self._update_medk2n_config_from_yaml(config)
        
        print("🚀 [MedK2N] Initializing training manager")
        print(f"📊 [MedK2N] Using config: feat_dim={MEDK2N_CONFIG['feat_dim']}, embed_dim={MEDK2N_CONFIG['embed_dim']}")
        self._initialize_modules()
        
    def _update_medk2n_config_from_yaml(self, config):
        """Update MEDK2N_CONFIG from YAML configuration"""
        global MEDK2N_CONFIG  # Need to modify global config
        medk2n_cfg = config.get('MEDK2N', {})
        
        # Update feat_dim (most important optimization parameter)
        if 'FEAT_DIM' in medk2n_cfg:
            MEDK2N_CONFIG['feat_dim'] = medk2n_cfg['FEAT_DIM']
            print(f"✅ [MedK2N] feat_dim updated from YAML: 512 → {MEDK2N_CONFIG['feat_dim']}")
        
        # Update embed_dim
        if 'EMBED_DIM' in medk2n_cfg:
            MEDK2N_CONFIG['embed_dim'] = medk2n_cfg['EMBED_DIM']
            print(f"✅ [MedK2N] embed_dim updated from YAML: {MEDK2N_CONFIG['embed_dim']}")
            
        # Update num_modalities
        if 'NUM_MODALITIES' in medk2n_cfg:
            MEDK2N_CONFIG['num_modalities'] = medk2n_cfg['NUM_MODALITIES']
            print(f"✅ [MedK2N] num_modalities updated from YAML: {MEDK2N_CONFIG['num_modalities']}")
            
        # Update other configurations
        if 'QUALITY_THRESHOLD' in medk2n_cfg:
            MEDK2N_CONFIG['quality_threshold'] = medk2n_cfg['QUALITY_THRESHOLD']
        if 'ENABLE_QUALITY_FEEDBACK' in medk2n_cfg:
            MEDK2N_CONFIG['enable_quality_feedback'] = medk2n_cfg['ENABLE_QUALITY_FEEDBACK']
        if 'PROGRESSIVE_FUSION' in medk2n_cfg:
            MEDK2N_CONFIG['progressive_fusion'] = medk2n_cfg['PROGRESSIVE_FUSION']
        
    def _initialize_modules(self):
        """Initialize MedK2N six core modules"""
        feat_dim = MEDK2N_CONFIG['feat_dim']
        embed_dim = MEDK2N_CONFIG['embed_dim']
        num_tasks = MEDK2N_CONFIG['num_modalities']
        
        # 1. PreweightNet - Task-specific weight prediction
        self.medk2n_modules['preweight'] = MedK2N_PreweightNet(
            feat_dim=feat_dim, embed_dim=embed_dim, num_modalities=num_tasks
        ).to(self.device)
        
        # 2. ThresholdNet - Adaptive threshold learning
        self.medk2n_modules['threshold'] = MedK2N_ThresholdNet(
            embed_dim=embed_dim
        ).to(self.device)
        
        # 3. EffeWeightNet - Effective weight computation
        self.medk2n_modules['effe_weight'] = MedK2N_EffeWeightNet(
            embed_dim=embed_dim
        ).to(self.device)
        
        # 4. ResFusionNet - Residual fusion network
        self.medk2n_modules['res_fusion'] = MedK2N_ResFusionNet(
            feat_dim=feat_dim
        ).to(self.device)
        # If model output feature channels don't match expected feat_dim, create adapter (1x1 conv)
        if self.in_feat_channels is not None and self.in_feat_channels != feat_dim:
            try:
                adapter = torch.nn.Conv2d(self.in_feat_channels, feat_dim, kernel_size=1).to(self.device)
                self.medk2n_modules['adapter'] = adapter
                print(f"🔧 [MedK2N] Created feature channel adapter: {self.in_feat_channels} -> {feat_dim}")
            except Exception as _e:
                print(f"⚠️ Unable to create adapter: {_e}")
        
        # 5. TaskHeadNet - Task-specific generation
        self.medk2n_modules['task_head'] = MedK2N_TaskHeadNet(
            input_dim=feat_dim, context_dim=embed_dim
        ).to(self.device)
        
        # 6. QualityFeedbackNet - Quality feedback network
        self.medk2n_modules['quality_feedback'] = MedK2N_QualityFeedbackNet(
            num_tasks=num_tasks, embed_dim=embed_dim
        ).to(self.device)
        
        print(f"✅ [MedK2N] Six core modules initialized: {list(self.medk2n_modules.keys())}")
        
    def get_all_parameters(self):
        """Get all MedK2N module parameters for optimizer"""
        params = []
        for module_name, module in self.medk2n_modules.items():
            params.extend(list(module.parameters()))
        return params
        
    def forward_pipeline(self, base_features, enhance_features, task_embeddings, prev_quality=None):
        """MedK2N complete forward pipeline"""
        B = base_features.size(0)
        embed_dim = MEDK2N_CONFIG['embed_dim']
        
        
        modality_embed = torch.randn(B, embed_dim).to(self.device)
        task_embed = task_embeddings if task_embeddings is not None else torch.randn(B, embed_dim).to(self.device)
        
        
        if len(self.quality_history) > 0:
            quality_history = torch.stack(self.quality_history[-10:], dim=1) if len(self.quality_history) >= 10 else torch.randn(B, 10, 32).to(self.device)
        else:
            quality_history = torch.randn(B, 10, 32).to(self.device)
            
        
        prev_outputs = torch.randn(B, 16).to(self.device)
        task_compatibility = torch.randn(B, 1).to(self.device)
        
        
        feat_dim = MEDK2N_CONFIG['feat_dim']
        if 'adapter' in self.medk2n_modules:
            try:
                adapter = self.medk2n_modules['adapter']
                if base_features.shape[1] != feat_dim:
                    base_features = adapter(base_features)
                if enhance_features.shape[1] != feat_dim:
                    enhance_features = adapter(enhance_features)
            except Exception:
                pass

        
        w_global = self.medk2n_modules['preweight'](
            base_features, enhance_features, prev_outputs,
            quality_history, task_compatibility, modality_embed
        )
        
        
        compat_matrix = torch.randn(B, embed_dim).to(self.device)
        perf_tracker = torch.randn(B, 5, 32).to(self.device)
        context_embedding = torch.randn(B, embed_dim).to(self.device)
        
        tau = self.medk2n_modules['threshold'](
            modality_embed, task_embed, compat_matrix,
            perf_tracker, context_embedding, w_global, w_global
        )
        
        
        hist_perf = torch.randn(B, 16).to(self.device)
        uncertainty = torch.randn(B, 1).to(self.device)
        quality_ind = torch.randn(B, 1).to(self.device)
        
        effective_weight, weight_components = self.medk2n_modules['effe_weight'](
            w_global, tau, task_embed, modality_embed,
            hist_perf, uncertainty, quality_ind
        )
        
        fused_features = self.medk2n_modules['res_fusion'](
            enhance_features, base_features, effective_weight
        )
     
        output_result = self.medk2n_modules['task_head'](fused_features, task_embed, prev_quality)
    
        return {
            'fused_features': fused_features,
            'output': output_result['output'],
            'quality_score': output_result['quality_score'],
            'quality_improvement': output_result['quality_improvement'],
            'effective_weight': effective_weight,
            'weight_components': weight_components,
            'feedback': output_result['feedback']
        }
        
    def compute_quality_feedback(self, current_outputs, previous_outputs, effective_weight_matrix):
        """Compute quality feedback"""
        return self.medk2n_modules['quality_feedback'](
            current_outputs, previous_outputs, effective_weight_matrix
        )
        
    def update_quality_history(self, quality_score):
        """Update quality history"""
        if isinstance(quality_score, torch.Tensor):
            self.quality_history.append(quality_score.mean().detach())
        if len(self.quality_history) > 50:  # Maintain history length
            self.quality_history = self.quality_history[-50:]
            
    def get_medk2n_loss(self, medk2n_output, target, base_loss):
        """Calculate MedK2N specific loss"""
        quality_loss = F.mse_loss(medk2n_output['output'], target)
        
        # Quality improvement reward
        quality_improvement = medk2n_output['quality_improvement'].mean()
        quality_reward = torch.clamp(quality_improvement, -1.0, 1.0)
        
        
        weight_reg = torch.mean(torch.abs(medk2n_output['effective_weight'] - 0.5))
        
        total_loss = quality_loss - 0.1 * quality_reward + 0.01 * weight_reg
        
        return {
            'total_loss': total_loss,
            'quality_loss': quality_loss,
            'quality_reward': quality_reward,
            'weight_reg': weight_reg,
            'quality_score': medk2n_output['quality_score'].mean(),
        }


from semseg.losses.accurate_loss import AccurateLoss, AccurateMetrics
class ImageSaver:
    """Simple image saver for sample images per epoch"""
    
    def __init__(self, save_dir, rank=0, save_every_batches: int = 100,
                 enabled: bool = True,
                 max_train_samples: int = 2,
                 max_val_samples: int = 4,
                 downscale: int = 2,
                 save_diffs: bool = False):
        self.rank = rank
        self.save_dir = save_dir
        self.enabled = bool(enabled)
        self.max_train_samples = int(max(1, max_train_samples))
        self.max_val_samples = int(max(1, max_val_samples))
        self.downscale = int(max(1, downscale))
        self.save_diffs = bool(save_diffs)
        if rank == 0:
            
            self.train_img_dir = os.path.join(save_dir, 'training_images')
            self.val_img_dir = os.path.join(save_dir, 'validation_images')
            os.makedirs(self.train_img_dir, exist_ok=True) 
            os.makedirs(self.val_img_dir, exist_ok=True)
            self.save_every_batches = int(save_every_batches)
            if self.enabled:
                print(f"📸 Images will be saved to: {self.train_img_dir} and {self.val_img_dir} (every {self.save_every_batches} batches; downscale x{self.downscale})")
            else:
                print(f"📷 Training/validation image saving disabled (use --enable-image-saving to enable)")
            
    
    def save_training_samples(self, inputs, target, pred, epoch, batch_idx, max_samples=None):
        """Save training sample images"""
        
        if inputs is None or target is None or pred is None:
            return
        
        # Save training images based on batch interval
        if self.rank == 0:
            
            pass

        if not self.enabled:
            return
        if self.rank != 0 or (hasattr(self, 'save_every_batches') and (batch_idx % int(self.save_every_batches) != 0)):
            return
            
        try:
            with torch.no_grad():
                
                if not hasattr(target, 'size'):
                    print(f"⚠️ Target object has no size attribute, skipping save - type: {type(target)}")
                    return
                    
                max_s = int(self.max_train_samples if max_samples is None else max_samples)
                num_samples = min(max_s, target.size(0))

                
                if isinstance(inputs, list):
                    
                    input_images = []
                    for modal_idx, inp in enumerate(inputs):
                        if inp.size(0) >= num_samples:
                            modal_img = inp[:num_samples].cpu()
                            if modal_img.shape[1] > 1:
                                modal_img = modal_img.mean(dim=1, keepdim=True)
                            input_images.append(modal_img)
                else:
                    input_img = inputs[:num_samples].cpu()
                    if input_img.shape[1] > 1:
                        input_img = input_img.mean(dim=1, keepdim=True)
                    input_images = [input_img]
                
                target_img = target[:num_samples].cpu()
                pred_img = pred[:num_samples].cpu()

                if target_img.shape[1] > 1:
                    target_img = target_img.mean(dim=1, keepdim=True)
                if pred_img.shape[1] > 1:
                    pred_img = pred_img.mean(dim=1, keepdim=True)
                
                def normalize_img(img):
                    img = img.float()
                    img_min = float(img.min())
                    img_max = float(img.max())
                    
                    if img_min >= -0.01 and img_max <= 1.01:
                        return torch.clamp(img, 0.0, 1.0)

                    if img_max > img_min:
                        normalized = (img - img_min) / (img_max - img_min)
                    else:
                        normalized = img
                    return torch.clamp(normalized, 0.0, 1.0)
                
                target_img = normalize_img(target_img)
                pred_img = normalize_img(pred_img)

                if self.downscale > 1:
                    try:
                        def _ds_t(t):
                            N, C, H, W = t.shape
                            H2, W2 = max(1, H // self.downscale), max(1, W // self.downscale)
                            return F.interpolate(t, size=(H2, W2), mode='bilinear', align_corners=False)
                        target_img = _ds_t(target_img)
                        pred_img = _ds_t(pred_img)
                        if isinstance(input_images, list) and len(input_images) > 0:
                            input_images = [_ds_t(x) for x in input_images]
                    except Exception:
                        pass
                
                for i in range(num_samples):

                    fig, axes = plt.subplots(1, len(input_images) + 2, figsize=(4 * (len(input_images) + 2), 4))
                    
                    for modal_idx, inp_img in enumerate(input_images):
                        normalized_inp = normalize_img(inp_img[i])
                        img_array = normalized_inp.squeeze().numpy().astype(np.float32)
                        axes[modal_idx].imshow(img_array, cmap='gray', vmin=0, vmax=1)
                        axes[modal_idx].set_title(f'Input Modal {modal_idx + 1}')
                        axes[modal_idx].axis('off')

                    target_array = target_img[i].squeeze().numpy().astype(np.float32)
                    axes[-2].imshow(target_array, cmap='gray', vmin=0, vmax=1)
                    axes[-2].set_title('Target')
                    axes[-2].axis('off')

                    pred_array = pred_img[i].squeeze().numpy().astype(np.float32)
                    axes[-1].imshow(pred_array, cmap='gray', vmin=0, vmax=1)
                    axes[-1].set_title('Prediction')
                    axes[-1].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.train_img_dir, f'epoch_{epoch:03d}_batch_{batch_idx:04d}_sample_{i}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                
                if epoch % 5 == 0 or batch_idx % 500 == 0: 
                    print(f"📸 Save training images: epoch {epoch}, batch {batch_idx}, {num_samples} samples")
        except Exception as e:
            print(f"⚠️ Training image save error: {e}")
    
    def save_validation_samples(self, inputs, target, pred, epoch, max_samples=None):
        if inputs is None or target is None or pred is None:
            return
            
        if not self.enabled:
            return
        if self.rank != 0:
            return
            
        try:
            with torch.no_grad():
                if not hasattr(target, 'size'):
                    print(f"⚠️ targetsize， - type: {type(target)}")
                    return
                    
                max_s = int(self.max_val_samples if max_samples is None else max_samples)
                num_samples = min(max_s, target.size(0))
                
                if isinstance(inputs, list):
                    input_images = []
                    for inp in inputs:
                        if inp.size(0) >= num_samples:
                            modal_img = inp[:num_samples].cpu()
                            if modal_img.shape[1] > 1:
                                modal_img = modal_img.mean(dim=1, keepdim=True)
                            input_images.append(modal_img)
                else:
                    input_img = inputs[:num_samples].cpu()
                    if input_img.shape[1] > 1:
                        input_img = input_img.mean(dim=1, keepdim=True)
                    input_images = [input_img]
                
                target_img = target[:num_samples].cpu()
                pred_img = pred[:num_samples].cpu()
                
                if target_img.shape[1] > 1:
                    target_img = target_img.mean(dim=1, keepdim=True)
                if pred_img.shape[1] > 1:
                    pred_img = pred_img.mean(dim=1, keepdim=True)
                
                def normalize_img(img):
                    img = img.float()
                    img_min = float(img.min())
                    img_max = float(img.max())
                    
                    if img_min >= -0.01 and img_max <= 1.01:
                        return torch.clamp(img, 0.0, 1.0)
                    
                    if img_max > img_min:
                        normalized = (img - img_min) / (img_max - img_min)
                    else:
                        normalized = img
                    return torch.clamp(normalized, 0.0, 1.0)
                
                target_img = normalize_img(target_img)
                pred_img = normalize_img(pred_img)

                if self.downscale > 1:
                    try:
                        def _ds_t(t):
                            N, C, H, W = t.shape
                            H2, W2 = max(1, H // self.downscale), max(1, W // self.downscale)
                            return F.interpolate(t, size=(H2, W2), mode='bilinear', align_corners=False)
                        target_img = _ds_t(target_img)
                        pred_img = _ds_t(pred_img)
                        if isinstance(input_images, list) and len(input_images) > 0:
                            input_images = [_ds_t(x) for x in input_images]
                    except Exception:
                        pass
                
                rows = 2
                cols = 4
                fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
                
                for i in range(min(num_samples, rows * cols // 2)):
                    row = i // 2
                    col = (i % 2) * 2
                    
                    target_array = target_img[i].squeeze().numpy().astype(np.float32)
                    axes[row, col].imshow(target_array, cmap='gray', vmin=0, vmax=1)
                    axes[row, col].set_title(f'Target {i+1}')
                    axes[row, col].axis('off')
                    
                    pred_array = pred_img[i].squeeze().numpy().astype(np.float32)
                    axes[row, col+1].imshow(pred_array, cmap='gray', vmin=0, vmax=1)
                    axes[row, col+1].set_title(f'Prediction {i+1}')
                    axes[row, col+1].axis('off')
                
                for i in range(num_samples, rows * cols // 2):
                    row = i // 2
                    col = (i % 2) * 2
                    axes[row, col].axis('off')
                    axes[row, col+1].axis('off')
                
                plt.suptitle(f'Validation Results - Epoch {epoch}', fontsize=16)
                plt.tight_layout()
                
                save_path = os.path.join(self.val_img_dir, f'validation_epoch_{epoch:03d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                if self.save_diffs:
                    diff_img = torch.abs(pred_img - target_img)
                    diff_img = normalize_img(diff_img)
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    for i in range(min(num_samples, 8)):
                        row = i // 4
                        col = i % 4
                        diff_array = diff_img[i].squeeze().numpy().astype(np.float32)
                        axes[row, col].imshow(diff_array, cmap='hot', vmin=0, vmax=1)
                        axes[row, col].set_title(f'Difference {i+1}')
                        axes[row, col].axis('off')
                    plt.suptitle(f'Prediction Differences - Epoch {epoch}', fontsize=16)
                    plt.tight_layout()
                    save_path = os.path.join(self.val_img_dir, f'differences_epoch_{epoch:03d}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                
                if epoch % 5 == 0:  
                    print(f"📸 Save validation images: epoch {epoch}, {num_samples} samples")
                
        except Exception as e:
            print(f"⚠️ Validation image save error: {e}")

class MemoryManager:

    def __init__(self, device):
        self.device = device
        self.cleanup_threshold = 0.70 
        
    def get_memory_info(self):
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(self.device) / 1024**3     # GB
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
        utilization = allocated / total
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'utilization': utilization
        }
    
    def cleanup_if_needed(self, force=False):
        memory_info = self.get_memory_info()
        
        if force or memory_info['utilization'] > self.cleanup_threshold:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
        return False
    
    def emergency_cleanup(self):
        print("🚨 Emergency memory cleanup...")
        for _ in range(5):  # 增加清理次数
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import time
            time.sleep(0.05)
        
        try:
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        except:
            pass


def calculate_medical_metrics(pred, target):
    with torch.no_grad():
        psnr = AccurateMetrics.calculate_psnr(pred, target, data_range=1.0)
        ssim_value = AccurateMetrics.calculate_ssim(pred, target, data_range=1.0)
        
    return {
        'psnr': psnr, 
        'ssim': ssim_value
    }

def _resize_pred_to_target(pred, target):
        try:
            if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
                if pred.dim() >= 3 and target.dim() >= 3:
                    if pred.shape[0] != target.shape[0]:
                        min_batch = min(pred.shape[0], target.shape[0])
                        pred = pred[:min_batch]
                        target = target[:min_batch]
                    
                    if pred.shape[2:] != target.shape[2:]:
                        with torch.cuda.device(pred.device):
                            available_mem = torch.cuda.get_device_properties(pred.device).total_memory
                            current_mem = torch.cuda.memory_allocated(pred.device)
                            required_mem = pred.numel() * 4 * 2  
                            
                            if current_mem + required_mem > available_mem * 0.9:
                                smaller_size = [s // 2 for s in target.shape[2:]]
                                pred_small = F.interpolate(pred, size=smaller_size, mode='bilinear', align_corners=False)
                                torch.cuda.empty_cache()
                                return F.interpolate(pred_small, size=target.shape[2:], mode='bilinear', align_corners=False)
                            else:
                                return F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        except Exception as e:
            print(f"[Warn] _resize_pred_to_target failed: {e}")
            torch.cuda.empty_cache()
        return pred

# Task type detection function removed - not needed for production environment

def create_run_name(input_modals, target_modal):
    """Create run name"""
    modals_str = ''.join(input_modals)
    if target_modal:
        return f"{modals_str}2{target_modal}"
    else:
        return f"{modals_str}_to_auto"

# parse_task_specification function kept but no longer uses inline string import
def parse_task_specification(task_str):
    """Parse task specification, e.g., A2B, ABC2D, ABCD2ABCD

    Returns:
      (input_modals: list[str], target_modals: list[str])
    """
    if not task_str or '2' not in task_str:
        return None, None

    parts = task_str.split('2')
    if len(parts) != 2:
        return None, None

    left, right = parts[0].strip(), parts[1].strip()
    if not left or not right:
        return None, None

    input_modals = [c for c in left if c.isupper() and c.isalpha()]
    target_modals = [c for c in right if c.isupper() and c.isalpha()]
    if len(input_modals) == 0 or len(target_modals) == 0:
        return None, None

    return input_modals, target_modals

def get_clip_weight_by_epoch(epoch, total_epochs):

    progress = epoch / max(total_epochs - 1, 1)
    
    if progress < 0.3:  
        weight = 0.05
    elif progress < 0.7:  
        weight = 0.1
    else:  
        weight = 0.2
    
    return weight

def adjust_clip_loss_weight(criterion, epoch, total_epochs, rank=0):
    if not CLIP_AVAILABLE:
        return
    
    new_weight = get_clip_weight_by_epoch(epoch, total_epochs)
    
    if hasattr(criterion, 'clip_weight'):
        old_weight = criterion.clip_weight
        criterion.clip_weight = new_weight
        
        if rank == 0 and abs(new_weight - old_weight) > 0.01:
            print(f"📊  - CLIP: {old_weight:.3f} → {new_weight:.3f} (Epoch {epoch+1}/{total_epochs})")
    elif hasattr(criterion, 'weights') and 'clip' in criterion.weights:
        old_weight = criterion.weights['clip']
        criterion.weights['clip'] = new_weight
        
        if rank == 0 and abs(new_weight - old_weight) > 0.01:
            print(f"📊  - CLIP: {old_weight:.3f} → {new_weight:.3f} (Epoch {epoch+1}/{total_epochs})")
    else:
        return

def create_enhanced_criterion_with_clip(args, device, dataset_cfg: Optional[dict] = None):
   
    from semseg.losses.accurate_loss import AccurateLoss
    
    if hasattr(args, 'rank') and args.rank == 0:
        print(f"🔧 :")
        print(f"   : {'' if getattr(args, 'enable_curriculum', False) else ''}")
    
    loss_cfg = {}
    clip_cfg = {}
    try:
        cfg_path = getattr(args, 'cfg', None)
        if cfg_path and os.path.exists(cfg_path):
            import yaml
            with open(cfg_path, 'r') as _f:
                _full = yaml.safe_load(_f) or {}
            loss_cfg = _full.get('LOSS', {}) or {}
            clip_cfg = _full.get('CLIP', {}) or {}
    except Exception:
        loss_cfg = {}
        clip_cfg = {}

    lambda_weighted_l1 = float(loss_cfg.get('lambda_weighted_l1', 6.0))    
    lambda_ssim = float(loss_cfg.get('lambda_ssim', 4.0))                   # Reduce: avoid over-optimization of structure
    
    # Perceptual loss (new: improve visual quality)
    lambda_perceptual = float(loss_cfg.get('lambda_perceptual', 3.0))       # New: VGG feature constraint
    
    # Semantic and progressive fusion loss (Med-K2N specific, increase weight)
    lambda_clip_total = float(loss_cfg.get('lambda_clip_total', 0.8))       # Increase: fully utilize modality semantics
    lambda_progressive_fusion = float(loss_cfg.get('lambda_progressive_fusion', 3.0))  # Increase: progressive fusion constraint
    lambda_causal_consistency = float(loss_cfg.get('lambda_causal_consistency', 2.0))  # Increase: causal consistency
    
    # Quality-aware loss (work with QualityFeedbackNet, increase weight)
    lambda_quality_aware = float(loss_cfg.get('lambda_quality_aware', 1.5))  # Increase: quality-aware loss
    lambda_modality_identity = float(loss_cfg.get('lambda_modality_identity', 1.2))  # Increase: modality identity loss
    
    # Auxiliary regularization loss (moderate increase)
    lambda_grad = float(loss_cfg.get('lambda_grad', 0.2))                   # Increase: edge preservation
    lambda_consistency = float(loss_cfg.get('lambda_consistency', 0.4))     # Increase: multi-frame consistency
    lambda_lesion_aware = float(loss_cfg.get('lambda_lesion_aware', 0.3))   # Increase: lesion awareness
    lambda_tv = float(loss_cfg.get('lambda_tv', 0.1))                       # Increase: smoothness constraint


    # Create basic loss function - Med-K2N enhanced version + perceptual loss
    criterion = AccurateLoss(
        device=device,
        # Basic reconstruction loss
        lambda_weighted_l1=lambda_weighted_l1,
        lambda_ssim=lambda_ssim,
        # Perceptual loss (new)
        lambda_perceptual=lambda_perceptual,
        # Auxiliary regularization loss
        lambda_grad=lambda_grad,
        lambda_consistency=lambda_consistency,
        lambda_lesion_aware=lambda_lesion_aware,
        lambda_tv=lambda_tv
    )

    # Print Med-K2N loss weight configuration (updated version)
    try:
        rank = int(os.environ.get('RANK', 0))
    except Exception:
        rank = 0
    if rank == 0:
        print("🎯 Med-K2N Loss Weight Configuration (Visual Quality Optimized):")
        print(f"   Basic Reconstruction: L1={lambda_weighted_l1:.1f}, SSIM={lambda_ssim:.1f}")
        print(f"   Perceptual Loss: Perceptual={lambda_perceptual:.1f} (VGG features)")
        print(f"   Med-K2N Core: Progressive={lambda_progressive_fusion:.1f}, Causal={lambda_causal_consistency:.1f}")
        print(f"   Quality Aware: Quality={lambda_quality_aware:.1f}, Identity={lambda_modality_identity:.1f}")
        print(f"   Semantic Understanding: CLIP={lambda_clip_total:.1f}")
        print(f"   Auxiliary Regularization: Grad={lambda_grad:.1f}, Consistency={lambda_consistency:.1f}")
        total_main_weight = lambda_weighted_l1 + lambda_ssim + lambda_perceptual + lambda_progressive_fusion + lambda_causal_consistency
        print(f"   📊 Main Weights: {total_main_weight:.1f} (L1+SSIM+Perceptual+Progressive+Causal)")
        
        # Updated weight balance check
        if total_main_weight > 25:
            print("⚠️  Warning: Main loss weights sum too high, may cause training instability")
        elif total_main_weight < 10:
            print("⚠️  : ，")
        else:
            print("✅ ")
    
    if CLIP_AVAILABLE and hasattr(args, 'use_clip_loss') and args.use_clip_loss:
        try:
            dataset_root = None
            if isinstance(dataset_cfg, dict):
                dataset_root = dataset_cfg.get('ROOT', None)
            
            if not dataset_root:
                raise RuntimeError("(DATASET.ROOT)")
                
            clip_integrator = CLIPLossIntegrator(device=device, dataset_root=dataset_root)
            
            source_letters = None
            target_letters = None
            if isinstance(dataset_cfg, dict):
                try:
                    src_modals = dataset_cfg.get('MODALS', None)
                    tgt_modals = dataset_cfg.get('TARGET_MODAL', None)
                    if isinstance(src_modals, (list, tuple)) and len(src_modals) > 0:
                        source_letters = list(src_modals)
                    elif isinstance(src_modals, str) and src_modals:
                        source_letters = [src_modals]
                    if isinstance(tgt_modals, (list, tuple)) and len(tgt_modals) > 0:
                        target_letters = list(tgt_modals)
                    elif isinstance(tgt_modals, str) and tgt_modals:
                        target_letters = [tgt_modals]
                except Exception:
                    pass
            
            initial_clip_weight = float(clip_cfg.get('weight', lambda_clip_total))
            enhanced_criterion = clip_integrator.integrate_with_existing_loss(
                criterion, initial_clip_weight, target_letters, source_letters
            )
            
            if rank == 0:
                print(f"✅ CLIP (: {initial_clip_weight})")
            return enhanced_criterion
            
        except Exception as e:
            print(f"⚠️ CLIP: {e}, continuing without CLIP loss.")
            
            return criterion
    else:
        return criterion

def _parse_stage_ratios(ratios_str: str):
    try:
        parts = [float(x.strip()) for x in ratios_str.split(',') if x.strip() != '']
        if len(parts) != 4:
            return [0.25, 0.25, 0.25, 0.25]
        s = sum(parts)
        if s <= 0:
            return [0.25, 0.25, 0.25, 0.25]
        return [p / s for p in parts]
    except Exception:
        return [0.25, 0.25, 0.25, 0.25]

def get_curriculum_stage(epoch: int, total_epochs: int, ratios: list[float]):
    e = epoch
    n = max(total_epochs, 1)
    r = ratios if ratios and len(ratios) == 4 else [0.25, 0.25, 0.25, 0.25]
    
    b1 = int(n * r[0])
    b2 = b1 + int(n * r[1])
    b3 = b2 + int(n * r[2])
    
    if e < b1:
        return 'easy'  
    elif e < b2:
        return 'medium' 
    elif e < b3:
        return 'hard'   
    else:
        return 'expert' 

def apply_curriculum_to_batch(inputs, target, stage: str, epoch: int, batch_idx: int, rank: int):

    inputs_list = inputs if isinstance(inputs, (list, tuple)) else [inputs]
    target_list = target if isinstance(target, (list, tuple)) else [target]
    k_total = len(inputs_list)
    n_total = len(target_list)

    seed = (epoch + 1) * 100003 + (batch_idx + 1) * 97 + (rank + 7)
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)

    def _rand_choice(choices):
        if not choices:
            return choices[0] if choices else 2  # 默认返回2
        idx = int(torch.randint(0, len(choices), (1,), generator=g).item())
        return choices[idx]

    def _rand_subset(total, size):
        size = max(0, min(size, total))
        if size == 0:
            return []
        if size >= total:
            return list(range(total))
        perm = torch.randperm(total, generator=g).tolist()
        return sorted(perm[:size])

    def _choose_non_overlapping_targets(num_targets, input_indices):
        forbidden = set(input_indices)
        available = [i for i in range(n_total) if i not in forbidden]
        if len(available) >= num_targets:
            selected_indices = _rand_subset(len(available), num_targets)
            selected_targets = [available[i] for i in selected_indices]
            return selected_indices, selected_targets
        else:
            selected = available[:]
            remaining = [i for i in range(n_total) if i not in selected]
            needed = num_targets - len(selected)
            if needed > 0 and remaining:
                additional_indices = _rand_subset(len(remaining), min(needed, len(remaining)))
                selected.extend([remaining[i] for i in additional_indices])
            return list(range(len(selected))), selected

    if stage == 'easy' or stage == '1to1':
        k, t = 1, 1
        input_indices = _rand_subset(k_total, k)
        forbidden = set(input_indices)
        target_candidates = [i for i in range(n_total) if i not in forbidden]
        if target_candidates:
            target_indices = [_rand_choice(target_candidates)]
        else:
            target_indices = [_rand_subset(n_total, 1)[0] if n_total > 0 else 0]
        
        inputs_sel = [inputs_list[i] for i in input_indices]
        target_sel = [target_list[target_indices[0]]]
        selected_target_idx = target_indices[0]
        pattern = '1->1'

    elif stage == 'medium' or stage == 'kto1':
        k = min(_rand_choice([2, 3]) or 2, k_total)
        t = 1
        input_indices = _rand_subset(k_total, k)
        _, target_indices = _choose_non_overlapping_targets(t, input_indices)
        target_indices = target_indices[:1]
        
        inputs_sel = [inputs_list[i] for i in input_indices]
        target_sel = [target_list[target_indices[0]]]
        selected_target_idx = target_indices[0]
        pattern = f'{k}->1'

    elif stage == 'hard' or stage == '1tok':
        k = 1
        t = min(_rand_choice([2, 3]) or 2, n_total)
        input_indices = _rand_subset(k_total, k)
        _, target_indices = _choose_non_overlapping_targets(t, input_indices)
        
        inputs_sel = [inputs_list[i] for i in input_indices]
        target_full = [None for _ in range(n_total)]
        for j in target_indices:
            if 0 <= j < n_total:
                target_full[j] = target_list[j]
        target_sel = target_full
        selected_target_idx = None
        pattern = f'1->{t}'

    else:  # 'expert' or 'kton'
        k = min(_rand_choice([2, 3]) or 2, k_total)
        t = min(_rand_choice([2, 3]) or 2, n_total)
        
        max_attempts = 10
        for attempt in range(max_attempts):
            input_indices = _rand_subset(k_total, k)
            
            available_targets = [i for i in range(n_total) if i not in input_indices]
            
            if len(available_targets) >= t:
                target_indices = _rand_subset(len(available_targets), t)
                target_indices = [available_targets[i] for i in target_indices]
                break
            elif len(available_targets) > 0:
                t = len(available_targets)
                target_indices = available_targets[:]
                break
            else:
                k = max(1, k - 1)
                if k == 0:
                    k, t = 1, 1
                    input_indices = _rand_subset(k_total, 1)
                    available = [i for i in range(n_total) if i not in input_indices]
                    target_indices = [available[0]] if available else [0]
                    break
        
        inputs_sel = [inputs_list[i] for i in input_indices]
        if t == 1:
            target_sel = [target_list[target_indices[0]]]
            selected_target_idx = target_indices[0]
        else:
            target_full = [None for _ in range(n_total)]
            for j in target_indices:
                if 0 <= j < n_total:
                    target_full[j] = target_list[j]
            target_sel = target_full
            selected_target_idx = None
        pattern = f'{k}->{t}'

    sel_info = {
        'input_indices': input_indices,
        'target_indices': target_indices,
        'pattern': pattern
    }
    return inputs_sel, target_sel, selected_target_idx, sel_info

def train_one_epoch_optimized(model, dataloader, criterion, optimizer, scheduler, device, 
                             epoch, memory_manager, rank, image_saver=None, writer=None,
                             lambda_distill: float = 0.5, max_batches: Optional[int] = None,
                             enable_curriculum: bool = False, stage: Optional[str] = None,
                             enable_modality_dropout: bool = False,
                             modal_dropout_p_same: float = 0.0,
                             modal_dropout_p_default: float = 0.0,
                             avoid_identity_in_consistency: bool = False,
                             medk2n_manager: Optional[MedK2N_TrainingManager] = None,
                             gradient_accumulation_steps: int = 1):
    """内存优化的训练函数"""
    model.train()
    total_loss = 0.0
    total_metrics = {
        'psnr': 0.0, 'ssim': 0.0
    }  # 添加训练指标统计
    num_batches = len(dataloader)
    processed_batches = 0  # 实际处理的batch数（考虑max_batches提前停止）
    metrics_batches = 0    # 实际进行过指标计算的batch数
    
    loss_stats = {'total': 0.0, 'L_causal': 0.0, 'L_metric': 0.0, 'count': 0}
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    else:
        pbar = dataloader
    
    scaler = GradScaler() 

    # Helper: safely move tensors (or nested lists/tuples) to device
    def _safe_to_device(x, device, non_blocking=True):
        import torch as _torch
        if x is None:
            return None
        if isinstance(x, _torch.Tensor):
            try:
                return x.to(device, non_blocking=non_blocking)
            except Exception:
                # fall back to CPU->device without non_blocking
                return x.to(device)
        if isinstance(x, (list, tuple)):
            lst = [_safe_to_device(xx, device, non_blocking=non_blocking) for xx in x]
            return type(x)(lst)
        # unknown type: return as-is
        return x
    
    use_curriculum = bool(enable_curriculum)
    cur_stage = stage or 'kton'
    for batch_idx, batch in enumerate(pbar):
        try:
            if batch_idx % 10 == 0:
                memory_manager.cleanup_if_needed()
            
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, target = batch
                inputs = _safe_to_device(inputs, device, non_blocking=True)
                if isinstance(target, (list, tuple)):
                    target = _safe_to_device(list(target), device, non_blocking=True)
                else:
                    target = _safe_to_device(target, device, non_blocking=True)
            else:
                if isinstance(batch, dict):
                    inputs_list = batch.get('inputs', None)
                    target_tensor = batch.get('target', None)
                    if inputs_list is None or target_tensor is None:
                        inputs, target = batch  # 回退
                    else:
                        inputs = _safe_to_device(inputs_list, device, non_blocking=True)
                        if isinstance(target_tensor, (list, tuple)):
                            target = _safe_to_device(list(target_tensor), device, non_blocking=True)
                        else:
                            target = _safe_to_device(target_tensor, device, non_blocking=True)
                else:
                    inputs, target = batch
                    inputs = _safe_to_device(inputs, device, non_blocking=True)
                    if isinstance(target, (list, tuple)):
                        target = _safe_to_device(list(target), device, non_blocking=True)
                    else:
                        target = _safe_to_device(target, device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            cur_inputs = inputs
            cur_target = target
            selected_pred_idx = None
            sel_info = None
            if use_curriculum:
                cur_inputs, tgt_list, selected_pred_idx, sel_info = apply_curriculum_to_batch(inputs, target, cur_stage, epoch, batch_idx, rank)
                if isinstance(tgt_list, (list, tuple)) and len(tgt_list) == 1:
                    cur_target = tgt_list[0]
                    selected_pred_idx = selected_pred_idx if selected_pred_idx is not None else 0
                else:
                    cur_target = tgt_list

            if enable_modality_dropout:
                try:
                    dataset = getattr(dataloader, 'dataset', None)
                    input_letters = list(getattr(dataset, 'input_modals', [])) if dataset is not None else []
                    target_letters = list(getattr(dataset, 'target_modals', [])) if dataset is not None else []
                    if sel_info and isinstance(sel_info, dict) and 'target_indices' in sel_info:
                        tgt_indices = list(sel_info['target_indices'])
                    else:
                        tgt_indices = list(range(len(target_letters)))
                    same_modal_inputs = set()
                    for tidx in tgt_indices:
                        if 0 <= tidx < len(target_letters):
                            t_letter = target_letters[tidx]
                            for i, in_letter in enumerate(input_letters):
                                if in_letter == t_letter:
                                    same_modal_inputs.add(i)
                    for i in range(len(cur_inputs)):
                        p = float(modal_dropout_p_same) if i in same_modal_inputs else float(modal_dropout_p_default)
                        if p > 0.0 and torch.rand(1, device=cur_inputs[i].device).item() < p:
                            cur_inputs[i] = torch.zeros_like(cur_inputs[i])
                except Exception as _e:
                    if rank == 0 and batch_idx == 0:
                        print(f"[Warn] Modality dropout skipped due to: {_e}")

            with autocast():
                out_gen, out_feat, out_dict = model(cur_inputs, multimask_output=True)
                preds = out_gen if isinstance(out_gen, list) else [out_gen]
                if all(p is None for p in preds):
                    try:
                        gen_from_dict = None
                        if isinstance(out_dict, dict):
                            gen_from_dict = out_dict.get('generated_rgb_frames', None)
                        if gen_from_dict is not None:
                            preds = gen_from_dict if isinstance(gen_from_dict, list) else [gen_from_dict]
                    except Exception:
                        pass
                
                if isinstance(preds, list) and len(preds) > 1:
                    min_h = min(p.shape[2] for p in preds if p is not None and len(p.shape) >= 3)
                    min_w = min(p.shape[3] for p in preds if p is not None and len(p.shape) >= 4)
                    target_size = (min_h, min_w)
                    
                    for i, p in enumerate(preds):
                        if p is not None and p.shape[2:] != target_size:
                            preds[i] = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
                
                if isinstance(cur_target, list):
                    pair_count = min(len(preds), len(cur_target))
                    total_task_loss = 0.0
                    task_losses_acc = {}
                    valid_pairs = 0
                    for i in range(pair_count):
                        p_i = preds[i]
                        t_i = cur_target[i]
                        if not (torch.is_tensor(p_i) and torch.is_tensor(t_i)):
                            continue
                        if torch.is_tensor(p_i) and torch.is_tensor(t_i):
                            preds[i] = _resize_pred_to_target(p_i, t_i)
                        if isinstance(cur_inputs, (list, tuple)) and len(cur_inputs) > 0:
                            _idx = i % len(cur_inputs)
                            if avoid_identity_in_consistency and len(cur_inputs) > 1:
                                _idx = (_idx + 1) % len(cur_inputs)
                            input_for_pair = [cur_inputs[_idx]]
                        else:
                            input_for_pair = cur_inputs
                        li, li_dict = criterion(preds[i], cur_target[i], input_for_pair)
                        total_task_loss = total_task_loss + li
                        for k, v in li_dict.items():
                            task_losses_acc[k] = task_losses_acc.get(k, 0.0) + float(v.item())
                        valid_pairs += 1
                    denom = max(1, valid_pairs)
                    task_loss = total_task_loss / denom
                    task_losses = {k: torch.tensor(v / denom, device=device) for k, v in task_losses_acc.items()}
                    pred_vis = next((p for p in reversed(preds) if p is not None), preds[0] if len(preds) > 0 else None)
                else:
    
                    pred_idx = selected_pred_idx if (selected_pred_idx is not None and 0 <= int(selected_pred_idx) < len(preds)) else (len(preds)-1)
          
                    if len(preds) > 0 and torch.is_tensor(preds[pred_idx]) and torch.is_tensor(cur_target):
                        preds[pred_idx] = _resize_pred_to_target(preds[pred_idx], cur_target)
   
                    if isinstance(cur_inputs, (list, tuple)) and len(cur_inputs) > 0:
                        _idx = pred_idx % len(cur_inputs)
                        if avoid_identity_in_consistency and len(cur_inputs) > 1:
                            _idx = (_idx + 1) % len(cur_inputs)
                        input_for_pair = [cur_inputs[_idx]]
                    else:
                        input_for_pair = cur_inputs
                    task_loss, task_losses = criterion(preds[pred_idx], cur_target, input_for_pair)
                    pred_vis = preds[pred_idx]

                loss = task_loss
                

                losses = dict(task_losses)

                if hasattr(loss, 'item'):
                    loss_stats['total'] += loss.item()
                else:
                    loss_stats['total'] += float(loss)
                loss_stats['count'] += 1
                for key in ['L_causal', 'L_metric']:
                    if key in losses:
                        if hasattr(losses[key], 'item'):
                            loss_stats[key] += losses[key].item()
                        else:
                            loss_stats[key] += float(losses[key])
            
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    if isinstance(cur_target, list):
                        pair_count = min(len(preds), len(cur_target))
                        metrics_sum = {key: 0.0 for key in total_metrics.keys()}
                        valid = 0
                        for i in range(pair_count):
                            if preds[i] is None or cur_target[i] is None:
                                continue
                            m = calculate_medical_metrics(preds[i], cur_target[i])
                            for key in metrics_sum.keys():
                                if key in m:
                                    metrics_sum[key] += m[key]
                            valid += 1
                    
                        if valid > 0:
                            denom = valid
                            for key in total_metrics.keys():
                                if key in metrics_sum:
                                    total_metrics[key] += metrics_sum[key] / denom
                            metrics_batches += 1
                   
                    else:
                    
                        if pred_vis is not None:
                            batch_metrics = calculate_medical_metrics(pred_vis, cur_target)
                            for key in total_metrics.keys():
                                if key in batch_metrics:
                                    total_metrics[key] += batch_metrics[key]
                            metrics_batches += 1
            

            if image_saver is not None and pred_vis is not None:
                try:
       
                    if isinstance(cur_target, list):
                        tgt_vis = cur_target[-1] if len(cur_target) > 0 and cur_target[-1] is not None else None
                    else:
                        tgt_vis = cur_target

                    if tgt_vis is not None and cur_inputs is not None:
                        image_saver.save_training_samples(cur_inputs, tgt_vis, pred_vis, epoch, batch_idx)
                except Exception as e:
                    print(f"⚠️ Training image save error: {e}")
                 # TensorBoard visualization removed; images are saved by ImageSaver to disk
            
            medk2n_loss_info = {}
            if medk2n_manager is not None:
                try:
       
                    base_features = out_feat if out_feat is not None else torch.randn(cur_inputs[0].size(0), 512, 64, 64).to(device)
                    enhance_features = base_features  
                    if base_features.dtype == torch.float16:
                        base_features = base_features.float()
                    if enhance_features.dtype == torch.float16:
                        enhance_features = enhance_features.float()

                    task_embeddings = torch.randn(cur_inputs[0].size(0), 128, dtype=torch.float32).to(device)  # 确保float32

                    for module_name, module in medk2n_manager.medk2n_modules.items():
                        if hasattr(module, 'float'):
                            module.float() 
                    
                    medk2n_output = medk2n_manager.forward_pipeline(
                        base_features, enhance_features, task_embeddings
                    )
                    

                    target_for_medk2n = cur_target[-1] if isinstance(cur_target, list) else cur_target
                    medk2n_loss_info = medk2n_manager.get_medk2n_loss(medk2n_output, target_for_medk2n, loss)
                    
                    medk2n_manager.update_quality_history(medk2n_output['quality_score'])

                    loss = loss + 0.1 * medk2n_loss_info['total_loss']  

                    losses.update({
                        'medk2n_quality_loss': medk2n_loss_info['quality_loss'].detach(),
                        'medk2n_quality_score': medk2n_loss_info['quality_score'].detach(),
                        'medk2n_quality_reward': medk2n_loss_info['quality_reward'].detach(),
                        'medk2n_weight_reg': medk2n_loss_info['weight_reg'].detach(),
                    })
                    
                    del medk2n_output, medk2n_loss_info, target_for_medk2n
                    del base_features, enhance_features, task_embeddings
                    
                    if rank == 0 and batch_idx % 100 == 0:  # 减少打印频率到每100个batch
                        quality_score = losses['medk2n_quality_score'].item()
            
                        pass  
                        
                except Exception as e:
       
                    if rank == 0 and batch_idx == 0:
                        error_msg = str(e)
                        if "should be the same" in error_msg or "out of memory" in error_msg.lower():
                            print(f"⚠️ MedK2N: {error_msg[:100]}...")  

            

            loss_t = loss if isinstance(loss, torch.Tensor) else torch.as_tensor(loss, device=device)
            loss_t = cast(torch.Tensor, loss_t)
            if loss_t.ndim > 0:
                loss_t = loss_t.mean()

            loss_t = loss_t / gradient_accumulation_steps
            scaler.scale(loss_t).backward()
            

            if (batch_idx + 1) % gradient_accumulation_steps == 0:

                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  
                
                try:
                    mt = model.module if hasattr(model, 'module') else model
                    if hasattr(mt, 'update_teacher_ema'):
                        mt.update_teacher_ema()
                except Exception:
                    pass
            
            total_loss += loss.item()
            
            
            if rank == 0 and hasattr(pbar, 'set_postfix') and batch_idx % 5 == 0:  
                avg_loss = total_loss / (batch_idx + 1)
                
                postfix_dict = {
                    'Loss': f'{float(loss.item()):.4f}',
                    'Avg': f'{avg_loss:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.1e}',
                    'Mem': f'{memory_manager.get_memory_info()["utilization"]*100:.0f}%'
                }
            
                if 'losses' in locals() and isinstance(losses, dict):
                    if 'L_causal' in losses:
                        postfix_dict['L_cas'] = f'{float(losses["L_causal"]):.3f}' 
                    if 'L_metric' in losses:
                        postfix_dict['L_met'] = f'{float(losses["L_metric"]):.3f}'
                    if 'clip' in losses:
                        postfix_dict['CLIP'] = f'{float(losses["clip"]):.3f}'
                    if 'medk2n_quality_score' in losses:
                        postfix_dict['MedQ'] = f'{float(losses["medk2n_quality_score"]):.3f}'  
                
                pbar.set_postfix(postfix_dict)

            del pred_vis, loss, losses
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                try:
                    if 'preds' in locals():
                        del preds
                    if 'cur_target' in locals():
                        del cur_target  
                    if 'cur_inputs' in locals():
                        del cur_inputs
                except:
                    pass
                    
                if batch_idx % (gradient_accumulation_steps * 5) == 0: 
                    memory_manager.cleanup_if_needed(force=False) 
                if batch_idx % (gradient_accumulation_steps * 20) == 0: 
                    torch.cuda.empty_cache() 
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if rank == 0:
                    print(f"❌ OOM in batch {batch_idx}, attempting recovery...")
                memory_manager.emergency_cleanup()
                optimizer.zero_grad(set_to_none=True)

                try:
                    del loss, losses, pred_vis
                    if 'preds' in locals():
                        del preds
                    if 'cur_target' in locals():
                        del cur_target
                    if 'cur_inputs' in locals():
                        del cur_inputs
                except:
                    pass
                
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
        
        except Exception as e:
            if rank == 0:
                print(f"❌ Error in batch {batch_idx}: {str(e)}")
            memory_manager.cleanup_if_needed(force=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        processed_batches += 1

        if max_batches is not None and (batch_idx + 1) >= int(max_batches):
            break

    avg_loss = total_loss / max(1, processed_batches)
    avg_train_metrics = {
        'psnr': total_metrics['psnr'] / max(1, metrics_batches),
        'ssim': total_metrics['ssim'] / max(1, metrics_batches)
    }
    
    if rank == 0 and loss_stats['count'] > 0:
        avg_loss = loss_stats['total'] / loss_stats['count']
        print(f"\n📊  - : {avg_loss:.4f}")
        for key in ['L_causal', 'L_metric']:
            if key in loss_stats and loss_stats[key] > 0:
                avg = loss_stats[key] / loss_stats['count']
                print(f"   {key}: {avg:.4f}")
        print(f"   : {loss_stats['count']}")
    
    return avg_loss, avg_train_metrics

def validate_optimized(model, dataloader, criterion, device, epoch, memory_manager, rank, image_saver=None, writer=None, max_batches: Optional[int] = None,
                       enable_curriculum: bool = False, stage: Optional[str] = None,
                       avoid_identity_in_consistency: bool = False,
                       medk2n_manager: Optional[MedK2N_TrainingManager] = None):
    model.eval()
    total_loss = 0.0
    total_metrics = {'psnr': 0.0, 'ssim': 0.0}
    valid_metric_batches = 0 
    num_batches = len(dataloader)
    processed_batches = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
    else:
        pbar = dataloader
    first_batch_saved = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                if batch_idx % 5 == 0:
                    memory_manager.cleanup_if_needed()
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, target = batch
                    inputs = [x.to(device, non_blocking=True) for x in inputs]
                    if isinstance(target, (list, tuple)):
                        target = [t.to(device, non_blocking=True) for t in target]
                    else:
                        target = target.to(device, non_blocking=True)
                elif isinstance(batch, dict):
                    inputs_list = batch.get('inputs', None)
                    target_tensor = batch.get('target', None)
                    if inputs_list is None or target_tensor is None:
                        inputs, target = batch  # 回退
                        inputs = [x.to(device, non_blocking=True) for x in inputs]
                        if isinstance(target, (list, tuple)):
                            target = [t.to(device, non_blocking=True) for t in target]
                        else:
                            target = target.to(device, non_blocking=True)
                    else:
                        inputs = [x.to(device, non_blocking=True) for x in inputs_list]
                        if isinstance(target_tensor, (list, tuple)):
                            target = [t.to(device, non_blocking=True) for t in target_tensor]
                        else:
                            target = target_tensor.to(device, non_blocking=True)
                else:
                    inputs, target = batch
                    inputs = [x.to(device, non_blocking=True) for x in inputs]
                    if isinstance(target, (list, tuple)):
                        target = [t.to(device, non_blocking=True) for t in target]
                    else:
                        target = target.to(device, non_blocking=True)
                
                cur_inputs = inputs
                cur_target = target
                selected_pred_idx = None
                if enable_curriculum:
                    cur_inputs, tgt_list, selected_pred_idx, _sel = apply_curriculum_to_batch(inputs, target, stage or 'kton', epoch, batch_idx, rank)
                    if isinstance(tgt_list, (list, tuple)) and len(tgt_list) == 1:
                        cur_target = tgt_list[0]
                        selected_pred_idx = selected_pred_idx if selected_pred_idx is not None else 0
                    else:
                        cur_target = tgt_list
                with autocast():
                    out_gen, out_feat, out_dict = model(cur_inputs, multimask_output=True)
                    preds = out_gen if isinstance(out_gen, list) else [out_gen]
                    if isinstance(cur_target, list):
                        pair_count = min(len(preds), len(cur_target))
                        total_task_loss = 0.0
                        for i in range(pair_count):
                            p_i = preds[i]
                            t_i = cur_target[i]
                            if not (torch.is_tensor(p_i) and torch.is_tensor(t_i)):
                                continue

                            preds[i] = _resize_pred_to_target(p_i, t_i)
                            if isinstance(cur_inputs, (list, tuple)) and len(cur_inputs) > 0:
                                _idx = i % len(cur_inputs)
                                if avoid_identity_in_consistency and len(cur_inputs) > 1:
                                    _idx = (_idx + 1) % len(cur_inputs)
                                input_for_pair = [cur_inputs[_idx]]
                            else:
                                input_for_pair = cur_inputs
                            li, _ = criterion(preds[i], cur_target[i], input_for_pair)
                            total_task_loss = total_task_loss + li
                        denom = max(1, pair_count)
                        task_loss = total_task_loss / denom
                        pred_vis = next((p for p in reversed(preds) if p is not None), preds[0] if len(preds) > 0 else None)
                    else:
                        pred_idx = selected_pred_idx if (selected_pred_idx is not None and 0 <= int(selected_pred_idx) < len(preds)) else (len(preds)-1)
                        if (
                            len(preds) > 0
                            and isinstance(preds[pred_idx], torch.Tensor)
                            and isinstance(cur_target, torch.Tensor)
                            and preds[pred_idx].shape != cur_target.shape
                        ):
                            preds[pred_idx] = F.interpolate(preds[pred_idx], size=cur_target.shape[2:], mode='bilinear', align_corners=False)
                        pred_vis = preds[pred_idx] if len(preds) > 0 else None
                        if pred_vis is not None:
                            if isinstance(cur_inputs, (list, tuple)) and len(cur_inputs) > 0:
                                _idx = pred_idx % len(cur_inputs)
                                if avoid_identity_in_consistency and len(cur_inputs) > 1:
                                    _idx = (_idx + 1) % len(cur_inputs)
                                input_for_pair = [cur_inputs[_idx]]
                            else:
                                input_for_pair = cur_inputs
                            task_loss, _ = criterion(pred_vis, cur_target, input_for_pair)
                        else:
                            task_loss = torch.tensor(0.0, device=device)

                    distill_dict = out_dict.get('losses', {}) if isinstance(out_dict, dict) else {}
                    distill_total = distill_dict.get('total_loss', torch.tensor(0.0, device=device))
                    if not isinstance(distill_total, torch.Tensor):
                        try:
                            distill_total = torch.as_tensor(float(distill_total), device=device)
                        except Exception:
                            distill_total = torch.tensor(0.0, device=device)
                    loss = task_loss + 0.5 * distill_total  # 验证阶段固定权重用于评估
                
                if image_saver is not None and batch_idx == 0 and not first_batch_saved and pred_vis is not None:
                    try:
                        if isinstance(cur_target, list):
                            tgt_vis = cur_target[-1] if len(cur_target) > 0 and cur_target[-1] is not None else None
                        else:
                            tgt_vis = cur_target

                        if tgt_vis is not None and cur_inputs is not None:
                            image_saver.save_validation_samples(cur_inputs, tgt_vis, pred_vis, epoch)
                            first_batch_saved = True
                    except Exception as _e:
                        if rank == 0:
                            print(f"⚠️ Validation image save error: {str(_e)}")
                
                total_loss += loss.item()
                
                if isinstance(cur_target, list):

                    pair_count = min(len(preds), len(cur_target))
                    psnr_sum, ssim_sum, valid = 0.0, 0.0, 0
                    for i in range(pair_count):
                        if preds[i] is None or cur_target[i] is None:
                            continue
                        m = calculate_medical_metrics(preds[i], cur_target[i])
                        psnr_sum += m['psnr']
                        ssim_sum += m['ssim']
                        valid += 1
                    denom = max(1, valid)
                    total_metrics['psnr'] += psnr_sum / denom
                    total_metrics['ssim'] += ssim_sum / denom
                    if valid > 0:
                        valid_metric_batches += 1
                else:
                    if pred_vis is not None:
                        batch_metrics = calculate_medical_metrics(pred_vis, cur_target)
                        total_metrics['psnr'] += batch_metrics['psnr']
                        total_metrics['ssim'] += batch_metrics['ssim']
                        valid_metric_batches += 1
                
                del loss, pred_vis, preds
                processed_batches += 1
                if max_batches is not None and (batch_idx + 1) >= int(max_batches):
                    break
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if rank == 0:
                        print(f"❌ Validation OOM in batch {batch_idx}")
                    memory_manager.emergency_cleanup()
                    continue
                else:
                    raise e
            
            except Exception as e:
                if rank == 0:
                    print(f"❌ Validation error in batch {batch_idx}: {str(e)}")
                memory_manager.cleanup_if_needed(force=True)
                continue
    avg_loss = total_loss / max(1, processed_batches)
    denom_batches = max(1, valid_metric_batches)
    avg_metrics = {k: v / denom_batches for k, v in total_metrics.items()}
    
    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss).cuda()
        psnr_tensor = torch.tensor(avg_metrics['psnr']).cuda()
        ssim_tensor = torch.tensor(avg_metrics['ssim']).cuda()

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(ssim_tensor, op=dist.ReduceOp.SUM)

        world_size = dist.get_world_size()
        avg_loss = loss_tensor.item() / world_size
        avg_metrics['psnr'] = psnr_tensor.item() / world_size
        avg_metrics['ssim'] = ssim_tensor.item() / world_size
    
    # TensorBoard logging removed; validation metrics printed and saved via ImageSaver
    
    return avg_loss, avg_metrics

def main():
    parser = argparse.ArgumentParser(description='集成版内存优化分布式训练脚本')
    parser.add_argument('--cfg', type=str, default='configs/medicalmri.yaml')
    parser.add_argument('--save-dir', '--save_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--niter', type=int, default=None, help='# of iter at starting learning rate (overrides config file if specified)')
    parser.add_argument('--niter_decay', type=int, default=None, help='# of iter to linearly decay learning rate to zero (overrides config file if specified)')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=None, help='Total batch size across all GPUs (overrides config file if specified)')
    parser.add_argument('--gpu-ids', '--gpu_ids', type=str, default=None, help='GPU IDs to use (comma separated, overrides config file if specified)')
    parser.add_argument('--task', type=str, default=None, help='Task specification (e.g., A2B, ABC2D, AB2CD)')
    parser.add_argument('--target-modals', type=str, default=None, help='Optional explicit target modalities string (e.g., "CD" for k->n)')
    parser.add_argument('--use-clip-loss', action='store_true', default=True, help='Enable CLIP loss for medical modality alignment (default: on)')
    parser.add_argument('--clip-curriculum', action='store_true', default=False, help='Enable curriculum learning for CLIP loss weights')
    parser.add_argument('--mean-teacher', action='store_true', default=True, help='Enable Mean Teacher (EMA) online distillation')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay for Mean Teacher')
    parser.add_argument('--lambda-distill', type=float, default=0.5, help='Weight for distillation loss in total loss')
    parser.add_argument('--enable-curriculum', action='store_true', default=True, help='Enable curriculum: easy->medium->hard->expert')
    parser.add_argument('--save-model-only', action='store_true', default=True, help='Also save a model-only checkpoint without optimizer/scheduler to prioritize stability and smaller files')
    parser.add_argument('--save-image-every-batches', type=int, default=100, help='How often (in batches) to save training images via ImageSaver')
    parser.add_argument('--enable-image-saving', '--enable_image_saving', action='store_true', default=False, help='Enable saving training/validation images to disk')
    parser.add_argument('--max-train-vis-samples', type=int, default=2, help='Max samples per training batch to visualize')
    parser.add_argument('--max-val-vis-samples', type=int, default=4, help='Max samples from first validation batch to visualize')
    parser.add_argument('--vis-downscale', type=int, default=2, help='Downscale factor applied before saving images (>=1)')
    parser.add_argument('--save-diff-images', action='store_true', default=False, help='Also save difference heatmaps between pred and target in validation')
    parser.add_argument('--disable-rgb-generation', action='store_true', default=False, help='Disable internal RGB generation branch inside the model to save memory')
    parser.add_argument('--curriculum-ratios', type=str, default='0.2,0.2,0.3,0.3', help='Stage ratios for curriculum (easy,medium,hard,expert)')
    parser.add_argument('--curriculum-apply-to-val', action='store_true', default=False, help='Apply curriculum to validation as well')
    parser.add_argument('--identity-weight', type=float, default=0.0, help='Legacy: Weight for identity mapping tasks')
    parser.add_argument('--cross-modal-weight', type=float, default=1.0, help='Legacy: Weight for cross-modal tasks')
    parser.add_argument('--max-train-batches', type=int, default=None, help='Limit number of training batches per epoch (for smoke tests)')
    parser.add_argument('--max-val-batches', type=int, default=None, help='Limit number of validation batches per epoch (for smoke tests)')
    parser.add_argument('--enable-modality-dropout', action='store_true', default=False, help='Enable higher-probability dropout for same-modal inputs when supervising that target')
    parser.add_argument('--modal-dropout-p-same', type=float, default=0.0, help='Drop prob for input equal to supervised target modality')
    parser.add_argument('--modal-dropout-p-default', type=float, default=0.0, help='Drop prob for other inputs')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    niter = args.niter if args.niter is not None else cfg.get('TRAIN', {}).get('NITER', cfg.get('TRAIN', {}).get('EPOCHS_REGULAR', 30))
    niter_decay = args.niter_decay if args.niter_decay is not None else cfg.get('TRAIN', {}).get('NITER_DECAY', cfg.get('TRAIN', {}).get('EPOCHS_DECAY', 30))

    def _extract_gpu_ids(_cfg: dict):
        def _norm(v):
            if v is None:
                return None
            if isinstance(v, list):
                return ','.join(map(str, v))
            return str(v)
        train_ids = _norm(_cfg.get('TRAIN', {}).get('GPU_IDS'))
        run_ids = _norm(_cfg.get('RUN', {}).get('GPU_IDS'))
        top_ids = _norm(_cfg.get('GPU_IDS'))
        return train_ids or run_ids or top_ids

    cfg_gpu_ids = _extract_gpu_ids(cfg)

    existing_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if existing_cuda_visible is not None and existing_cuda_visible.strip() != '':

        gpu_ids = existing_cuda_visible
        print(f"🎯 GPU: CUDA_VISIBLE_DEVICES={gpu_ids}")
    else:

        gpu_ids = args.gpu_ids if args.gpu_ids is not None else cfg_gpu_ids
        if gpu_ids is None:
            gpu_ids = '0'  # 默认用0号卡
        
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"🎯 GPU: CUDA_VISIBLE_DEVICES={gpu_ids}")
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    try:
        assert torch.cuda.current_device() == local_rank, (
            f"CUDA current_device={torch.cuda.current_device()} != local_rank={local_rank}"
        )
    except Exception as _e:
        print(f"[rank {rank}] ⚠️ : {_e}")

    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    print(f"[rank {rank}] world_size={world_size}, local_rank={local_rank}, device={device}")
    if rank == 0:
        print(f"[Init] world_size={world_size}, GPUs={gpu_ids}{' task='+args.task if args.task else ''}")
    
    memory_manager = MemoryManager(device)
    if args.task:
        input_modals, target_modals = parse_task_specification(args.task)
        if input_modals:
            cfg['DATASET']['MODALS'] = input_modals
            tgt_list = list(target_modals) if target_modals else []
            if args.target_modals:
                cfg['DATASET']['TARGET_MODAL'] = list(args.target_modals)
                if rank == 0:
                    print(f"🎯 : {input_modals} -> {list(args.target_modals)} (k->n)")
            else:
                if len(tgt_list) == 0:
                    cfg['DATASET']['TARGET_MODAL'] = None
                elif len(tgt_list) == 1:
                    cfg['DATASET']['TARGET_MODAL'] = tgt_list[0]
                else:
                    cfg['DATASET']['TARGET_MODAL'] = tgt_list
                if rank == 0:
                    n = len(tgt_list)
                    tgt_echo = ''.join(tgt_list) if n > 1 else (tgt_list[0] if n == 1 else None)
                    print(f"🎯 : {input_modals} -> {tgt_echo}{' (k->n)' if n > 1 else ''}")
    else:
        try:
            task_str = str(cfg.get('DATASET', {}).get('TASK', '') or '').strip()
            if task_str and '2' in task_str:
                in_mods, tgt_mods = parse_task_specification(task_str)
                if in_mods:
                    cfg['DATASET']['MODALS'] = in_mods
                    tgt_list = list(tgt_mods) if tgt_mods else []
                    if len(tgt_list) == 0:
                        cfg['DATASET']['TARGET_MODAL'] = None
                    elif len(tgt_list) == 1:
                        cfg['DATASET']['TARGET_MODAL'] = tgt_list[0]
                    else:
                        cfg['DATASET']['TARGET_MODAL'] = tgt_list
                    if rank == 0:
                        n = len(tgt_list)
                        tgt_echo = ''.join(tgt_list) if n > 1 else (tgt_list[0] if n == 1 else None)
                        print(f"🎯  YAML : {in_mods} -> {tgt_echo}{' (k->n)' if n > 1 else ''}")
        except Exception as _e:
            if rank == 0:
                print(f"[Warn]  YAML  DATASET.TASK : {_e}")
    
    dataset_cfg = cfg['DATASET']
    yaml_save_dir = (
        cfg.get('SAVE_DIR', None)
        or cfg.get('OUTPUT_DIR', None)
        or cfg.get('DATASET', {}).get('SAVE_DIR', None)
        or 'outputs/integrated_training'
    )
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = yaml_save_dir
    if args.task or dataset_cfg.get('MODALS'):
        input_modals = dataset_cfg.get('MODALS', ['A'])
        target_modal = dataset_cfg.get('TARGET_MODAL', None)
        if isinstance(target_modal, list):
            target_str = ''.join(target_modal)
        else:
            target_str = target_modal
        run_name = create_run_name(input_modals, target_str)
        save_dir = os.path.join(save_dir, run_name)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"💾 : {save_dir}")
    args.save_dir = save_dir
    
    fix_seeds(42 + rank)
    
    if args.batch_size is not None:
        if rank == 0:
            print(f"📦 batch_size: {args.batch_size} ( {cfg['TRAIN']['BATCH_SIZE']})")
        cfg['TRAIN']['BATCH_SIZE'] = args.batch_size
    else:
        if rank == 0:
            print(f"📦 batch_size: {cfg['TRAIN']['BATCH_SIZE']}")

    clip_cfg_yaml = cfg.get('CLIP', {}) or {}
    if 'enable' in clip_cfg_yaml:
        args.use_clip_loss = bool(clip_cfg_yaml.get('enable'))
    cur_cfg_yaml = cfg.get('CURRICULUM', {}) or {}
    if len(cur_cfg_yaml) > 0:
        if 'ENABLE_CURRICULUM' in cur_cfg_yaml:
            args.enable_curriculum = bool(cur_cfg_yaml.get('ENABLE_CURRICULUM'))
        if 'CURRICULUM_RATIOS' in cur_cfg_yaml:
            ratios = cur_cfg_yaml.get('CURRICULUM_RATIOS')
            if isinstance(ratios, (list, tuple)):
                args.curriculum_ratios = ','.join([str(x) for x in ratios])
            elif isinstance(ratios, str):
                args.curriculum_ratios = ratios
        if 'CURRICULUM_APPLY_TO_VAL' in cur_cfg_yaml:
            args.curriculum_apply_to_val = bool(cur_cfg_yaml.get('CURRICULUM_APPLY_TO_VAL'))
    
    train_transform = get_train_augmentation(cfg['TRAIN']['IMAGE_SIZE'])
    val_transform = get_val_augmentation(cfg['EVAL']['IMAGE_SIZE'])
    
    dataset_cfg = cfg['DATASET']
    
    dataset_modals = dataset_cfg.get('MODALS', None)
    dataset_target = dataset_cfg.get('TARGET_MODAL', None)
    if rank == 0:
        echo_modals = dataset_modals if dataset_modals is not None else 'auto'
        tgt_echo = dataset_target if dataset_target is not None else 'auto(last)'
        print(f"[Data] root={dataset_cfg.get('ROOT')} modals={echo_modals} target={tgt_echo}")
    
    train_dataset = MedicalMRI(
        root=dataset_cfg.get('ROOT'),
        split='train',
        transform=train_transform,
        modals=dataset_modals,
        target_modal=dataset_target
    )
    val_dataset = MedicalMRI(
        root=dataset_cfg.get('ROOT'),
        split='val',
        transform=val_transform,
        modals=dataset_modals,
        target_modal=dataset_target
    )
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None
    
    batch_size_per_gpu = max(1, int(cfg['TRAIN']['BATCH_SIZE']) // world_size)
    val_batch_size_per_gpu = max(1, int(cfg.get('EVAL', {}).get('BATCH_SIZE', cfg['TRAIN']['BATCH_SIZE'])) // world_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,  # 减少worker数量
        pin_memory=False,  # 关闭pin_memory以节省内存
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
    batch_size=val_batch_size_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False
    )
    
    if rank == 0:
        print(f"[Data] train={len(train_dataset)} val={len(val_dataset)} batch/gpu={batch_size_per_gpu} mapping={train_dataset.modal_mapping}")

    empty_flag = int(len(train_dataset) == 0 or len(val_dataset) == 0)
    if world_size > 1:
        t = torch.tensor(empty_flag, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        empty_flag = int(t.item() > 0)
    if empty_flag:
        if rank == 0:
            if len(train_dataset) == 0:
                print("❌ ！。")
            if len(val_dataset) == 0:
                print("❌ ！。")
        if world_size > 1:
            dist.barrier()
        return
    
    try:
        model_yaml = cfg.get('MODEL', {}) or {}
        backbone_name = str(model_yaml.get('BACKBONE', 'sam2_hiera_b+'))
        checkpoint = model_yaml.get('PRETRAINED', "/data1/tempf/sam2.1_hiera_base_plus.pt")
        is_v21 = ('2.1' in str(checkpoint)) or backbone_name.startswith('sam2.1')
        version_dir = 'sam2.1' if is_v21 else 'sam2'

        base = backbone_name
        if not base.endswith('.yaml'):
            base = f"{base}.yaml"

        # normalize prefix for v2.1 vs v2.0 naming
        if is_v21:
            # target format: sam2.1_hiera_b+.yaml
            if not base.startswith('sam2.1_'):
                if base.startswith('sam2_'):
                    base = base.replace('sam2_', 'sam2.1_', 1)
                else:
                    base = 'sam2.1_' + base
        else:
            # target format: sam2_hiera_b+.yaml
            if base.startswith('sam2.1_'):
                base = base.replace('sam2.1_', 'sam2_', 1)

        # Prefer a config variant that matches the dataset image size (e.g. *_256.yaml)
        # Determine image size from TRAIN/EVAL blocks (fallback to 256)
        img_size = None
        try:
            train_img = cfg.get('TRAIN', {}) or {}
            eval_img = cfg.get('EVAL', {}) or {}
            img_size = train_img.get('IMAGE_SIZE') or eval_img.get('IMAGE_SIZE')
            if isinstance(img_size, (list, tuple)):
                img_size = int(img_size[0])
            elif isinstance(img_size, int):
                img_size = int(img_size)
        except Exception:
            img_size = None

        candidates = []
        # candidate 1: standard hydra-style path
        candidates.append(f"configs/{version_dir}/{base}")
        # candidate 2: direct base (used in some scripts)
        candidates.append(base)

        # if image size is 256, prefer *_256.yaml variants (common in repo)
        if img_size == 256:
            if not base.endswith('_256.yaml'):
                alt = base.replace('.yaml', '_256.yaml')
                candidates.insert(0, alt)  # try alt first
                candidates.insert(0, f"configs/{version_dir}/{alt}")

        # Try candidates in order until build_sam2 succeeds
        build_errs = []
        teacher = student = None
        for cand in candidates:
            try:
                if rank == 0:
                    print(f"🧩 : {cand}")
                # prepare hydra overrides to align model image_size with dataset
                hydra_overrides = []
                if img_size is not None:
                    hydra_overrides.append(f"++model.image_size={int(img_size)}")
                    # set sensible memory_attention feat sizes (approx img_size/32 used in repo)
                    feat = max(1, int(img_size // 32))
                    hydra_overrides.append(f"++model.memory_attention.layer.self_attention.feat_sizes=[{feat},{feat}]")
                    hydra_overrides.append(f"++model.memory_attention.layer.cross_attention.feat_sizes=[{feat},{feat}]")
                teacher = build_sam2(cand, checkpoint, device=str(device), hydra_overrides_extra=hydra_overrides)
                student = build_sam2(cand, checkpoint, device=str(device), hydra_overrides_extra=hydra_overrides)
                model_cfg_file = cand
                if rank == 0:
                    print(f"📦 : {checkpoint}")
                break
            except Exception as e:
                build_errs.append((cand, repr(e)))
                if rank == 0:
                    print(f"⚠️  {cand}: {e}")

        if teacher is None or student is None:
            # fallback: raise the first build error to surface the problem
            if rank == 0:
                print("❌ SAM2，:")
                for c, err in build_errs:
                    print(f"  - {c}: {err}")
            raise RuntimeError("无法加载任何 SAM2 配置文件，请检查 sam2 配置和训练图像尺寸是否匹配。")

        lora_cfg = SAM2LoRAConfig()
        
        try:
            lora_cfg.update_from_yaml_config(cfg)
            if rank == 0:
                print(f"✅ YAML")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  YAML，: {e}")
        
        _env_act = os.environ.get('ENABLE_ACT_HOOKS')
        if _env_act is not None and _env_act != '':
            lora_cfg.enable_act_hooks = bool(int(_env_act))
        _env_cross = os.environ.get('ENABLE_CROSS_FUSION')
        if _env_cross is not None and _env_cross != '':
            lora_cfg.enable_cross_frame_fusion = bool(int(_env_cross))
        _env_btl = os.environ.get('ENABLE_BOTTLENECK_SPIRAL')
       
        if _env_btl is not None and _env_btl != '':
            lora_cfg.enable_bottleneck_spiral = bool(int(_env_btl))
        try:
            tgt_cfg = dataset_cfg.get('TARGET_MODAL', None)
            n_targets = len(tgt_cfg) if isinstance(tgt_cfg, list) else 1
            if n_targets > 0:
                lora_cfg.num_output_frames = n_targets
        except Exception:
            pass
        if getattr(args, 'disable_rgb_generation', False):
            try:
                lora_cfg.enable_rgb_generation = False
                if rank == 0:
                    print("🧩 CLIRGB (--disable-rgb-generation)")
            except Exception:
                pass

        if hasattr(lora_cfg, 'mean_teacher'):
            lora_cfg.mean_teacher = bool(args.mean_teacher)

        if hasattr(lora_cfg, 'ema_decay'):
            lora_cfg.ema_decay = float(args.ema_decay)
        model = create_sam2_lora_model(
            teacher_model=teacher,
            student_model=student,
            config=lora_cfg
        ).to(device)

        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
            if rank == 0:
                print("🔗  DistributedDataParallel")
        else:
            if rank == 0:
                print("⚠️  ， torchrun GPU")
                print("   GPU")

        if rank == 0:
            print("🏗️ SAM2")
            
        detected_feat_ch = None
        try:
            model_eval = model.module if hasattr(model, 'module') else model
            model_eval.eval()
            try:
                img_size = cfg.get('TRAIN', {}).get('IMAGE_SIZE') or cfg.get('EVAL', {}).get('IMAGE_SIZE') or 256
                if isinstance(img_size, (list, tuple)):
                    img_size = int(img_size[0])
            except Exception:
                img_size = 256
            try:
                n_modals = len(cfg.get('DATASET', {}).get('MODALS', []) or [1])
            except Exception:
                n_modals = 1
            dummy = [torch.zeros(1, 3, int(img_size), int(img_size), device=device) for _ in range(max(1, n_modals))]
            with torch.no_grad():
                out = model_eval(dummy, multimask_output=True)
            # out is (out_gen, out_feat, out_dict)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                out_feat = out[1]
                if out_feat is not None and hasattr(out_feat, 'shape'):
                    if isinstance(out_feat, (list, tuple)):
                        detected_feat_ch = out_feat[0].shape[1]
                    else:
                        detected_feat_ch = out_feat.shape[1]
            if detected_feat_ch is not None and rank == 0:
                print(f"🔍 : {detected_feat_ch}")
        except Exception as _e:
            detected_feat_ch = None
            if rank == 0:
                print(f"⚠️ : {_e}")

        medk2n_manager = MedK2N_TrainingManager(config=cfg, device=device, in_feat_channels=detected_feat_ch)
        
        medk2n_module_count = 0
        for module_name, module in medk2n_manager.medk2n_modules.items():
            if hasattr(module, 'float'):
                module.float()  
                medk2n_module_count += 1
        
        if rank == 0:
            print(f"✅ MedK2N: {medk2n_module_count} float32")
            
    except Exception as e:
        if rank == 0:
            print(f"❌ SAM2: {e}")
        return
    
    try:
        criterion = create_enhanced_criterion_with_clip(args, device, dataset_cfg)
        if rank == 0:
            if CLIP_AVAILABLE and getattr(args, 'use_clip_loss', False):
                print("💪 （CLIP，）")
            else:
                print("💪 （）")
    except Exception as e:
        if rank == 0:
            print(f"⚠️ : {e}")
            print("   AccurateLoss")
        
        criterion = AccurateLoss(
            device=str(device),
            lambda_weighted_l1=0.5,      
            lambda_ssim=0.5,                  
            lambda_grad=0.2,                  
            lambda_consistency=0.2,      
            lambda_lesion_aware=0.1,  
            lambda_tv=0                        
        )
        if rank == 0:
            print("💪 ")
    
    model_for_optim = model.module if hasattr(model, 'module') else model
    opt_cfg = cfg.get('OPTIMIZER', {}) or {}
    train_cfg = cfg.get('TRAIN', {}) or {}
    
    opt_lr = float(train_cfg.get('LR', opt_cfg.get('LR', 1e-4)))

    opt_name = str(opt_cfg.get('NAME', 'adamw')).lower()
    opt_wd = float(opt_cfg.get('WEIGHT_DECAY', 0.01))
    
    beta1 = float(train_cfg.get('BETA1', 0.9))
    beta2 = float(train_cfg.get('BETA2', 0.999))
    
    model_params = list(model_for_optim.parameters())
    medk2n_params = medk2n_manager.get_all_parameters()
    all_params = model_params + medk2n_params
    
    optimizer = get_optimizer(all_params, opt_name, lr=opt_lr, weight_decay=opt_wd, 
                               betas=(beta1, beta2))

    if rank == 0:
        sam_param_count = sum(p.numel() for p in model_params if p.requires_grad)
        medk2n_param_count = sum(p.numel() for p in medk2n_params if p.requires_grad)
        print(f"📊 : SAM2={sam_param_count:,}, MedK2N={medk2n_param_count:,}, ={sam_param_count+medk2n_param_count:,}")
        print(f"🎯 : lr={opt_lr:.1e}, beta1={beta1}, beta2={beta2}, wd={opt_wd}")

    amp_enabled = train_cfg.get('AMP', True)
    gradient_accumulation_steps = int(train_cfg.get('GRADIENT_ACCUMULATION_STEPS', 1))
    print_interval = int(train_cfg.get('PRINT_INTERVAL', 100))
    eval_interval = int(train_cfg.get('EVAL_INTERVAL', 1000))
    save_interval = int(train_cfg.get('SAVE_INTERVAL', 5000))
    
    if rank == 0:
        print(f"📋 : AMP={amp_enabled}, ={gradient_accumulation_steps}")
        print(f"📊 ={print_interval}, ={eval_interval}, ={save_interval}")

    total_epochs = niter + niter_decay
    sched_cfg = cfg.get('SCHEDULER', {}) or {}
    sched_name = str(sched_cfg.get('NAME', 'cosineannealinglr')).lower()
    if sched_name == 'cosineannealinglr':
        eta_min = float(sched_cfg.get('MIN_LR', 1.0e-7))
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
        if rank == 0:
            print(f"🗓️ : CosineAnnealingLR (T_max={total_epochs}, eta_min={eta_min})")
    else:
        eta_min = 1.0e-7
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
        if rank == 0:
            print(f"[Warn]  {sched_name}， CosineAnnealingLR。 warmup， NAME: cosineannealinglr")
    
    image_saver = ImageSaver(
        args.save_dir,
        rank,
        save_every_batches=args.save_image_every_batches,
        enabled=bool(args.enable_image_saving),
        max_train_samples=int(args.max_train_vis_samples),
        max_val_samples=int(args.max_val_vis_samples),
        downscale=int(max(1, args.vis_downscale)),
        save_diffs=bool(args.save_diff_images),
    )
    
    # TensorBoard removed; visualizations saved by ImageSaver. Keep writer=None for compatibility.
    writer = None
    
    
    if rank == 0:
        print(f"[Start] epochs={total_epochs} init_mem={memory_manager.get_memory_info()}")
        if getattr(args, 'enable_image_saving', False):
            print(f"[Image] save_every={args.save_image_every_batches} dir={args.save_dir}")
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.isfile(args.resume):
        if rank == 0:
            print(f"🔄 : {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model_state = model.module if hasattr(model, 'module') else model
        model_state.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        if rank == 0:
            print(f"✅ {start_epoch}，: {best_val_loss:.6f}")
    
    for epoch in range(start_epoch, total_epochs):
        memory_manager.cleanup_if_needed(force=True)
        
        if getattr(args, 'use_clip_loss', False) and getattr(args, 'clip_curriculum', False):
            adjust_clip_loss_weight(criterion, epoch, total_epochs, rank)
        
        if rank == 0:
            print(f"[Epoch {epoch+1}/{total_epochs}] lr={optimizer.param_groups[0]['lr']:.2e}")
    
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
    
        stage_ratios = _parse_stage_ratios(args.curriculum_ratios)
        stage = get_curriculum_stage(epoch, total_epochs, stage_ratios) if args.enable_curriculum else 'kton'
        gradient_accumulation_steps = cfg.get('TRAIN', {}).get('GRADIENT_ACCUMULATION_STEPS', 1)
        train_loss, train_metrics = train_one_epoch_optimized(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch, memory_manager, rank, image_saver, writer,
            lambda_distill=float(args.lambda_distill), max_batches=args.max_train_batches,
            enable_curriculum=bool(args.enable_curriculum), stage=stage,
            enable_modality_dropout=bool(args.enable_modality_dropout),
            modal_dropout_p_same=float(args.modal_dropout_p_same),
            modal_dropout_p_default=float(args.modal_dropout_p_default),
            medk2n_manager=medk2n_manager,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        val_loss, val_metrics = validate_optimized(
            model, val_loader, criterion, device, epoch, memory_manager, rank, image_saver, writer,
            max_batches=args.max_val_batches, enable_curriculum=bool(args.curriculum_apply_to_val), stage=stage,
            avoid_identity_in_consistency=False,  # 课程学习已处理恒等映射
            medk2n_manager=medk2n_manager,
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
    # TensorBoard logging removed; epoch metrics printed and visualizations saved to disk
        
        if rank == 0:
            print(f"[Result] epoch={epoch+1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_psnr={train_metrics['psnr']:.3f} val_psnr={val_metrics['psnr']:.3f} train_ssim={train_metrics['ssim']:.4f} val_ssim={val_metrics['ssim']:.4f} mem={memory_manager.get_memory_info()['utilization']*100:.1f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics,
                    'config': cfg
                }, f'{args.save_dir}/best_model.pth')
                if getattr(args, 'save_model_only', False):
                    torch.save({'model_state_dict': model_state, 'epoch': epoch, 'val_loss': val_loss, 'config': cfg}, f'{args.save_dir}/best_model_model_only.pth')
                print(f"[Save] best epoch={epoch+1} val_loss={val_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics,
                    'config': cfg
                }, f'{args.save_dir}/checkpoint_epoch_{epoch+1}.pth')
                if getattr(args, 'save_model_only', False):
                    torch.save({'model_state_dict': model_state, 'epoch': epoch, 'val_loss': val_loss, 'config': cfg}, f'{args.save_dir}/checkpoint_epoch_{epoch+1}_model_only.pth')
                print(f"[Save] checkpoint epoch={epoch+1}")
    
        try:
            metrics_csv = os.path.join(args.save_dir, 'metrics.csv')
            header = False
            if not os.path.exists(metrics_csv):
                header = True
            with open(metrics_csv, 'a') as f:
                if header:
                    f.write('epoch,train_loss,val_loss,train_psnr,val_psnr,train_ssim,val_ssim,lr\n')
                f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{train_metrics['psnr']:.6f},{val_metrics['psnr']:.6f},{train_metrics['ssim']:.6f},{val_metrics['ssim']:.6f},{current_lr:.6e}\n")
        except Exception as _e:
            if rank == 0:
                print(f"⚠️  metrics.csv: {_e}")

    if rank == 0:
        print(f"[Done] best_val_loss={best_val_loss:.4f} out_dir={args.save_dir}")
        # TensorBoard removed; logs not generated. Visualizations saved to disk.
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()





