#!/usr/bin/env python3
"""
更准确和适配的损失函数实现
专门针对病灶保持优化 - 干净版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torchvision.models as models

class AccurateLoss(nn.Module):
    """更准确的医学图像损失函数 - 专门针对病灶保持优化，添加感知损失"""
    
    def __init__(self, device='cuda', lambda_weighted_l1=0.0, lambda_ssim=0.0, 
                 lambda_grad=0.0, lambda_consistency=0.0, lambda_lesion_aware=0.0, 
                 lambda_tv=0.0, lambda_perceptual=0.0,
                 # 保持向后兼容的旧参数
                 lambda_l1=0.0):
        super().__init__()
        self.device = device
        
        # 基础损失函数权重
        self.lambda_weighted_l1 = lambda_weighted_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_consistency = lambda_consistency
        self.lambda_lesion_aware = lambda_lesion_aware
        self.lambda_tv = lambda_tv
        self.lambda_perceptual = lambda_perceptual  # 新增感知损失
        
        # 向后兼容：如果使用旧参数，设置相应属性
        self.lambda_l1 = lambda_l1
        
        # 如果使用旧参数且新参数为0，则映射
        if lambda_l1 > 0 and lambda_weighted_l1 == 0.0:
            self.lambda_weighted_l1 = lambda_l1
                
        # 初始化损失函数组件
        self.gradient_loss = EnhancedGradientLoss()
        if self.lambda_weighted_l1 > 0:
            self.weighted_l1_loss = WeightedL1Loss()
        if self.lambda_lesion_aware > 0:
            self.lesion_aware_loss = LesionAwareLoss()
        
        # 初始化感知损失 (使用VGG19特征)
        if self.lambda_perceptual > 0:
            self.perceptual_loss = PerceptualLoss(device=device)
    
    def forward(self, pred, target, input_modal=None):
        losses = {}
        total_loss = 0.0
        
        # 加权L1损失 - 针对病灶区域
        if self.lambda_weighted_l1 > 0:
            if hasattr(self, 'weighted_l1_loss'):
                losses['weighted_l1'] = self.weighted_l1_loss(pred, target) * self.lambda_weighted_l1
            else:
                losses['weighted_l1'] = F.l1_loss(pred, target) * self.lambda_weighted_l1  # 回退到普通L1
            total_loss += losses['weighted_l1']
        else:
            losses['weighted_l1'] = torch.tensor(0.0, device=self.device)
        
        # 向后兼容：使用常规L1损失
        if hasattr(self, 'lambda_l1') and self.lambda_l1 > 0 and self.lambda_weighted_l1 == 0.0:
            l1_loss = F.l1_loss(pred, target)
            losses['l1'] = l1_loss * self.lambda_l1
            total_loss += losses['l1']
        else:
            losses['l1'] = torch.tensor(0.0, device=self.device)
        
        # SSIM损失 - 使用统一的torchmetrics实现
        if self.lambda_ssim > 0:
            # 正确处理tanh输出：将[-1,1]映射到[0,1]后再计算SSIM
            def normalize_for_ssim(x):
                # 检测输出范围并正确映射
                x_min, x_max = x.min(), x.max()
                if x_min >= -1.01 and x_max <= 1.01 and (x_min < -0.01 or x_max > 1.01):
                    # 可能是tanh输出，映射[-1,1] -> [0,1]
                    return (x + 1.0) * 0.5
                else:
                    # 已经在[0,1]范围或需要min-max归一化
                    return torch.clamp(x, 0.0, 1.0)
            
            pred_norm = normalize_for_ssim(pred)
            target_norm = normalize_for_ssim(target)
            
            ssim_value = ssim(pred_norm, target_norm, data_range=1.0)
            ssim_loss = 1 - ssim_value
            losses['ssim'] = ssim_loss * self.lambda_ssim
            total_loss += losses['ssim']
        else:
            losses['ssim'] = torch.tensor(0.0, device=self.device)
        
        # 梯度损失
        if self.lambda_grad > 0:
            losses['gradient'] = self.gradient_loss(pred, target) * self.lambda_grad
            total_loss += losses['gradient']
        else:
            losses['gradient'] = torch.tensor(0.0, device=self.device)
        
        # 模态一致性损失
        if self.lambda_consistency > 0 and input_modal is not None:
            losses['consistency'] = self.calculate_consistency_loss(pred, input_modal) * self.lambda_consistency
            total_loss += losses['consistency']
        else:
            losses['consistency'] = torch.tensor(0.0, device=self.device)
        
        # 病灶感知损失
        if self.lambda_lesion_aware > 0:
            if hasattr(self, 'lesion_aware_loss'):
                losses['lesion_aware'] = self.lesion_aware_loss(pred, target) * self.lambda_lesion_aware
            else:
                # 简单的高强度区域L1损失作为回退
                high_intensity_mask = (target > 0.4).float()
                lesion_l1 = torch.abs(pred - target) * high_intensity_mask
                lesion_loss = lesion_l1.sum() / (high_intensity_mask.sum() + 1e-8)
                losses['lesion_aware'] = lesion_loss * self.lambda_lesion_aware
            total_loss += losses['lesion_aware']
        else:
            losses['lesion_aware'] = torch.tensor(0.0, device=self.device)
        
        # TV损失
        if self.lambda_tv > 0:
            losses['tv'] = self.calculate_tv_loss(pred) * self.lambda_tv
            total_loss += losses['tv']
        else:
            losses['tv'] = torch.tensor(0.0, device=self.device)
        
        # 感知损失 (VGG特征)
        if self.lambda_perceptual > 0 and hasattr(self, 'perceptual_loss'):
            losses['perceptual'] = self.perceptual_loss(pred, target) * self.lambda_perceptual
            total_loss += losses['perceptual']
        else:
            losses['perceptual'] = torch.tensor(0.0, device=self.device)
        
        return total_loss, losses
    
    def calculate_consistency_loss(self, pred, input_modal):
        """计算模态一致性损失"""
        if isinstance(input_modal, list) and len(input_modal) > 0:
            # 使用第一个模态计算梯度结构一致性
            input_grad_x = torch.abs(input_modal[0][:, :, :, :-1] - input_modal[0][:, :, :, 1:])
            input_grad_y = torch.abs(input_modal[0][:, :, :-1, :] - input_modal[0][:, :, 1:, :])
            
            pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
            pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
            
            consistency_loss = F.l1_loss(pred_grad_x, input_grad_x) + F.l1_loss(pred_grad_y, input_grad_y)
            return consistency_loss
        return torch.tensor(0.0, device=self.device)
    
    def calculate_tv_loss(self, pred):
        """计算总变分损失"""
        batch_size, channels, height, width = pred.shape
        
        h_tv = torch.pow((pred[:, :, 1:, :] - pred[:, :, :height-1, :]), 2).sum()
        w_tv = torch.pow((pred[:, :, :, 1:] - pred[:, :, :, :width-1]), 2).sum()
        
        count_h = batch_size * channels * (height - 1) * width
        count_w = batch_size * channels * height * (width - 1)
        
        return (h_tv / count_h + w_tv / count_w) / 2.0
    



class WeightedL1Loss(nn.Module):
    """加权L1损失 - 针对病灶区域给予更高权重"""
    
    def __init__(self, focus_threshold=0.3, weight_multiplier=3.0):
        super().__init__()
        self.focus_threshold = focus_threshold
        self.weight_multiplier = weight_multiplier
    
    def forward(self, pred, target):
        # 计算强度梯度，识别重要区域（如病灶）
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # 将梯度填充回原始尺寸
        target_grad_x_padded = F.pad(target_grad_x, (0, 1, 0, 0), mode='replicate')
        target_grad_y_padded = F.pad(target_grad_y, (0, 0, 0, 1), mode='replicate')
        
        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(target_grad_x_padded**2 + target_grad_y_padded**2 + 1e-8)
        
        # 归一化梯度幅值
        grad_min = gradient_magnitude.min()
        grad_max = gradient_magnitude.max()
        if grad_max > grad_min:
            gradient_magnitude = (gradient_magnitude - grad_min) / (grad_max - grad_min + 1e-8)
        
        # 同时考虑高强度区域（病灶通常是高强度）
        high_intensity_mask = (target > self.focus_threshold).float()
        
        # 创建综合权重：梯度区域 + 高强度区域
        importance_map = torch.clamp(gradient_magnitude + high_intensity_mask, 0, 1)
        weights = 1.0 + self.weight_multiplier * importance_map
        
        # 加权L1损失
        l1_loss = torch.abs(pred - target)
        weighted_loss = (l1_loss * weights).mean()
        
        return weighted_loss


class LesionAwareLoss(nn.Module):
    """病灶感知损失 - 专门针对高强度病灶区域"""
    
    def __init__(self, intensity_threshold=0.4, contrast_threshold=0.2):
        super().__init__()
        self.intensity_threshold = intensity_threshold
        self.contrast_threshold = contrast_threshold
    
    def forward(self, pred, target):
        # 识别高强度区域（潜在病灶）
        high_intensity_mask = (target > self.intensity_threshold).float()
        
        # 识别高对比度区域（病灶边缘）
        # 计算局部标准差来检测对比度
        kernel_size = 5
        padding = kernel_size // 2
        
        # 使用卷积计算局部均值
        avg_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=target.device) / (kernel_size * kernel_size)
        if target.shape[1] > 1:
            target_gray = target.mean(dim=1, keepdim=True)
        else:
            target_gray = target
        
        local_mean = F.conv2d(target_gray, avg_kernel, padding=padding)
        local_var = F.conv2d((target_gray - local_mean)**2, avg_kernel, padding=padding)
        local_std = torch.sqrt(local_var + 1e-8)
        
        # 扩展到所有通道
        if target.shape[1] > 1:
            local_std = local_std.expand_as(target)
        
        contrast_mask = (local_std > self.contrast_threshold).float()
        
        # 结合高强度和高对比度区域
        lesion_mask = torch.clamp(high_intensity_mask + contrast_mask * 0.5, 0, 1)
        
        # 病灶区域的L1损失
        lesion_l1 = torch.abs(pred - target) * lesion_mask
        lesion_loss = lesion_l1.sum() / (lesion_mask.sum() + 1e-8)
        
        # 整体L1损失（较小权重）
        global_l1 = torch.abs(pred - target).mean()
        
        # 结合损失：病灶区域权重更高
        return 0.2 * global_l1 + 0.8 * lesion_loss


class EnhancedGradientLoss(nn.Module):
    """增强的梯度损失"""
    
    def forward(self, pred, target):
        # 确保数据类型一致
        if pred.dtype != target.dtype:
            target = target.to(pred.dtype)
        
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # 如果是多通道，转换为单通道
        if pred.shape[1] > 1:
            pred_gray = torch.mean(pred, dim=1, keepdim=True)
            target_gray = torch.mean(target, dim=1, keepdim=True)
        else:
            pred_gray = pred
            target_gray = target
        
        # 计算梯度
        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)
        
        # 梯度幅值
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # L1损失
        gradient_loss = F.l1_loss(pred_grad_mag, target_grad_mag)
        
        return gradient_loss


class AccurateMetrics:
    """统一的评估指标计算 - 确保项目中SSIM和PSNR计算的一致性"""
    
    @staticmethod
    def calculate_psnr(pred, target, data_range=1.0):
        """计算PSNR - 统一使用data_range=1.0"""
        # 转换为numpy进行计算
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # 确保数据在正确范围内
        pred_np = np.clip(pred_np, 0, data_range)
        target_np = np.clip(target_np, 0, data_range)
        
        mse = np.mean((pred_np - target_np) ** 2)
        if mse == 0:
            return 100.0
        return 20 * np.log10(data_range / np.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(pred, target, data_range=1.0):
        """计算SSIM - 使用统一的torchmetrics实现，data_range=1.0"""
        # 正确处理tanh输出：将[-1,1]映射到[0,1]
        def normalize_for_ssim(x):
            x_min, x_max = x.min(), x.max()
            if x_min >= -1.01 and x_max <= 1.01 and (x_min < -0.01 or x_max > 1.01):
                # 可能是tanh输出，映射[-1,1] -> [0,1]
                return (x + 1.0) * 0.5
            else:
                # 已经在[0,1]范围
                return torch.clamp(x, 0.0, data_range)
        
        pred_norm = normalize_for_ssim(pred)
        target_norm = normalize_for_ssim(target)
        
        # 使用torchmetrics的SSIM实现
        ssim_value = ssim(pred_norm, target_norm, data_range=data_range)
        return ssim_value.item() if hasattr(ssim_value, 'item') else float(ssim_value)


class PerceptualLoss(nn.Module):
    """感知损失 - 使用VGG19特征，改善视觉质量"""
    
    def __init__(self, device='cuda', layers=None, vgg_path="/data1/tempf/vgg19-dcbb9e9d.pth"):
        super().__init__()
        self.device = device
        self.vgg_path = vgg_path
        
        # 默认使用VGG19的多个层
        if layers is None:
            layers = ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1']
        self.layers = layers
        
        # 加载预训练VGG19 - 使用自定义路径
        try:
            if os.path.exists(vgg_path):
                # 从本地路径加载
                print(f"[PerceptualLoss] 🎯 从本地加载VGG19权重: {vgg_path}")
                vgg = models.vgg19(pretrained=False)
                state_dict = torch.load(vgg_path, map_location='cpu')
                vgg.load_state_dict(state_dict)
                self.vgg = vgg.features.to(device)
            else:
                # 回退到默认预训练权重
                print(f"[PerceptualLoss] ⚠️ 本地VGG权重未找到，使用默认预训练权重")
                vgg = models.vgg19(pretrained=True).features
                self.vgg = vgg.to(device)
        except Exception as e:
            print(f"[PerceptualLoss] ❌ VGG加载失败: {e}")
            print(f"[PerceptualLoss] 🔄 使用默认预训练权重作为备选")
            vgg = models.vgg19(pretrained=True).features  
            self.vgg = vgg.to(device)
        
        # 冻结VGG参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # VGG层映射
        self.layer_name_mapping = {
            'relu_1_1': '1',
            'relu_2_1': '6',
            'relu_3_1': '11',
            'relu_4_1': '20',
            'relu_5_1': '29'
        }
        
        # 预处理：ImageNet标准化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, pred, target):
        # 确保输入在[0,1]范围
        pred = self.normalize_input(pred)
        target = self.normalize_input(target)
        
        # 转换为3通道（如果需要）
        pred = self.to_3channel(pred)
        target = self.to_3channel(target)
        
        # 确保mean和std在与输入相同的设备上
        device = pred.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        
        # ImageNet标准化
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # 提取特征并计算损失
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # 不同层的权重
        
        for i, layer in enumerate(self.layers):
            if layer in pred_features and layer in target_features:
                weight = weights[i] if i < len(weights) else 1.0
                layer_loss = F.mse_loss(pred_features[layer], target_features[layer])
                loss += weight * layer_loss
        
        return loss
    
    def normalize_input(self, x):
        """将输入标准化到[0,1]范围"""
        x_min, x_max = x.min(), x.max()
        if x_min >= -1.01 and x_max <= 1.01 and (x_min < -0.01 or x_max > 1.01):
            # 可能是tanh输出，映射[-1,1] -> [0,1]
            return (x + 1.0) * 0.5
        else:
            # 已经在[0,1]范围或需要clamp
            return torch.clamp(x, 0.0, 1.0)
    
    def to_3channel(self, x):
        """将输入转换为3通道"""
        if x.shape[1] == 1:
            # 灰度转RGB：重复3次
            return x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            return x
        else:
            # 多通道：取前3个通道
            return x[:, :3, :, :]
    
    def extract_features(self, x):
        """提取VGG特征"""
        features = {}
        temp_x = x
        
        for name, module in self.vgg._modules.items():
            temp_x = module(temp_x)
            for layer_name, layer_idx in self.layer_name_mapping.items():
                if name == layer_idx:
                    features[layer_name] = temp_x
        
        return features
