import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalMemoryEnhancement(nn.Module):

    def __init__(self, feature_dims=[64, 128, 256], memory_sizes=[32, 16, 8]):
        super().__init__()
        self.num_levels = len(feature_dims)
        self.feature_dims = feature_dims
        self.memory_sizes = memory_sizes
        
        # 多尺度记忆编码器
        self.memory_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GroupNorm(8, dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1)
            ) for dim in feature_dims
        ])
        
        # 层级间记忆融合
        self.level_fusion = nn.ModuleList([
            nn.Conv2d(feature_dims[i] + (feature_dims[i-1] if i > 0 else 0), 
                     feature_dims[i], 1)
            for i in range(self.num_levels)
        ])
        
        # 记忆池化策略
        self.memory_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in memory_sizes
        ])
    
    def encode_hierarchical_memory(self, multi_scale_features, masks=None):
        """编码多尺度记忆特征"""
        hierarchical_memories = []
        
        for level, (feat, encoder, pool) in enumerate(zip(
            multi_scale_features, self.memory_encoders, self.memory_pools
        )):
            # 编码当前尺度特征
            encoded_feat = encoder(feat)
            
            # 如果有掩码，应用掩码引导
            if masks is not None:
                mask_resized = F.interpolate(masks, size=feat.shape[-2:], 
                                           mode='bilinear', align_corners=False)
                encoded_feat = encoded_feat * (mask_resized + 0.1)  # 避免完全遮盖
            
            # 池化到记忆尺寸
            memory_feat = pool(encoded_feat)
            hierarchical_memories.append(memory_feat)
        
        return hierarchical_memories
    
    def retrieve_and_fuse_memory(self, current_features, stored_memories):
        """检索并融合分层记忆"""
        enhanced_features = []
        
        for level, (curr_feat, memory_bank) in enumerate(zip(current_features, stored_memories)):
            # 上采样记忆到当前特征尺寸
            if len(memory_bank) > 0:
                # 选择最相关的记忆（简化版本，可扩展为注意力机制）
                relevant_memory = torch.stack(memory_bank).mean(0)  # 平均融合
                upsampled_memory = F.interpolate(
                    relevant_memory, size=curr_feat.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
                
                # 特征融合
                if level > 0:
                    # 包含上一层的信息
                    prev_feat_up = F.interpolate(
                        enhanced_features[level-1], size=curr_feat.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                    fused_input = torch.cat([curr_feat, upsampled_memory, prev_feat_up], dim=1)
                else:
                    fused_input = torch.cat([curr_feat, upsampled_memory], dim=1)
                
                enhanced_feat = self.level_fusion[level](fused_input)
            else:
                enhanced_feat = curr_feat
            
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features
