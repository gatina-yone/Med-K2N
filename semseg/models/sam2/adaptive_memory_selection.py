import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMemorySelector(nn.Module):
    """
    自适应记忆选择机制 - 根据模态重要性动态选择记忆
    """
    def __init__(self, hidden_dim=256, num_modalities=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # 模态重要性评估网络
        self.modality_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 记忆质量评估网络
        self.memory_quality_assessor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 跨模态关联度计算
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
    
    def forward(self, memory_features, current_features, modality_indices):
        """
        Args:
            memory_features: List of [B, C, H, W] from different modalities
            current_features: [B, C, H, W] current modality features
            modality_indices: [B] current modality index
        """
        B, C, H, W = current_features.shape
        selected_memories = []
        memory_weights = []
        
        for i, mem_feat in enumerate(memory_features):
            # 1. 计算模态重要性分数
            mem_global = F.adaptive_avg_pool2d(mem_feat, 1).flatten(1)
            modality_score = self.modality_scorer(mem_global)
            
            # 2. 计算记忆质量分数
            quality_score = self.memory_quality_assessor(mem_feat)
            
            # 3. 计算与当前特征的关联度
            curr_tokens = current_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            mem_tokens = mem_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            
            attn_output, attn_weights = self.cross_modal_attention(
                curr_tokens, mem_tokens, mem_tokens
            )
            correlation_score = attn_weights.mean(dim=(1, 2)).unsqueeze(-1)  # [B, 1]
            
            # 4. 综合权重计算
            final_weight = modality_score * quality_score * correlation_score
            memory_weights.append(final_weight)
            
            # 5. 根据权重选择记忆
            if final_weight.mean() > 0.5:  # 阈值可调
                selected_memories.append(mem_feat * final_weight.view(B, 1, 1, 1))
        
        return selected_memories, memory_weights
