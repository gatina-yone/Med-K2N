import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveMemoryLearning(nn.Module):
    """
    对比学习增强记忆 - 学习模态间的差异性和互补性
    """
    def __init__(self, feature_dim=256, projection_dim=128, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        
        # 特征投影网络
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # 模态特定编码器
        self.modality_encoders = nn.ModuleDict({
            'T2w': self._make_encoder(),
            'T2f': self._make_encoder(), 
            'T1n': self._make_encoder()
        })
        
        # 跨模态预测头
        self.cross_modal_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def _make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
            nn.GroupNorm(16, self.feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, multi_modal_features, modality_names):
        """
        Args:
            multi_modal_features: List of [B, C, H, W] features
            modality_names: List of modality names ['T2w', 'T2f', 'T1n']
        """
        encoded_features = []
        projected_features = []
        
        # 编码各模态特征
        for feat, modal_name in zip(multi_modal_features, modality_names):
            encoded = self.modality_encoders[modal_name](feat)
            projected = self.projector(encoded)
            
            encoded_features.append(encoded)
            projected_features.append(projected)
        
        # 计算对比损失
        contrastive_loss = self.compute_contrastive_loss(projected_features)
        
        # 跨模态预测
        cross_modal_predictions = []
        for i, proj_feat in enumerate(projected_features):
            predicted = self.cross_modal_predictor(proj_feat)
            cross_modal_predictions.append(predicted)
        
        return {
            'encoded_features': encoded_features,
            'projected_features': projected_features,
            'cross_predictions': cross_modal_predictions,
            'contrastive_loss': contrastive_loss
        }
    
    def compute_contrastive_loss(self, projected_features):
        """计算模态间对比损失"""
        if len(projected_features) < 2:
            return torch.tensor(0.0, device=projected_features[0].device)
        
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(projected_features)):
            for j in range(i + 1, len(projected_features)):
                # 正样本：同一样本的不同模态
                pos_sim = F.cosine_similarity(
                    projected_features[i], projected_features[j], dim=1
                ) / self.temperature
                
                # 负样本：不同样本的同模态或不同模态
                batch_size = projected_features[i].shape[0]
                neg_sims = []
                
                for k in range(batch_size):
                    for l in range(batch_size):
                        if k != l:  # 不同样本
                            neg_sim = F.cosine_similarity(
                                projected_features[i][k:k+1], 
                                projected_features[j][l:l+1], 
                                dim=1
                            ) / self.temperature
                            neg_sims.append(neg_sim)
                
                if neg_sims:
                    neg_sims = torch.cat(neg_sims)
                    # InfoNCE损失
                    logits = torch.cat([pos_sim.unsqueeze(1), 
                                      neg_sims.view(batch_size, -1)], dim=1)
                    labels = torch.zeros(batch_size, dtype=torch.long, 
                                       device=logits.device)
                    loss = F.cross_entropy(logits, labels)
                    total_loss += loss
                    num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def get_enhanced_memory(self, current_features, historical_features, 
                          current_modality, historical_modalities):
        """获取对比学习增强的记忆特征"""
        # 当前特征编码
        current_encoded = self.modality_encoders[current_modality](current_features)
        current_projected = self.projector(current_encoded)
        
        enhanced_memories = []
        
        for hist_feat, hist_modal in zip(historical_features, historical_modalities):
            # 历史特征编码
            hist_encoded = self.modality_encoders[hist_modal](hist_feat)
            hist_projected = self.projector(hist_encoded)
            
            # 计算相似度权重
            similarity = F.cosine_similarity(current_projected, hist_projected, dim=1)
            weight = torch.sigmoid(similarity).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            # 加权历史特征
            enhanced_memory = hist_feat * weight
            enhanced_memories.append(enhanced_memory)
        
        return enhanced_memories
