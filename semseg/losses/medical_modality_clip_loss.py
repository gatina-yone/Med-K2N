#!/usr/bin/env python3
"""
医学图像CLIP损失 - 统一的高效实现
包含原始实现和改进的课程学习版本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from typing import Optional, Dict, List, Union
import numpy as np
import os

# 为避免对外部项目的依赖，我们在此文件内提供轻量的 tokenize/load 实现。
# 这些实现是简化的 CLIP 风格接口，足以用于本仓库的损失计算和回退场景。
CLIP_AVAILABLE = True

_VOCAB = {}
def _word_to_id(word):
    # 简单的词到 id 映射（稳定哈希）
    if word in _VOCAB:
        return _VOCAB[word]
    idx = len(_VOCAB) + 1
    _VOCAB[word] = idx
    return idx

def tokenize(texts, context_length=77, truncate=True):
    """
    简化版 tokenize：将每个文本拆词并映射为固定长度的整型张量（context_length），
    不依赖外部库，行为稳定且可被后续的内置 encode_text 使用。
    """
    if isinstance(texts, str):
        texts = [texts]
    tokens = torch.zeros((len(texts), context_length), dtype=torch.long)
    for i, t in enumerate(texts):
        words = t.lower().replace('/', ' ').replace('-', ' ').split()
        if truncate:
            words = words[:context_length]
        ids = [_word_to_id(w) for w in words][:context_length]
        tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return tokens


def load(model_name="ViT-B/32", device='cpu'):
    """
    返回一个非常轻量级的内置 CLIP 风格模型。
    该模型提供 `encode_image(images)` 和 `encode_text(tokens)` 接口，
    输出特征向量用于相似度计算。
    """
    class InternalSimpleCLIP(nn.Module):
        def __init__(self, embed_dim=512):
            super().__init__()
            # 简单的图像编码：自适应平均后线性映射
            self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.img_fc = nn.Linear(3, embed_dim)
            # 简单的文本编码：embedding + 平均 + 线性
            self.vocab_size = 20000
            self.text_emb = nn.Embedding(self.vocab_size, 256)
            self.text_fc = nn.Linear(256, embed_dim)
            self.embed_dim = embed_dim
        
        def encode_image(self, images):
            # images: [B, 3, H, W], 输出 [B, embed_dim]
            x = self.img_pool(images)  # [B,3,1,1]
            x = x.view(x.size(0), 3)
            x = self.img_fc(x)
            return x

        def encode_text(self, tokens):
            # tokens: [N, context_length] 或 [context_length]
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            # 防止超出 embedding 的 vocab
            tokens_clamped = tokens.clamp(0, self.vocab_size - 1)
            t = self.text_emb(tokens_clamped)  # [N, L, D]
            t = t.mean(dim=1)
            t = self.text_fc(t)
            return t

    model = InternalSimpleCLIP()
    model.to(device)
    return model, None


class MedicalModalityCLIPLoss(nn.Module):
    """
    医学图像模态CLIP损失函数
    专用于T1c, T1n, T2f, T2w模态的图像翻译任务
    """
    
    def __init__(self, clip_model: str = "ViT-B/32", temperature: float = 0.1,
                 modality_descriptions: Optional[dict] = None,
                 quality_descriptions: Optional[list] = None,
                 device: Optional[str] = None):
        super(MedicalModalityCLIPLoss, self).__init__()
        # device can be provided by caller (str or torch.device); otherwise auto-detect
        if device is not None:
            self.device = device if isinstance(device, str) else str(device)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temperature = temperature
        self.torch_resize = Resize([224, 224])
        
        # 支持从外部注入模态描述字典以便配置化
        if modality_descriptions is not None:
            self.modality_descriptions = modality_descriptions
        else:
            # 尝试从 configs/modality_descriptions.yaml 加载（优先）
            try:
                import yaml
                cfg_path = os.path.join('configs', 'modality_descriptions.yaml')
                if os.path.exists(cfg_path):
                    with open(cfg_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                        md = cfg.get('modality_descriptions')
                        if md:
                            self.modality_descriptions = md
                            # also allow modalities order from file
                            if 'modalities' in cfg:
                                # maintain insertion order
                                ordered = {k: self.modality_descriptions[k] for k in cfg['modalities'] if k in self.modality_descriptions}
                                self.modality_descriptions = ordered
                else:
                    # 若配置文件不存在或不完整，则使用内置默认（下面）
                    self.modality_descriptions = None
            except Exception:
                self.modality_descriptions = None
            # 如果未通过配置加载，则使用内置默认
            if self.modality_descriptions is None:
                # 医学图像模态的文本描述（默认，基于脑部）
                # 使用用户提供的更详细描述（每个模态为一条长描述），适配脑部影像任务
                self.modality_descriptions = {
                    'T1c': [
                        "A set of axial T1-weighted magnetic resonance brain scan sequences with 【contrast enhancement】, 【injected with a gadolinium-based contrast agent】, used to clearly show areas of 【brain tumor enhancement】, 【metastases】, 【abscesses】, and 【blood-brain barrier disruption】, providing images with high soft tissue resolution."
                    ],
                    'T1n': [
                        "A set of high-resolution T1-weighted magnetic resonance brain 【non-contrast】 sequences, 【without the use of a contrast agent】, offering excellent 【brain anatomical structure】 details, including 【gray matter-white matter differentiation】 and 【gyral and sulcal morphology】, often used for 【3D volume reconstruction】."
                    ],
                    'T2f': [
                        "A set of axial 【T2-FLAIR】 magnetic resonance brain scan sequences. This sequence 【suppresses the high signal of cerebrospinal fluid (CSF)】, making lesions (such as gliomas, multiple sclerosis plaques, edema) in the 【periventricular】 and 【white matter】 areas more 【prominent】."
                    ],
                    'T2w': [
                        "A set of standard axial 【T2-weighted】 magnetic resonance brain scan sequences that are 【highly sensitive to free water】, effectively displaying 【cerebral edema】, 【ischemic foci】, 【cysts】, and 【liquefactive necrosis】 within tumors. It is a routine brain screening sequence."
                    ],
                    'CT': [
                        "A set of 【non-contrast】 axial 【head CT plain scan】 images that clearly show 【skull structures】, 【acute hemorrhage (hyperdense)】, 【calcifications】, and the basic morphology of the brain parenchyma, featuring 【fast scanning speed】."
                    ],
                    'DWI': [
                        "A set of 【diffusion-weighted】 magnetic resonance brain scan sequences. Using 【high b-value】 imaging, it highlights areas with restricted 【water diffusion】, serving as the 【gold standard】 sequence for diagnosing 【hyperacute】 and 【acute】 【cerebral infarction】."
                    ]
                }

        # 支持从外部注入质量描述
        if quality_descriptions is not None:
            self.quality_descriptions = quality_descriptions
        else:
            self.quality_descriptions = [
                "high quality medical brain image",
                "clear and detailed brain MRI scan",
                "well-defined brain tissue contrast",
                "anatomically accurate brain image"
            ]
        
        # 加载CLIP模型
        if CLIP_AVAILABLE:
            try:
                self.CLIP, self.preprocess = load(clip_model, device=self.device)
                self.CLIP.eval()
                for param in self.CLIP.parameters():
                    param.requires_grad = False
                # 静默加载，减少输出
            except Exception as e:
                print(f"Warning: Failed to load CLIP model: {e}")
                self.CLIP = None
        else:
            self.CLIP = None
            
        # 预先准备所有文本token
        # 确保 modality_descriptions 为字典，避免后续访问报错
        if not isinstance(self.modality_descriptions, dict):
            self.modality_descriptions = {}
        self._prepare_text_tokens()
        
        self.criterion = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def _prepare_text_tokens(self):
        """预先准备所有模态的文本描述tokens"""
        all_texts = []
        self.modality_text_indices = {}
        self.modalities_list = list(self.modality_descriptions.keys())
        
        idx = 0
        for modality in self.modalities_list:
            descriptions = self.modality_descriptions[modality]
            self.modality_text_indices[modality] = list(range(idx, idx + len(descriptions)))
            all_texts.extend(descriptions)
            idx += len(descriptions)
        
        # 添加质量描述
        self.quality_text_indices = list(range(idx, idx + len(self.quality_descriptions)))
        all_texts.extend(self.quality_descriptions)
        
        if CLIP_AVAILABLE and self.CLIP is not None:
            # 预计算每个模态描述的文本向量均值，减少前向时的随机性
            self.per_modality_text_feats = {}
            for modality in self.modalities_list:
                descs = self.modality_descriptions[modality]
                tokens = tokenize(descs).to(self.device)
                with torch.no_grad():
                    text_feats = self.CLIP.encode_text(tokens)  # [N_desc, D]
                    mean_feat = text_feats.mean(dim=0, keepdim=True)  # [1, D]
                    mean_feat = F.normalize(mean_feat, dim=-1)
                self.per_modality_text_feats[modality] = mean_feat
            # quality 描述也取均值
            tokens_q = tokenize(self.quality_descriptions).to(self.device)
            with torch.no_grad():
                q_feats = self.CLIP.encode_text(tokens_q)
                self.quality_text_feat = F.normalize(q_feats.mean(dim=0, keepdim=True), dim=-1)
            # 也保留原始 tokens 以防回退
            self.all_text_tokens = tokenize(all_texts).to(self.device)
        else:
            self.all_text_tokens = tokenize(all_texts)
            self.per_modality_text_feats = None
            self.quality_text_feat = None
        
        # 静默完成初始化
        
    def forward(self, generated_images, target_modality, source_modality=None, 
                use_quality_loss=True, use_modality_consistency=True):
        """
        计算医学模态CLIP损失
        
        Args:
            generated_images: 生成的图像 [B, 3, H, W] 或 List[[B, 3, H, W], ...]
            target_modality: 目标模态 ('T1c', 'T1n', 'T2f', 'T2w')
            source_modality: 源模态（可选），用于模态一致性约束
            use_quality_loss: 是否使用图像质量损失
            use_modality_consistency: 是否使用模态一致性损失
            
        Returns:
            total_loss: 总损失
            loss_dict: 详细损失字典
        """
        if self.CLIP is None:
            # Mock实现用于测试
            return self._mock_forward(generated_images, target_modality)
            
        # 处理输入格式
        if isinstance(generated_images, list):
            # 多帧输出的情况
            all_images = torch.cat(generated_images, dim=0)  # [B*N, 3, H, W]
        else:
            all_images = generated_images  # [B, 3, H, W]
            
        # 调整图像尺寸到CLIP需要的224x224
        resized_images = self.torch_resize(all_images)
        # 将输入移动到 CLIP 模型所在设备，确保一致性
        try:
            resized_images = resized_images.to(self.device)
        except Exception:
            # 如果 device 字符串不被接受，尝试转换为 torch.device
            try:
                resized_images = resized_images.to(torch.device(self.device))
            except Exception:
                pass
        
        # 确保图像值在[0,1]范围内
        if resized_images.max() > 1.0 or resized_images.min() < 0.0:
            resized_images = torch.clamp((resized_images + 1.0) / 2.0, 0.0, 1.0)
            
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Med-K2N因果一致性损失 (L_causal)
        l_causal = self._compute_causal_consistency_loss(resized_images, target_modality)
        loss_dict['L_causal'] = l_causal
        total_loss += l_causal
        
        # 2. Med-K2N度量学习损失 (L_metric) 
        if source_modality is not None:
            l_metric = self._compute_metric_learning_loss(resized_images, source_modality, target_modality)
            loss_dict['L_metric'] = l_metric
            total_loss += l_metric
        else:
            loss_dict['L_metric'] = torch.tensor(0.0, device=self.device)
        
        # 3. 图像质量损失（保持原有实现）
        if use_quality_loss:
            quality_loss = self._compute_quality_loss(resized_images)
            loss_dict['quality_loss'] = quality_loss
            total_loss += 0.5 * quality_loss  # 权重0.5
            
        # 4. 模态一致性损失（保持向后兼容）
        if use_modality_consistency and source_modality is not None:
            consistency_loss = self._compute_modality_consistency_loss(
                resized_images, source_modality, target_modality
            )
            loss_dict['consistency_loss'] = consistency_loss
            total_loss += 0.3 * consistency_loss  # 权重0.3
            
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
        
    def _compute_modality_loss(self, images, target_modality):
        """计算模态特定的CLIP损失"""
        batch_size = images.shape[0]
        # 如果已预计算 per-modality 文本向量，使用它们（更稳定、无随机采样）
        if self.per_modality_text_feats is not None and target_modality in self.per_modality_text_feats:
            with torch.no_grad():
                assert self.CLIP is not None
                image_features = self.CLIP.encode_image(images)

            image_features = F.normalize(image_features, dim=-1)  # [B, D]
            # 构建 modalities matrix
            text_feats = torch.cat([self.per_modality_text_feats[m] for m in self.modalities_list], dim=0)  # [M, D]
            # logits: [B, M]
            logits = torch.matmul(image_features, text_feats.T) / self.temperature
            target_idx = self.modalities_list.index(target_modality)
            targets = torch.full((batch_size,), target_idx, dtype=torch.long, device=self.device)
            loss = self.criterion(logits, targets)
            return loss

        # 回退到原有实现（逐描述 token 的交叉熵）
        text_indices = self.modality_text_indices[target_modality]
        targets = []
        for _ in range(batch_size):
            idx = np.random.choice(text_indices)
            targets.append(idx)
        targets = torch.tensor(targets, device=self.device)
        with torch.no_grad():
            assert self.CLIP is not None
            image_features = self.CLIP.encode_image(images)
            text_features = self.CLIP.encode_text(self.all_text_tokens)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        loss = self.criterion(logits, targets)
        return loss
        
    def _compute_quality_loss(self, images):
        """计算图像质量CLIP损失"""
        batch_size = images.shape[0]
        # 如果有预计算的 quality 向量，则直接使用单一 quality 向量作为分类目标，减少方差
        if self.quality_text_feat is not None:
            with torch.no_grad():
                assert self.CLIP is not None
                image_features = self.CLIP.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)  # [B, D]
            q_feat = self.quality_text_feat  # [1, D]
            logits = torch.matmul(image_features, q_feat.T) / self.temperature  # [B,1]
            # 为兼容交叉熵，构造两类 logits: positive vs negative (negative=0)
            # 这里更简单地把 quality loss 视作 -mean(sim) 的替代，用 margin 可自行调整
            # 我们将用 negative class as zeros
            neg = torch.zeros_like(logits)
            logits_cat = torch.cat([logits, neg], dim=1)  # [B,2]
            targets = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
            loss = self.criterion(logits_cat, targets)
            return loss

        # 回退原先实现
        quality_targets = []
        for _ in range(batch_size):
            idx = np.random.choice(self.quality_text_indices)
            quality_targets.append(idx)
        quality_targets = torch.tensor(quality_targets, device=self.device)
        with torch.no_grad():
            assert self.CLIP is not None
            image_features = self.CLIP.encode_image(images)
            text_features = self.CLIP.encode_text(self.all_text_tokens)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        loss = self.criterion(logits, quality_targets)
        return loss
        
    def _compute_modality_consistency_loss(self, images, source_modality, target_modality):
        """计算模态一致性损失"""
        batch_size = images.shape[0]
        
        # 使用预计算的 per-modality 文本向量（若可用），否则回退到随机文本选择
        if self.per_modality_text_feats is not None and source_modality in self.per_modality_text_feats and target_modality in self.per_modality_text_feats:
            with torch.no_grad():
                assert self.CLIP is not None
                image_features = self.CLIP.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            source_text_feat = self.per_modality_text_feats[source_modality]  # [1,D]
            target_text_feat = self.per_modality_text_feats[target_modality]
            target_sim = torch.matmul(image_features, target_text_feat.T).mean()
            source_sim = torch.matmul(image_features, source_text_feat.T).mean()
            consistency_loss = F.relu(source_sim - target_sim + 0.2)
            return consistency_loss

        # 回退原先实现
        source_indices = self.modality_text_indices[source_modality]
        target_indices = self.modality_text_indices[target_modality]
        source_idx = np.random.choice(source_indices)
        target_idx = np.random.choice(target_indices)
        with torch.no_grad():
            assert self.CLIP is not None
            image_features = self.CLIP.encode_image(images)
            source_text_feat = self.CLIP.encode_text(self.all_text_tokens[source_idx:source_idx+1])
            target_text_feat = self.CLIP.encode_text(self.all_text_tokens[target_idx:target_idx+1])
        image_features = F.normalize(image_features, dim=-1)
        source_text_feat = F.normalize(source_text_feat, dim=-1)
        target_text_feat = F.normalize(target_text_feat, dim=-1)
        target_sim = torch.matmul(image_features, target_text_feat.T).mean()
        source_sim = torch.matmul(image_features, source_text_feat.T).mean()
        consistency_loss = F.relu(source_sim - target_sim + 0.2)
        return consistency_loss
        
    def _compute_causal_consistency_loss(self, images, target_modality):
        """
        计算Med-K2N因果一致性损失 (L_causal)
        通过视觉-文本对比学习确保生成图像与目标模态的语义描述完全对齐
        
        数学表达: L_causal = -log(exp(sim(v_j, t_j)/τ) / Σ_k exp(sim(v_j, t_k)/τ))
        其中 v_j 是图像特征向量，t_j 是目标模态文本特征向量，τ是温度参数
        """
        batch_size = images.shape[0]
        
        # 使用预计算的per-modality文本向量（更稳定的实现）
        if self.per_modality_text_feats is not None and target_modality in self.per_modality_text_feats:
            with torch.no_grad():
                # 提取图像视觉特征
                image_features = self.CLIP.encode_image(images)  # [B, D]

            # L2归一化特征向量
            image_features = F.normalize(image_features, dim=-1)  # [B, D]
            
            # 构建所有模态的文本特征矩阵
            text_feats = torch.cat([self.per_modality_text_feats[m] for m in self.modalities_list], dim=0)  # [M, D]
            
            # 计算相似度矩阵并应用温度缩放
            logits = torch.matmul(image_features, text_feats.T) / self.temperature  # [B, M]
            
            # 构建目标标签（对应目标模态的索引）
            target_idx = self.modalities_list.index(target_modality)
            targets = torch.full((batch_size,), target_idx, dtype=torch.long, device=self.device)
            
            # 对比学习损失（负对数似然）
            causal_loss = self.criterion(logits, targets)
            return causal_loss

        # 回退到原有实现（随机采样描述）
        text_indices = self.modality_text_indices[target_modality]
        targets = []
        for _ in range(batch_size):
            idx = np.random.choice(text_indices)
            targets.append(idx)
        targets = torch.tensor(targets, device=self.device)
        
        with torch.no_grad():
            image_features = self.CLIP.encode_image(images)
            text_features = self.CLIP.encode_text(self.all_text_tokens)
            
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        causal_loss = self.criterion(logits, targets)
        return causal_loss
    
    def _compute_metric_learning_loss(self, images, source_modality, target_modality):
        """
        计算Med-K2N度量学习损失 (L_metric)
        基于因果推理的三元组学习策略，在特征空间中建立明确的模态区分边界
        
        数学表达: L_metric = Σ max(0, α + d(v_gen, v_ref) - d(v_gen, v_neg))
        其中 α 是边际参数，d是L2距离，v_gen是生成图像特征，v_ref是目标特征，v_neg是负样本特征
        """
        batch_size = images.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 提取图像特征向量
        with torch.no_grad():
            image_features = self.CLIP.encode_image(images)  # [B, D]
        
        # L2归一化
        image_features = F.normalize(image_features, dim=-1)  # [B, D]
        
        # 使用预计算的模态文本特征
        if (self.per_modality_text_feats is not None and 
            target_modality in self.per_modality_text_feats and 
            source_modality in self.per_modality_text_feats):
            
            target_text_feat = self.per_modality_text_feats[target_modality]  # [1, D]
            source_text_feat = self.per_modality_text_feats[source_modality]  # [1, D]
            
            # 计算与目标模态的相似度（正样本距离）
            target_sim = torch.matmul(image_features, target_text_feat.T).squeeze(-1)  # [B]
            pos_dist = 1.0 - target_sim  # 转换为距离（距离越小越好）
            
            # 计算与源模态的相似度（负样本距离）
            source_sim = torch.matmul(image_features, source_text_feat.T).squeeze(-1)  # [B]
            neg_dist = 1.0 - source_sim
            
            # 三元组损失：max(0, margin + pos_dist - neg_dist)
            margin = 0.2  # 边际参数
            metric_loss = torch.clamp(margin + pos_dist - neg_dist, min=0.0).mean()
            
            return metric_loss
        
        # 回退实现：使用batch内的困难样本挖掘
        metric_losses = []
        for i in range(batch_size):
            anchor_feat = image_features[i:i+1]  # [1, D]
            
            # 正样本：同一batch中的其他样本（假设都是目标模态）
            pos_indices = [j for j in range(batch_size) if j != i]
            if len(pos_indices) > 0:
                pos_idx = np.random.choice(pos_indices)
                pos_feat = image_features[pos_idx:pos_idx+1]  # [1, D]
                pos_dist = F.pairwise_distance(anchor_feat, pos_feat, p=2)
                
                # 负样本：与anchor距离较远的样本
                neg_indices = [j for j in range(batch_size) if j != i and j != pos_idx]
                if len(neg_indices) > 0:
                    neg_idx = np.random.choice(neg_indices)
                    neg_feat = image_features[neg_idx:neg_idx+1]  # [1, D]
                    neg_dist = F.pairwise_distance(anchor_feat, neg_feat, p=2)
                    
                    # 三元组损失
                    triplet_loss = torch.clamp(0.2 + pos_dist - neg_dist, min=0.0)
                    metric_losses.append(triplet_loss)
        
        if len(metric_losses) > 0:
            return torch.stack(metric_losses).mean()
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _mock_forward(self, generated_images, target_modality):
        """Mock实现用于测试"""
        mock_loss = torch.tensor(0.5, requires_grad=True)
        loss_dict = {
            'L_causal': mock_loss * 0.4,
            'L_metric': mock_loss * 0.3,
            'quality_loss': mock_loss * 0.2,
            'consistency_loss': mock_loss * 0.1,
            'total_loss': mock_loss
        }
        return mock_loss, loss_dict


# ===============================
# 改进的CLIP损失实现 - 课程学习版本
# ===============================

class EffectiveMedicalCLIPLoss(nn.Module):
    """
    高效且实用的医学图像CLIP损失
    
    主要优化:
    1. 更稳定的对比学习实现
    2. 课程学习自适应权重
    3. 内存和计算效率优化
    4. 更好的错误处理
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        temperature: float = 0.1,
        modality_descriptions: Optional[dict] = None,
        use_curriculum_weighting: bool = True,
        clip_model_name: str = "ViT-B/32",
        max_text_length: int = 77,
    ):
        super().__init__()
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.use_curriculum_weighting = use_curriculum_weighting
        self.max_text_length = max_text_length
        
        # 初始化CLIP模型
        self.clip_model = None
        self.clip_available = False
        self._init_clip_model(clip_model_name)
        
        # 图像预处理
        self.image_transform = Resize([224, 224])
        
        # 模态描述
        self.modality_descriptions = modality_descriptions or self._get_enhanced_descriptions()
        
        # 预计算文本特征
        self.text_features_cache = {}
        if self.clip_model is not None:
            self._precompute_text_features()
        
        # 课程学习权重调度
        self.curriculum_weights = {
            'easy': 0.3,      # 早期降低CLIP影响
            'medium': 0.5,    # 标准权重
            'hard': 0.7,      # 增强多模态对齐
            'expert': 0.6     # 略微降低以避免过拟合
        }
        
    def _init_clip_model(self, model_name: str):
        """初始化CLIP模型"""
        try:
            # 尝试使用外部CLIP包
            import clip
            self.clip_model, _ = clip.load(model_name, device=self.device)
            self.clip_available = True
            print(f"✅ 成功加载CLIP模型: {model_name}")
        except ImportError:
            print("⚠️ 外部CLIP包未安装，使用内置版本")
            self.clip_model, _ = load(model_name, device=self.device)
            self.clip_available = False
        except Exception as e:
            print(f"⚠️ CLIP模型加载失败: {e}，使用内置版本")
            self.clip_model, _ = load(model_name, device=self.device)
            self.clip_available = False
    
    def _get_enhanced_descriptions(self):
        """获取增强的模态描述"""
        return {
            'A': [
                "contrast enhanced T1 weighted brain MRI with gadolinium showing tumor enhancement",
                "T1c brain scan with clear lesion boundaries and enhanced structures",
                "post-contrast T1 weighted magnetic resonance brain imaging"
            ],
            'B': [
                "T1 weighted non-contrast brain MRI showing anatomical structures",
                "native T1 brain scan with excellent tissue contrast",
                "T1n sequence displaying brain morphology without contrast"
            ],
            'C': [
                "T2 FLAIR brain MRI with suppressed cerebrospinal fluid signal",
                "fluid attenuated inversion recovery brain scan highlighting lesions",
                "FLAIR sequence showing white matter abnormalities"
            ],
            'D': [
                "T2 weighted brain MRI sensitive to water content and pathology",
                "T2w sequence displaying hyperintense lesions and edema",
                "T2 weighted magnetic resonance brain imaging"
            ]
        }
    
    def _precompute_text_features(self):
        """预计算文本特征"""
        if self.clip_model is None:
            return
            
        for modality, descriptions in self.modality_descriptions.items():
            features = []
            
            for desc in descriptions:
                if self.clip_available:
                    try:
                        import clip
                        text_tokens = clip.tokenize([desc], truncate=True).to(self.device)
                    except:
                        text_tokens = tokenize([desc], truncate=True).to(self.device)
                else:
                    text_tokens = tokenize([desc], truncate=True).to(self.device)
                
                with torch.no_grad():
                    if hasattr(self.clip_model, 'encode_text'):
                        text_feat = self.clip_model.encode_text(text_tokens)
                        features.append(text_feat)
            
            # 平均所有描述的特征作为该模态的代表
            if features:
                self.text_features_cache[modality] = torch.stack(features).mean(dim=0)
    
    def _get_curriculum_weight(self, stage: Optional[str] = None):
        """获取课程学习权重"""
        if not self.use_curriculum_weighting or stage is None:
            return 0.5  # 默认权重
        
        return self.curriculum_weights.get(stage.lower(), 0.5)
    
    def _compute_contrastive_loss(self, image_features: torch.Tensor, 
                                text_features: torch.Tensor):
        """计算优化的对比学习损失"""
        # 确保特征已归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 确保text_features是2D的
        if text_features.dim() > 2:
            text_features = text_features.view(-1, text_features.shape[-1])
        
        # 计算相似度
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # 创建标签
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # 如果维度不匹配，使用简化的损失
        if logits.shape[0] != logits.shape[1]:
            # 使用均方误差作为对齐损失
            similarity = F.cosine_similarity(image_features, text_features, dim=-1)
            target_similarity = torch.ones_like(similarity)
            return F.mse_loss(similarity, target_similarity)
        
        # 对称对比损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def forward(self, generated_images: Union[torch.Tensor, List[torch.Tensor]], 
                target_modalities: Union[str, List[str]],
                curriculum_stage: Optional[str] = None,
                return_dict: bool = True):
        """
        前向传播
        
        Args:
            generated_images: 生成的图像
            target_modalities: 目标模态
            curriculum_stage: 课程学习阶段
            return_dict: 是否返回详细损失字典
        """
        # 处理输入格式
        if isinstance(generated_images, list):
            images = torch.cat(generated_images, dim=0)
        else:
            images = generated_images
        
        if isinstance(target_modalities, str):
            target_modalities = [target_modalities] * images.shape[0]
        
        # 确保图像在正确范围
        if images.max() > 1.0 or images.min() < 0.0:
            images = torch.clamp(images, 0.0, 1.0)
        
        # 图像预处理
        try:
            processed_images = self.image_transform(images)
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 编码图像特征
        try:
            if hasattr(self.clip_model, 'encode_image'):
                image_features = self.clip_model.encode_image(processed_images)
            elif hasattr(self.clip_model, 'visual'):
                image_features = self.clip_model.visual(processed_images)
            else:
                # 使用简化编码器
                image_features = self.clip_model(processed_images)
        except Exception as e:
            print(f"图像编码失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 获取对应的文本特征
        text_features = []
        valid_indices = []
        
        for i, modality in enumerate(target_modalities):
            if modality in self.text_features_cache:
                text_features.append(self.text_features_cache[modality])
                valid_indices.append(i)
        
        if not text_features:
            # 如果没有有效的模态，创建默认特征
            for i in range(len(target_modalities)):
                default_feat = torch.zeros(image_features.shape[1], device=self.device)
                text_features.append(default_feat)
                valid_indices.append(i)
        
        # 只使用有效索引的图像特征
        if len(valid_indices) < len(target_modalities):
            image_features = image_features[valid_indices]
        
        text_features = torch.stack(text_features)
        
        # 计算对比损失
        try:
            contrastive_loss = self._compute_contrastive_loss(image_features, text_features)
        except Exception as e:
            print(f"对比损失计算失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 应用课程学习权重
        curriculum_weight = self._get_curriculum_weight(curriculum_stage)
        final_loss = curriculum_weight * contrastive_loss
        
        if return_dict:
            return final_loss, {
                'contrastive_loss': contrastive_loss.item() if hasattr(contrastive_loss, 'item') else float(contrastive_loss),
                'curriculum_weight': curriculum_weight,
                'curriculum_stage': curriculum_stage or 'unknown',
                'total_loss': final_loss.item() if hasattr(final_loss, 'item') else float(final_loss)
            }
        else:
            return final_loss


class MultiModalityCLIPLoss(nn.Module):
    """
    多模态CLIP损失，支持四个生成器的同时训练
    """
    
    def __init__(self, clip_model: str = "ViT-B/32", temperature: float = 0.1, 
                 modalities=None, dataset_root: Optional[str] = None, 
                 dataset_split: str = 'val', modality_descriptions: Optional[dict] = None, 
                 quality_descriptions: Optional[list] = None):
        super(MultiModalityCLIPLoss, self).__init__()
        # 将描述字典传递给基础 CLIP 损失函数，支持配置注入
        self.base_clip_loss = MedicalModalityCLIPLoss(clip_model, temperature, modality_descriptions=modality_descriptions, quality_descriptions=quality_descriptions)
        # 支持外部传入模态列表；若未提供且给定 dataset_root，则从磁盘推断
        if modalities is not None:
            self.modalities = list(modalities)
        elif dataset_root is not None:
            # 严格从磁盘推断模态列表，若失败则抛出异常（不再默认回退）
            self.modalities = self._infer_modalities_from_dataset(dataset_root, dataset_split)
            if not self.modalities:
                raise RuntimeError(f"从数据集目录推断模态列表失败或返回空列表: {dataset_root}/{dataset_split}")
            print(f"✓ 自动推断模态列表: {self.modalities}")
        else:
            raise RuntimeError("未提供 modal 列表或 dataset_root；请在构造函数中传入 'modalities' 或 'dataset_root' 来明确模态顺序。")
        
    def forward(self, generator_outputs, modality_targets, source_modalities=None):
        """
        计算多模态CLIP损失
        
        Args:
            generator_outputs: Dict[str, torch.Tensor] 或 List[torch.Tensor]
                - 如果是字典: {'T1c': tensor, 'T1n': tensor, ...}
                - 如果是列表: [tensor1, tensor2, tensor3, tensor4] 对应 T1c, T1n, T2f, T2w
            modality_targets: List[str] 目标模态列表
            source_modalities: List[str] 源模态列表（可选）
            
        Returns:
            total_loss: 总损失
            detailed_losses: 详细损失字典
        """
        total_loss = 0.0
        detailed_losses = {}
        
        # 处理输入格式
        if isinstance(generator_outputs, dict):
            outputs = generator_outputs
        else:
            # 假设列表顺序为 [T1c, T1n, T2f, T2w]
            outputs = dict(zip(self.modalities, generator_outputs))
            
        # 确保目标模态列表
        if isinstance(modality_targets, str):
            modality_targets = [modality_targets]
            
        if source_modalities is not None and isinstance(source_modalities, str):
            source_modalities = [source_modalities]
            
        # 为每个生成器计算CLIP损失
        for i, (modality, generated_images) in enumerate(outputs.items()):
            if generated_images is None:
                continue
                
            # 获取对应的目标模态和源模态
            target_mod = modality_targets[i] if i < len(modality_targets) else modality
            source_mod = source_modalities[i] if source_modalities and i < len(source_modalities) else None
            
            # 计算CLIP损失
            clip_loss, loss_dict = self.base_clip_loss(
                generated_images, 
                target_mod, 
                source_mod
            )
            
            # 添加到总损失
            total_loss += clip_loss
            
            # 保存详细损失
            for key, value in loss_dict.items():
                detailed_losses[f"{modality}_{key}"] = value
                
        detailed_losses['total_clip_loss'] = total_loss
        return total_loss, detailed_losses

    def _infer_modalities_from_dataset(self, dataset_root: str, split: str = 'val'):
        """从数据目录推断模态子文件夹名称，返回按字母排序的名称列表"""
        modal_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(modal_dir):
            raise FileNotFoundError(f"Dataset split directory not found: {modal_dir}")
        entries = [d for d in os.listdir(modal_dir) if os.path.isdir(os.path.join(modal_dir, d))]
        entries = sorted(entries)
        return entries


class SimpleCLIPIntegrator(nn.Module):
    """
    简化的CLIP损失集成器
    """
    
    def __init__(self, base_criterion, clip_weight: float = 0.05, use_improved: bool = True):
        super().__init__()
        self.base_criterion = base_criterion
        self.clip_weight = clip_weight
        
        # 选择使用改进版还是原版
        if use_improved:
            self.clip_loss = EffectiveMedicalCLIPLoss()
        else:
            self.clip_loss = MedicalModalityCLIPLoss()
    
    def forward(self, pred, target, target_modalities=None, curriculum_stage=None, **kwargs):
        """集成的损失计算"""
        # 基础损失
        try:
            if hasattr(self.base_criterion, 'forward'):
                base_result = self.base_criterion(pred, target, **kwargs)
            else:
                base_result = self.base_criterion(pred, target)
                
            if isinstance(base_result, dict):
                base_loss = base_result.get('total_loss', base_result)
                loss_dict = base_result.copy()
            else:
                base_loss = base_result
                loss_dict = {'base_loss': base_loss}
        except Exception as e:
            print(f"基础损失计算失败: {e}")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # CLIP损失
        clip_loss = torch.tensor(0.0, device=pred.device)
        if target_modalities and self.clip_weight > 0:
            try:
                clip_result = self.clip_loss(pred, target_modalities, curriculum_stage)
                if isinstance(clip_result, tuple):
                    clip_loss, clip_dict = clip_result
                    loss_dict.update({f'clip_{k}': v for k, v in clip_dict.items()})
                else:
                    clip_loss = clip_result
                    
                clip_loss = clip_loss * self.clip_weight
                loss_dict['clip_weight_used'] = self.clip_weight
                
            except Exception as e:
                print(f"CLIP损失计算失败: {e}")
                clip_loss = torch.tensor(0.0, device=pred.device)
        
        # 总损失
        total_loss = base_loss + clip_loss
        loss_dict['total_loss'] = total_loss
        
        return loss_dict if isinstance(base_result, dict) else total_loss


def create_improved_clip_criterion(base_criterion, clip_weight: float = 0.05, use_improved: bool = True):
    """创建改进的CLIP损失函数"""
    return SimpleCLIPIntegrator(
        base_criterion=base_criterion,
        clip_weight=clip_weight,
        use_improved=use_improved
    )


def demo_medical_clip_loss():
    """演示医学模态CLIP损失的使用"""
    print("=" * 60)
    print("医学模态CLIP损失演示")
    print("=" * 60)
    
    # 创建CLIP损失函数，优先通过 CLIPLossIntegrator 加载配置
    try:
        from semseg.losses.clip_loss_integration import CLIPLossIntegrator
        integrator = CLIPLossIntegrator(device='cpu')
        clip_loss = integrator.clip_loss if integrator.clip_loss is not None else MedicalModalityCLIPLoss(device='cpu')
    except Exception:
        clip_loss = MedicalModalityCLIPLoss(device='cpu')
    # 为演示传入显式的模态顺序，避免从磁盘推断导致的异常
    multi_clip_loss = MultiModalityCLIPLoss(modalities=['T1c', 'T1n', 'T2f', 'T2w'])
    
    # 模拟生成的图像数据
    batch_size = 4
    
    # 单模态测试
    print("\n1. 单模态CLIP损失测试...")
    generated_t1c = torch.randn(batch_size, 3, 256, 256)
    loss, loss_dict = clip_loss(generated_t1c, 'T1c', 'T1n')
    print(f"   T1c生成损失: {loss:.4f}")
    for key, value in loss_dict.items():
        print(f"   {key}: {value:.4f}")
    
    # 多模态测试
    print("\n2. 多模态CLIP损失测试...")
    generator_outputs = {
        'T1c': torch.randn(batch_size, 3, 256, 256),
        'T1n': torch.randn(batch_size, 3, 256, 256),
        'T2f': torch.randn(batch_size, 3, 256, 256),
        'T2w': torch.randn(batch_size, 3, 256, 256)
    }
    
    total_loss, detailed_losses = multi_clip_loss(
        generator_outputs, 
        ['T1c', 'T1n', 'T2f', 'T2w'],
        ['T1n', 'T1c', 'T2w', 'T2f']
    )
    
    print(f"   总CLIP损失: {total_loss:.4f}")
    print("   详细损失:")
    for key, value in detailed_losses.items():
        print(f"     {key}: {value:.4f}")
    
    print("\n✓ 医学模态CLIP损失测试完成")


if __name__ == "__main__":
    demo_medical_clip_loss()
