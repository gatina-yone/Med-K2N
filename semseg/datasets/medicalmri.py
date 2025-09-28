import os
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Union, Optional
import glob
import re
from torch.utils.data import DataLoader
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
import cv2

class MedicalMRI(Dataset):
    r"""
    Custom dataset for Medical MRI generation, supporting multiple modalities as inputs.
    Loads 2D (256, 256) .npy files (grayscale), assumes dataset structure with modality folders.
    No classes/palette since this is for generation, not segmentation.
    Automatically discovers modalities from folder structure and maps them to ABCD naming.
    Supports both one-to-one and many-to-one tasks based on file structure.
    
    Features:
    - Flexible file naming: Supports multiple file naming conventions
    - Dynamic modal discovery: Auto-detects available modalities
    - Robust parsing: Handles various filename formats gracefully
    - ABCD mapping: Maps modalities to A, B, C, D for consistency
    
    Supported file naming formats:
    - BraTS-[A-Z]+-\d+-\d+_{modal}_slice_\d+\.npy (original format)
    - {prefix}_{modal}_slice_{number}.npy
    - {prefix}_{modal}_{number}.npy  
    - {prefix}_{number}.npy
    - Any format with extractable sample ID and slice number
    """
    def __init__(self, root: str = '/data1/tempf/data/memorysam/medical_mri_sampled', split: str = 'train', transform=None, modals=None, modalities=None, target_modal=None):
        super().__init__()
        # 允许空split用于直接指向模态文件夹的情况
        if split and split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train', 'val', or empty string for direct path mode, got: {split}")
        self.root = root
        self.split = split
        self.transform = transform
        
        # 动态发现数据集中的模态文件夹
        self.modal_dirs = self._discover_modalities(split)
        
        # 将模态映射为 ABCD
        self.modal_mapping = self._create_modal_mapping()
        
        # 如果用户指定了模态，使用指定的；否则使用所有发现的模态
        if modalities is not None:
            specified_modals = modalities if isinstance(modalities, list) else [modalities]
        elif modals is not None:
            specified_modals = modals if isinstance(modals, list) else [modals]
        else:
            specified_modals = None
            
        if specified_modals:
            # 验证指定的模态是否存在
            available_modals = list(self.modal_mapping.keys())
            self.used_modals = [m for m in specified_modals if m in available_modals]
            if not self.used_modals:
                raise ValueError(f"None of specified modals {specified_modals} found in available modals {available_modals}")
        else:
            self.used_modals = list(self.modal_mapping.keys())
        
        # 设置目标模态（支持列表）。默认使用最后一个模态作为目标
        if target_modal is None:
            self.target_modals = [self.used_modals[-1]]
        elif isinstance(target_modal, (list, tuple)):
            self.target_modals = [str(m) for m in target_modal]
        else:
            self.target_modals = [str(target_modal)]

        # 输入模态：默认为 used_modals 去除 target_modals 后的集合；
        # 若二者相同（如 ABCD2ABCD），则使用相同集合作为输入，实现自编码/自回归式训练。
        self.input_modals = [m for m in self.used_modals if m not in self.target_modals]
        if len(self.input_modals) == 0:
            # 退化到与目标相同的模态作为输入（保持顺序与 used_modals 一致）
            self.input_modals = list(self.used_modals)
        # 向后兼容的单值属性（仅当目标数量为1时有效）
        self.target_modal = self.target_modals[0] if len(self.target_modals) == 1 else None
        
        self.ignore_label = None
        self.samples = self._get_samples(split)  # list of (sample_id, slice_num)
        
        if not self.samples:
            raise Exception(f"No images found in {self.root}")
        print(f"Found {len(self.samples)} {split} images.")
        print(f"Available modalities: {self.modal_dirs}")
        print(f"Modal mapping: {self.modal_mapping}")
        print(f"Input modals: {self.input_modals}, Target modals: {self.target_modals}")

    def __len__(self) -> int:
        return len(self.samples)
    
    def _discover_modalities(self, split: str) -> list:
        """动态发现数据集中的模态文件夹"""
        # 如果split为空，直接使用root作为模态文件夹的根目录
        if not split:
            split_path = self.root
            print(f"使用直接路径模式: {split_path}")
            
            # 检查是否存在 train/val 等标准划分文件夹
            # 如果用户指定了直接路径但数据仍按 train/val 组织，自动选择 val
            potential_splits = ['val', 'test', 'train']
            found_split_folders = []
            modal_folders = []
            
            for item in os.listdir(split_path):
                item_path = os.path.join(split_path, item)
                if os.path.isdir(item_path):
                    if item in potential_splits:
                        found_split_folders.append(item)
                    else:
                        # 检查是否是模态文件夹
                        item_lower = item.lower()
                        if any(modality in item_lower for modality in ['t1c', 't1n', 't2f', 't2w', 'flair']):
                            modal_folders.append(item)
            
            # 如果发现了划分文件夹但没有模态文件夹，说明数据在划分文件夹内
            if found_split_folders and not modal_folders:
                # 优先选择 val，其次 test，最后 train
                for preferred_split in ['val', 'test', 'train']:
                    if preferred_split in found_split_folders:
                        split_path = os.path.join(split_path, preferred_split)
                        # 只在直接路径推理模式下更新split，不影响训练
                        self.infer_split = preferred_split
                        print(f"自动选择数据划分: {preferred_split}")
                        print(f"更新数据路径为: {split_path}")
                        break
            else:
                # 直接找到了模态文件夹
                self.infer_split = split
        else:
            split_path = os.path.join(self.root, split)
            self.infer_split = split
            
        if not os.path.exists(split_path):
            raise ValueError(f"Data path {split_path} does not exist")
        
        # 获取所有子文件夹作为模态
        modal_dirs = []
        for item in os.listdir(split_path):
            item_path = os.path.join(split_path, item)
            if os.path.isdir(item_path):
                # 检查是否是常见的医学模态文件夹名
                item_lower = item.lower()
                if any(modality in item_lower for modality in ['t1c', 't1n', 't2f', 't2w', 'flair']):
                    modal_dirs.append(item)
        
        # 如果没有找到明确的医学模态，则使用所有子文件夹
        if not modal_dirs:
            for item in os.listdir(split_path):
                item_path = os.path.join(split_path, item)
                if os.path.isdir(item_path):
                    modal_dirs.append(item)
        
        modal_dirs = sorted(modal_dirs)  # 保证顺序一致
        if not modal_dirs:
            raise ValueError(f"No modality folders found in {split_path}")
        
        print(f"发现模态文件夹: {modal_dirs}")
        return modal_dirs
    
    def _create_modal_mapping(self) -> dict:
        """将发现的模态映射为 ABCD 标识符"""
        modal_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # 支持最多8个模态
        if len(self.modal_dirs) > len(modal_letters):
            raise ValueError(f"Too many modalities ({len(self.modal_dirs)}), maximum supported is {len(modal_letters)}")
        
        return {modal_letters[i]: self.modal_dirs[i] for i in range(len(self.modal_dirs))}
        
    def __getitem__(self, index: int) -> Tuple[List[Tensor], Union[Tensor, List[Tensor]]]:
        sample_id, slice_num = self.samples[index]
        
        # Load input modalities as list
        inputs = []
        for modal in self.input_modals:
            modal_dir = self.modal_mapping[modal]
            
            # 尝试找到匹配的文件
            filename = self._find_matching_file(sample_id, slice_num, modal_dir)
            if filename:
                # 智能路径构建：优先使用推理模式的路径，回退到标准路径
                if hasattr(self, 'infer_split') and self.infer_split != self.split:
                    # 推理模式且自动选择了不同的split
                    if self.infer_split:
                        modal_path = os.path.join(self.root, self.infer_split, modal_dir, filename)
                    else:
                        modal_path = os.path.join(self.root, modal_dir, filename)
                else:
                    # 标准模式
                    if self.split:
                        modal_path = os.path.join(self.root, self.split, modal_dir, filename)
                    else:
                        modal_path = os.path.join(self.root, modal_dir, filename)
            else:
                # 回退到构造的文件名
                filename = self._construct_filename(sample_id, modal, slice_num)
                if hasattr(self, 'infer_split') and self.infer_split != self.split:
                    # 推理模式且自动选择了不同的split
                    if self.infer_split:
                        modal_path = os.path.join(self.root, self.infer_split, modal_dir, filename)
                    else:
                        modal_path = os.path.join(self.root, modal_dir, filename)
                else:
                    # 标准模式
                    if self.split:
                        modal_path = os.path.join(self.root, self.split, modal_dir, filename)
                    else:
                        modal_path = os.path.join(self.root, modal_dir, filename)
            
            inputs.append(self._open_npy(modal_path))
        
        # Load target(s)
        targets = []
        for tgt_modal in self.target_modals:
            target_dir = self.modal_mapping[tgt_modal]
            target_filename = self._find_matching_file(sample_id, slice_num, target_dir)
            if target_filename:
                # 智能路径构建：优先使用推理模式的路径，回退到标准路径
                if hasattr(self, 'infer_split') and self.infer_split != self.split:
                    # 推理模式且自动选择了不同的split
                    if self.infer_split:
                        target_path = os.path.join(self.root, self.infer_split, target_dir, target_filename)
                    else:
                        target_path = os.path.join(self.root, target_dir, target_filename)
                else:
                    # 标准模式
                    if self.split:
                        target_path = os.path.join(self.root, self.split, target_dir, target_filename)
                    else:
                        target_path = os.path.join(self.root, target_dir, target_filename)
            else:
                target_filename = self._construct_filename(sample_id, tgt_modal, slice_num)
                if hasattr(self, 'infer_split') and self.infer_split != self.split:
                    # 推理模式且自动选择了不同的split
                    if self.infer_split:
                        target_path = os.path.join(self.root, self.infer_split, target_dir, target_filename)
                    else:
                        target_path = os.path.join(self.root, target_dir, target_filename)
                else:
                    # 标准模式
                    if self.split:
                        target_path = os.path.join(self.root, self.split, target_dir, target_filename)
                    else:
                        target_path = os.path.join(self.root, target_dir, target_filename)
            targets.append(self._open_npy(target_path))

        # Apply transform if available
        if self.transform:
            # Stack inputs to multi-channel tensor for augmentation
            if len(inputs) > 0:
                img = torch.cat(inputs, dim=0)  # e.g., N modals * 3ch = N*3ch tensor

                # 为数据增强动态调整均值和标准差 - 修复：保持identity normalization
                original_transform = self.transform
                if hasattr(original_transform, 'transforms'):
                    for t in original_transform.transforms:
                        if hasattr(t, 'mean') and hasattr(t, 'std'):
                            # 修复：保持identity normalization参数，避免数据范围被错误变换
                            num_channels = img.shape[0]
                            t.mean = [0.0] * num_channels  # identity: (x - 0) / 1 = x
                            t.std = [1.0] * num_channels
                # 多目标场景：将多个目标在通道维拼接，保证与img使用一致的空间增强
                target_cat = torch.cat(targets, dim=0) if len(targets) > 1 else targets[0]
                transformed = self.transform({'img': img, 'mask': target_cat})
                # Split back to list of 3ch tensors
                channels_per_modal = 3
                inputs = list(torch.split(transformed['img'], channels_per_modal, dim=0))  # 按每个模态3通道拆分
                # 还原目标列表
                if len(self.target_modals) > 1:
                    targets = list(torch.split(transformed['mask'], channels_per_modal, dim=0))
                else:
                    targets = [transformed['mask']]
            else:
                # 无独立输入的情况（如 ABCD2ABCD）：
                # 使用拼接后的目标作为 img 与 mask 一起进行一致的空间增强，随后再拆分回去
                target_cat = torch.cat(targets, dim=0) if len(targets) > 1 else targets[0]
                transformed = self.transform({'img': target_cat, 'mask': target_cat})
                channels_per_modal = 3
                if len(self.target_modals) > 1:
                    targets = list(torch.split(transformed['mask'], channels_per_modal, dim=0))
                else:
                    targets = [transformed['mask']]

        # --- 归一化到[0,1]（dataloader阶段，鲁棒处理每个模态3通道一组） ---
        def _to_01(x: Tensor) -> Tensor:
            if x is None:
                return x
            # 检查数据是否已经在[0,1]范围内（容忍小的数值误差）
            x_min = float(x.min())
            x_max = float(x.max())
            
            # 修复：如果数据已经在[0,1]范围内，直接返回（避免重复归一化）
            if x_min >= -0.01 and x_max <= 1.01 and x_min >= 0.0:
                return torch.clamp(x, 0.0, 1.0)
            
            # 优先识别[-1,1]并线性映射到[0,1]
            if x_min >= -1.01 and x_max <= 1.01 and x_min < -0.01:
                x = (x + 1.0) * 0.5
            else:
                # 否则做min-max归一化（避免除0）
                rng = x_max - x_min
                if rng > 1e-6:
                    x = (x - x_min) / rng
            return torch.clamp(x, 0.0, 1.0)

        # 对inputs按模态（每3通道）归一化
        norm_inputs: List[Tensor] = []
        for t in inputs:
            if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[0] % 3 == 0:
                # 以3通道为一组归一化（通常单组即可）
                if t.shape[0] == 3:
                    norm_inputs.append(_to_01(t))
                else:
                    chunks = torch.split(t, 3, dim=0)
                    chunks = [ _to_01(c) for c in chunks ]
                    norm_inputs.append(torch.cat(chunks, dim=0))
            else:
                norm_inputs.append(_to_01(t) if isinstance(t, torch.Tensor) else t)
        inputs = norm_inputs

        # 对targets（单或多目标）做相同处理
        if isinstance(targets, list):
            norm_targets: List[Tensor] = []
            for t in targets:
                if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[0] % 3 == 0:
                    if t.shape[0] == 3:
                        norm_targets.append(_to_01(t))
                    else:
                        chunks = torch.split(t, 3, dim=0)
                        chunks = [ _to_01(c) for c in chunks ]
                        norm_targets.append(torch.cat(chunks, dim=0))
                else:
                    norm_targets.append(_to_01(t) if isinstance(t, torch.Tensor) else t)
            targets = norm_targets
        else:
            t = targets
            if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[0] % 3 == 0:
                if t.shape[0] == 3:
                    targets = _to_01(t)
                else:
                    chunks = torch.split(t, 3, dim=0)
                    chunks = [ _to_01(c) for c in chunks ]
                    targets = torch.cat(chunks, dim=0)
            else:
                targets = _to_01(t) if isinstance(t, torch.Tensor) else t

        # 返回：单目标返回Tensor，多目标返回list
        if len(self.target_modals) == 1:
            return inputs, targets[0]
        else:
            return inputs, targets
    
    def _get_original_modal_name(self, modal_dir: str) -> str:
        """根据文件夹名推断原始模态名（用于文件名）"""
        return modal_dir.lower()  # 简单地转换为小写
    
    def _parse_filename(self, filename: str) -> tuple:
        r"""
        解析文件名以提取样本ID和切片号，支持多种文件命名格式
        增强鲁棒性，不依赖特定的文件命名模式
        
        支持的格式：
        1. BraTS-[A-Z]+-\d+-\d+_{modal}_slice_\d+\.npy  (原格式)
        2. [任意前缀]_{modal}_slice_\d+\.npy
        3. [任意前缀]_{modal}_\d+\.npy
        4. [任意前缀]_\d+\.npy
        5. 通用格式：任意包含数字的文件名
        """
        filename_without_ext = filename.replace('.npy', '')
        
        # 格式1: BraTS-[A-Z]+-\d+-\d+_{modal}_slice_\d+
        match = re.match(r'(BraTS-[A-Z]+-\d+-\d+)_[a-z0-9]+_slice_(\d+)', filename_without_ext)
        if match:
            return match.group(1), match.group(2)
        
        # 格式2: 任意前缀_{modal}_slice_{数字}
        match = re.match(r'(.+)_[a-z0-9]+_slice_(\d+)', filename_without_ext)
        if match:
            return match.group(1), match.group(2)
        
        # 格式3: 任意前缀_{modal}_{数字}
        match = re.match(r'(.+)_[a-z0-9]+_(\d+)', filename_without_ext)
        if match:
            return match.group(1), match.group(2)
        
        # 格式4: 任意前缀_{数字} (精确匹配，避免过度捕获)
        match = re.match(r'(.+?)_(\d+)$', filename_without_ext)
        if match:
            return match.group(1), match.group(2)
        
        # 格式5: 无分隔符，数字结尾 (更精确的匹配)
        match = re.match(r'([a-zA-Z_]+)(\d+)$', filename_without_ext)
        if match:
            return match.group(1), match.group(2)
        
        # 格式6: 更通用的数字匹配
        numbers = re.findall(r'\d+', filename_without_ext)
        if numbers:
            slice_num = numbers[-1]  # 使用最后一个数字作为切片号
            # 去除所有数字，得到基础名称
            base_name = re.sub(r'\d+', '_', filename_without_ext).strip('_')
            if not base_name:
                # 如果没有基础名称，使用文件名减去最后的数字
                base_name = filename_without_ext[:-len(slice_num)].rstrip('_')
            return base_name, slice_num
        
        # 如果所有格式都不匹配，返回None
        print(f"Warning: Unable to parse filename format: {filename}")
        return None, None
    
    def _construct_filename(self, sample_id: str, modal: str, slice_num: str) -> str:
        """
        根据样本ID、模态和切片号构造文件名
        使用智能模式匹配自动适配不同的文件命名格式
        """
        modal_dir = self.modal_mapping[modal]
        original_modal_name = self._get_original_modal_name(modal_dir)
        modal_path = os.path.join(self.root, self.split, modal_dir)
        
        if not os.path.exists(modal_path):
            # 如果路径不存在，返回默认格式
            return f"{sample_id}_{original_modal_name}_slice_{slice_num}.npy"
        
        # 查看目录中实际的文件，学习文件命名模式
        existing_files = [f for f in os.listdir(modal_path) if f.endswith('.npy')]
        
        if existing_files:
            # 分析第一个文件的格式来确定命名模式
            example_file = existing_files[0]
            example_sample, example_slice = self._parse_filename(example_file)
            
            if example_sample and example_slice:
                # 基于示例文件的格式构造新文件名
                template = example_file.replace(example_sample, '{sample}').replace(example_slice, '{slice}')
                # 替换模态名
                for modal_name in ['t1c', 't1n', 't2f', 't2w', 'T1c', 'T1n', 'T2f', 'T2w']:
                    if modal_name in template:
                        template = template.replace(modal_name, '{modal}')
                        break
                
                # 应用模板
                filename = template.format(
                    sample=sample_id,
                    slice=slice_num,
                    modal=original_modal_name
                )
                return filename
        
        # 回退到默认格式
        return f"{sample_id}_{original_modal_name}_slice_{slice_num}.npy"
    
    def _find_matching_file(self, sample_id: str, slice_num: str, modal_dir: str) -> Optional[str]:
        """
        在指定目录中查找匹配的文件
        使用宽松的匹配策略提高鲁棒性
        """
        # 根据split构建路径
        if hasattr(self, 'infer_split') and self.infer_split != self.split:
            # 推理模式且自动选择了不同的split
            if self.infer_split:
                modal_path = os.path.join(self.root, self.infer_split, modal_dir)
            else:
                modal_path = os.path.join(self.root, modal_dir)
        elif self.split:
            modal_path = os.path.join(self.root, self.split, modal_dir)
        else:
            # 直接路径模式
            modal_path = os.path.join(self.root, modal_dir)
            
        if not os.path.exists(modal_path):
            return None
        
        # 获取所有.npy文件
        all_files = [f for f in os.listdir(modal_path) if f.endswith('.npy')]
        
        # 尝试精确匹配
        for filename in all_files:
            parsed_sample, parsed_slice = self._parse_filename(filename)
            if parsed_sample == sample_id and parsed_slice == slice_num:
                return filename
        
        # 尝试宽松匹配（包含样本ID和切片号）
        for filename in all_files:
            if sample_id in filename and slice_num in filename:
                return filename
        
        return None
    def _open_npy(self, file: str) -> Tensor:
        """Load .npy file (256, 256) and convert to 3-channel tensor."""
        img = np.load(file)  # Shape: (256, 256)
        img = img.astype(np.float32)
        # 修复: 启用归一化确保数据在正确的[0,1]范围内 - 这是解决强度问题的关键
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # Convert to 3 channels (C, H, W) for compatibility with augmentations
        img = np.stack([img, img, img], axis=0)  # Shape: (3, 256, 256)
        return torch.from_numpy(img).float()

    def _get_samples(self, split_name: str) -> list:
        # 允许空split_name用于直接路径模式
        if split_name and split_name not in ['train', 'val']:
            raise ValueError(f"Split must be 'train', 'val', or empty string, got: {split_name}")
            
        # 使用第一个输入模态作为参考来发现样本
        if not self.input_modals:
            # 如果没有输入模态，使用第一个目标模态作为参考
            reference_modal = self.target_modal if self.target_modal is not None else (self.target_modals[0] if hasattr(self, 'target_modals') and len(self.target_modals) > 0 else None)
            if reference_modal is None:
                raise ValueError("No input or target modalities available to reference samples")
        else:
            reference_modal = self.input_modals[0]
        
        reference_dir = self.modal_mapping[reference_modal]
        
        # 根据split_name构建路径
        if hasattr(self, 'infer_split') and self.infer_split != split_name:
            # 推理模式且自动选择了不同的split
            if self.infer_split:
                pattern = os.path.join(self.root, self.infer_split, reference_dir, "*.npy")
            else:
                pattern = os.path.join(self.root, reference_dir, "*.npy")
        elif split_name:
            pattern = os.path.join(self.root, split_name, reference_dir, "*.npy")
        else:
            # 直接路径模式
            pattern = os.path.join(self.root, reference_dir, "*.npy")
            
        modal_files = sorted(glob.glob(pattern))
        print(f"在路径 {pattern} 中找到 {len(modal_files)} 个文件")
        
        samples = []
        for file_path in modal_files:
            base_name = os.path.basename(file_path)
            
            # 尝试多种文件命名格式，增强鲁棒性
            sample_id, slice_num = self._parse_filename(base_name)
            if sample_id and slice_num:
                samples.append((sample_id, slice_num))
        
        # Sort samples
        samples = sorted(samples)
        
        # 验证所有需要的模态文件都存在
        valid_samples = []
        for sample_id, slice_num in samples:
            sample_valid = True
            
            # 检查输入模态文件
            for modal in self.input_modals:
                modal_dir = self.modal_mapping[modal]
                filename = self._find_matching_file(sample_id, slice_num, modal_dir)
                if not filename:
                    print(f"Warning: Missing input file for {sample_id}, slice {slice_num}, modal {modal}")
                    sample_valid = False
                    break
            
            # 检查目标模态文件（支持多目标）
            if sample_valid:
                for tgt_modal in (self.target_modals if hasattr(self, 'target_modals') else [self.target_modal]):
                    target_dir = self.modal_mapping[tgt_modal]
                    target_filename = self._find_matching_file(sample_id, slice_num, target_dir)
                    if not target_filename:
                        print(f"Warning: Missing target file for {sample_id}, slice {slice_num}, modal {tgt_modal}")
                        sample_valid = False
                        break
            
            if sample_valid:
                valid_samples.append((sample_id, slice_num))
        
        return valid_samples


# 为了向后兼容，提供别名
MedicalMRIDataset = MedicalMRI





