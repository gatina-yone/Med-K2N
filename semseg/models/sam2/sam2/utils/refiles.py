#!/usr/bin/env python3
"""
随机采样数据重组脚本
从 BraTS 数据集中随机选择10%的数据，重组为标准的训练/验证结构
"""

import os
import shutil
import glob
import random
import re
from pathlib import Path
from collections import defaultdict

# 配置参数
SAMPLE_RATIO = 0.1  # 采样比例，可以修改这个值
RANDOM_SEED = 42    # 随机种子，确保结果可重现

def set_random_seed(seed=RANDOM_SEED):
    """设置随机种子"""
    random.seed(seed)
    print(f"设置随机种子: {seed}")

def extract_file_info(filename):
    """
    从文件名中提取病人ID和切片编号
    例如: BraTS-MET-00002-000_t1n_slice_047.npy
    返回: (patient_id, slice_num, modality)
    """
    pattern = r'(BraTS-MET-\d+-\d+)_(\w+)_slice_(\d+)\.npy'
    match = re.match(pattern, filename)
    
    if match:
        patient_id = match.group(1)
        modality = match.group(2)
        slice_num = match.group(3)
        return patient_id, slice_num, modality
    else:
        return None, None, None

def get_available_samples(source_dir, reference_modality='t1n'):
    """
    获取可用的样本列表 (patient_id, slice_num)
    使用参考模态来确定可用的样本
    """
    reference_dir = os.path.join(source_dir, reference_modality)
    samples = {'train': set(), 'test': set()}
    
    for split in ['train', 'test']:
        split_dir = os.path.join(reference_dir, split)
        if os.path.exists(split_dir):
            npy_files = glob.glob(os.path.join(split_dir, "*.npy"))
            
            for file_path in npy_files:
                filename = os.path.basename(file_path)
                patient_id, slice_num, modality = extract_file_info(filename)
                
                if patient_id and slice_num:
                    samples[split].add((patient_id, slice_num))
    
    return samples

def verify_data_consistency(source_dir, modality_map):
    """
    验证所有模态的数据是否一致
    确保每个模态都有相同的病人ID和切片编号
    """
    print("\n验证源数据一致性...")
    
    all_samples = {}
    
    # 收集每个模态的样本
    for source_modality in modality_map.keys():
        modality_samples = {'train': set(), 'test': set()}
        
        for split in ['train', 'test']:
            split_dir = os.path.join(source_dir, source_modality, split)
            if os.path.exists(split_dir):
                npy_files = glob.glob(os.path.join(split_dir, "*.npy"))
                
                for file_path in npy_files:
                    filename = os.path.basename(file_path)
                    patient_id, slice_num, modality = extract_file_info(filename)
                    
                    if patient_id and slice_num:
                        modality_samples[split].add((patient_id, slice_num))
        
        all_samples[source_modality] = modality_samples
    
    # 检查一致性
    consistency_issues = []
    reference_modality = list(modality_map.keys())[0]
    reference_samples = all_samples[reference_modality]
    
    for split in ['train', 'test']:
        print(f"\n{split} 数据一致性检查:")
        reference_set = reference_samples[split]
        print(f"  参考模态 {reference_modality}: {len(reference_set)} 个样本")
        
        for modality, samples in all_samples.items():
            current_set = samples[split]
            print(f"  {modality}: {len(current_set)} 个样本", end="")
            
            if current_set == reference_set:
                print(" ✓")
            else:
                print(" ✗")
                missing = reference_set - current_set
                extra = current_set - reference_set
                
                if missing:
                    print(f"    缺失样本: {len(missing)} 个")
                    consistency_issues.append(f"{split}/{modality}: 缺失 {len(missing)} 个样本")
                    
                if extra:
                    print(f"    多余样本: {len(extra)} 个")
                    consistency_issues.append(f"{split}/{modality}: 多余 {len(extra)} 个样本")
    
    if consistency_issues:
        print(f"\n⚠️ 发现数据一致性问题:")
        for issue in consistency_issues:
            print(f"  - {issue}")
        
        user_input = input("\n是否继续执行? (y/N): ").strip().lower()
        if user_input != 'y':
            print("用户选择中止执行")
            return False
    else:
        print(f"\n✅ 所有模态的数据完全一致！")
    
    return True

def sample_data(available_samples, sample_ratio=SAMPLE_RATIO):
    """随机采样数据"""
    sampled = {}
    
    for split, samples in available_samples.items():
        samples_list = list(samples)
        sample_size = max(1, int(len(samples_list) * sample_ratio))
        
        # 随机采样
        sampled_list = random.sample(samples_list, sample_size)
        sampled[split] = set(sampled_list)
        
        print(f"{split}: 从 {len(samples_list)} 个样本中选择了 {len(sampled_list)} 个 ({sample_ratio*100:.1f}%)")
    
    return sampled

def create_directory_structure(target_dir):
    """创建目标目录结构"""
    modalities = ['T1n', 'T1c', 'T2w', 'T2f']
    splits = ['train', 'val']
    
    print("创建目标目录结构...")
    for split in splits:
        for modality in modalities:
            dir_path = Path(target_dir) / split / modality
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return True

def move_selected_files(source_dir, target_dir, sampled_data, modality_map):
    """移动选中的文件"""
    moved_stats = defaultdict(lambda: defaultdict(int))
    total_moved = 0
    missing_files = []
    
    for source_modality, target_modality in modality_map.items():
        print(f"\n处理 {source_modality} -> {target_modality}...")
        
        # 处理训练数据和测试数据
        split_map = {'train': 'train', 'test': 'val'}
        
        for source_split, target_split in split_map.items():
            if source_split not in sampled_data:
                continue
                
            source_split_dir = os.path.join(source_dir, source_modality, source_split)
            target_split_dir = os.path.join(target_dir, target_split, target_modality)
            
            if not os.path.exists(source_split_dir):
                print(f"    警告: 源目录不存在: {source_split_dir}")
                continue
            
            # 获取所有文件
            all_files = glob.glob(os.path.join(source_split_dir, "*.npy"))
            moved_count = 0
            
            # 检查选中样本的文件是否都存在
            for patient_id, slice_num in sampled_data[source_split]:
                expected_filename = f"{patient_id}_{source_modality}_slice_{slice_num}.npy"
                expected_path = os.path.join(source_split_dir, expected_filename)
                
                if os.path.exists(expected_path):
                    try:
                        target_path = os.path.join(target_split_dir, expected_filename)
                        shutil.move(expected_path, target_path)
                        moved_count += 1
                        total_moved += 1
                    except Exception as e:
                        print(f"    错误: 移动文件失败 {expected_filename}: {e}")
                else:
                    missing_files.append(f"{source_split}/{source_modality}: {expected_filename}")
            
            moved_stats[target_split][target_modality] = moved_count
            
            if moved_count > 0:
                print(f"  {source_split} -> {target_split}: 移动了 {moved_count} 个文件")
            else:
                print(f"  {source_split} -> {target_split}: 没有文件被移动")
    
    # 报告缺失的文件
    if missing_files:
        print(f"\n⚠️ 发现缺失的文件:")
        for missing in missing_files[:10]:  # 只显示前10个
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... 还有 {len(missing_files) - 10} 个文件缺失")
    
    return moved_stats, total_moved

def copy_config_files(source_root, target_root):
    """复制配置文件"""
    config_files = [
        'dataset_info.txt',
        'file_mapping.csv', 
        'file_mapping.json',
        'slice_range_stats.json'
    ]
    
    print("\n复制配置文件...")
    copied_files = []
    
    for file_name in config_files:
        source_path = Path(source_root) / file_name
        target_path = Path(target_root) / file_name
        
        if source_path.exists():
            try:
                shutil.copy2(source_path, target_path)
                copied_files.append(file_name)
                print(f"  已复制: {file_name}")
            except Exception as e:
                print(f"  错误: 复制配置文件失败 {file_name}: {e}")
        else:
            print(f"  跳过: {file_name} (不存在)")
    
    return copied_files

def display_tree_structure(root_dir):
    """显示目录树结构"""
    print(f"\n新的文件结构:")
    print(f"{root_dir}/")
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print("  目录不存在")
        return
    
    # 简单的树状显示
    for split in ['train', 'val']:
        split_path = root_path / split
        if split_path.exists():
            print(f"├── {split}/")
            modalities = ['T1n', 'T1c', 'T2w', 'T2f']
            for i, modality in enumerate(modalities):
                modality_path = split_path / modality
                if modality_path.exists():
                    prefix = "└──" if i == len(modalities) - 1 else "├──"
                    file_count = len(list(modality_path.glob("*.npy")))
                    print(f"│   {prefix} {modality}/ ({file_count} files)")
    
    # 显示根目录下的配置文件
    config_files = list(root_path.glob("*.txt")) + list(root_path.glob("*.csv")) + list(root_path.glob("*.json"))
    if config_files:
        print("├── 配置文件:")
        for file_path in config_files:
            print(f"│   ├── {file_path.name}")

def count_files_by_category(target_dir):
    """统计各类别的文件数量"""
    print(f"\n文件统计:")
    
    modalities = ['T1n', 'T1c', 'T2w', 'T2f']
    splits = ['train', 'val']
    
    total_files = 0
    
    for split in splits:
        print(f"{split}:")
        split_total = 0
        
        for modality in modalities:
            modality_path = Path(target_dir) / split / modality
            if modality_path.exists():
                file_count = len(list(modality_path.glob("*.npy")))
                print(f"  {modality}: {file_count} 个文件")
                split_total += file_count
            else:
                print(f"  {modality}: 0 个文件 (目录不存在)")
        
        print(f"  {split} 总计: {split_total} 个文件")
        total_files += split_total
    
    print(f"\n总计: {total_files} 个数据文件")

def generate_sampling_report(sampled_data, target_dir):
    """生成采样报告"""
    print(f"\n采样报告:")
    
    for split, samples in sampled_data.items():
        print(f"\n{split.upper()} 数据集:")
        
        # 按病人ID分组
        patients = defaultdict(list)
        for patient_id, slice_num in samples:
            patients[patient_id].append(slice_num)
        
        for patient_id, slices in sorted(patients.items()):
            slices.sort()
            slice_range = f"{slices[0]}-{slices[-1]}" if len(slices) > 1 else slices[0]
            print(f"  {patient_id}: {len(slices)} 个切片 (slice_{slice_range})")
    
    # 保存采样信息到文件
    report_file = os.path.join(target_dir, "sampling_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"数据采样报告\n")
        f.write(f"采样比例: {SAMPLE_RATIO*100:.1f}%\n")
        f.write(f"随机种子: {RANDOM_SEED}\n")
        f.write(f"采样时间: {os.popen('date').read().strip()}\n\n")
        
        for split, samples in sampled_data.items():
            f.write(f"{split.upper()} 数据集 ({len(samples)} 个样本):\n")
            
            patients = defaultdict(list)
            for patient_id, slice_num in samples:
                patients[patient_id].append(slice_num)
            
            for patient_id, slices in sorted(patients.items()):
                slices.sort()
                f.write(f"  {patient_id}: {slices}\n")
            f.write("\n")
    
    print(f"\n采样报告已保存到: {report_file}")

def verify_sample_consistency(target_dir):
    """验证采样数据的一致性"""
    print(f"\n验证数据一致性...")
    
    modalities = ['T1n', 'T1c', 'T2w', 'T2f']
    splits = ['train', 'val']
    
    for split in splits:
        print(f"\n{split} 数据:")
        
        # 收集每个模态的样本
        modality_samples = {}
        for modality in modalities:
            modality_path = Path(target_dir) / split / modality
            if modality_path.exists():
                files = list(modality_path.glob("*.npy"))
                samples = set()
                for file_path in files:
                    filename = file_path.name
                    patient_id, slice_num, _ = extract_file_info(filename)
                    if patient_id and slice_num:
                        samples.add((patient_id, slice_num))
                modality_samples[modality] = samples
        
        # 检查一致性
        if modality_samples:
            reference_samples = list(modality_samples.values())[0]
            all_consistent = True
            
            for modality, samples in modality_samples.items():
                if samples != reference_samples:
                    print(f"  警告: {modality} 的样本与其他模态不一致")
                    print(f"    {modality} 有 {len(samples)} 个样本")
                    print(f"    参考有 {len(reference_samples)} 个样本")
                    all_consistent = False
                else:
                    print(f"  {modality}: {len(samples)} 个样本 ✓")
            
            if all_consistent:
                print(f"  所有模态的样本完全一致 ✓")
        else:
            print(f"  没有找到数据")

def main():
    """主函数"""
    # 配置路径
    SOURCE_DIR = "/data1/tempf/data/MET_60_consistent-256-normalized"
    TARGET_DIR = "/data1/tempf/data/memorysam/medical_mri_sampled"
    
    # 检查源目录是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"错误: 源目录 {SOURCE_DIR} 不存在")
        return 1
    
    print(f"开始随机采样数据重组...")
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    print(f"采样比例: {SAMPLE_RATIO*100:.1f}%")
    print("-" * 50)
    
    # 设置随机种子
    set_random_seed()
    
    # 模态映射 (源目录名 -> 目标目录名)
    modality_map = {
        't1n': 'T1n',
        't1c': 'T1c', 
        't2w': 'T2w',
        't2f': 'T2f'
    }
    
    # 验证源数据一致性
    if not verify_data_consistency(SOURCE_DIR, modality_map):
        return 1
    
    # 获取可用样本
    print("\n扫描可用样本...")
    available_samples = get_available_samples(SOURCE_DIR)
    
    for split, samples in available_samples.items():
        print(f"{split}: 找到 {len(samples)} 个样本")
    
    if not any(available_samples.values()):
        print("错误: 没有找到任何样本")
        return 1
    
    # 随机采样
    print(f"\n随机采样 {SAMPLE_RATIO*100:.1f}% 的数据...")
    sampled_data = sample_data(available_samples, SAMPLE_RATIO)
    
    # 创建目标目录结构
    create_directory_structure(TARGET_DIR)
    
    # 移动选中的文件
    print(f"\n开始移动文件...")
    moved_stats, total_moved = move_selected_files(SOURCE_DIR, TARGET_DIR, sampled_data, modality_map)
    
    # 复制配置文件
    print(f"\n" + "-" * 30)
    copied_configs = copy_config_files(SOURCE_DIR, TARGET_DIR)
    
    # 显示结果
    print(f"\n" + "=" * 50)
    print("随机采样数据重组完成！")
    print(f"总共移动了 {total_moved} 个数据文件")
    print(f"复制了 {len(copied_configs)} 个配置文件")
    
    # 显示新的文件结构
    display_tree_structure(TARGET_DIR)
    
    # 统计文件数量
    count_files_by_category(TARGET_DIR)
    
    # 验证数据一致性
    verify_sample_consistency(TARGET_DIR)
    
    # 生成采样报告
    generate_sampling_report(sampled_data, TARGET_DIR)
    
    return 0

if __name__ == "__main__":
    exit(main())