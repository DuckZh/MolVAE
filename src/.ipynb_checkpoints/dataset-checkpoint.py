# dataset.py

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from config import *

def mol_to_graph(mol, length, smiles=''):
    """
    将分子转换为新的特征向量。
    新向量结构: [长度信息(1)] + [原子1特征] + [原子2特征] + ...
    每个原子特征 = [原子类型OneHot] + [是否在环] + [是否芳香]
    """
    try:
        # 1. 初始化一个全0向量，长度就是我们新计算的 F_COL
        graph = torch.zeros(F_COL, dtype=torch.float)
        
        # 2. 设置长度信息 (向量的第0位)
        mol_length = mol.GetNumAtoms() - 1
        # 防止越界
        mol_length = min(mol_length, MAX_SIZE - 1)
        graph[0] = float(mol_length)
        
        # 3. 遍历所有原子，填充特征
        for i, atom in enumerate(mol):
            if i >= MAX_SIZE:
                break # 防止分子太大越界
            
            # 计算当前原子特征在向量中的起始位置
            # 1 (长度信息) + i * (每个原子的新特征长度)
            start_idx = 1 + i * (ATOM_LEN + 2)
            
            # --- A. 原子类型 One-Hot ---
            atomic_num = atom.GetAtomicNum()
            if atomic_num in atom_encoder_m:
                atom_type_idx = atom_encoder_m[atomic_num]
                graph[start_idx + atom_type_idx] = 1.0
            
            # --- B. 结构特征 ---
            # 是否在环
            graph[start_idx + ATOM_LEN] = float(atom.IsInRing())
            # 是否芳香
            graph[start_idx + ATOM_LEN + 1] = float(atom.GetIsAromatic())
            
        return graph

    except Exception as e:
        print(f'Error processing {smiles}: {e}')
        return None

# ==================== 读取数据 ====================
print("Loading data...")
df = pd.read_csv(DATA_FILE, names=["SMILES", "Length"], skiprows=1)

# 先转换为 mol 对象
df["mol"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
df = df[df["mol"].notna()].reset_index(drop=True)

# 用原子数筛选
df["num_atoms"] = df["mol"].apply(lambda x: x.GetNumAtoms())
df = df[df["num_atoms"] <= MAX_SIZE].reset_index(drop=True)
print(f"筛选后剩余 {len(df)} 个分子 (原子数 <= {MAX_SIZE})")

# 生成新的特征向量
# 注意：mol_to_graph 函数的参数变了，不再需要 length 和 smiles 来计算键
df['graph'] = df[['mol', 'Length', 'SMILES']].apply(
    lambda row: mol_to_graph(mol=row['mol'], length=row['Length'], smiles=row['SMILES']), axis=1
)
df = df[df['graph'].notna()].reset_index(drop=True)
data_list = df['graph'].to_list()
print(f"成功加载 {len(data_list)} 个分子")

# ==================== Dataset 类 ====================
# (这部分保持不变)
class MolGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# ==================== 划分数据集 ====================
# (这部分保持不变)
mol_dataset = MolGraphDataset(data_list)
train_size = int(len(mol_dataset) * (1 - DIVIDE_RATIO))
test_size = len(mol_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    mol_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
)

# ==================== DataLoader ====================
# (这部分保持不变)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
)
print(f"训练集: {len(train_dataset)} 个样本, {len(train_loader)} 个批次")
print(f"测试集: {len(test_dataset)} 个样本, {len(test_loader)} 个批次")