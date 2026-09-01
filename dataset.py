"""
数据集模块：将 SMILES 转换为分子特征矩阵，构建 DataLoader
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from config import *


def node_features(mol, max_length=MAX_SIZE, atom_types=ATOM_LEN):
    """
    生成节点特征矩阵 - 只包含原子类型，不包含环信息
    """
    features = np.zeros([max_length, atom_types], dtype=np.float32)
    
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_length:
            break
        atomic_num = atom.GetAtomicNum()
        if atomic_num in atom_encoder_m:
            features[i, atom_encoder_m[atomic_num]] = 1.0
    
    # 填充虚拟原子
    actual_atoms = min(mol.GetNumAtoms(), max_length)
    features[actual_atoms:, 0] = 1.0
    
    return torch.tensor(features, dtype=torch.float32)


def bond_features(mol, max_length=MAX_SIZE):
    """生成边特征矩阵"""
    bond_types = BOND_LEN
    A = torch.zeros(bond_types, dtype=torch.int32)
    A[0] = 1
    A = A.expand(max_length, max_length, bond_types).clone()
    
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        i = min(begin, end)
        j = max(begin, end)
        if i >= max_length or j >= max_length:
            continue
        bond_code = bond_encoder_m[bond.GetBondType()]
        bond_onehot = torch.zeros(bond_types, dtype=torch.int32)
        bond_onehot[bond_code] = 1
        A[i, j] = bond_onehot
    
    A = A.reshape(max_length, -1)
    return A


def mol_to_graph(mol, length, smiles=''):
    try:
        # 原子数独热
        mol_length = min(mol.GetNumAtoms(), MAX_SIZE) - 1
        vec_length = torch.zeros([MAX_SIZE, 1], dtype=torch.float)
        vec_length[mol_length, 0] = 1.0
        
        # 节点特征 - 只有原子类型 (ATOM_LEN)
        atom_feat = node_features(mol, MAX_SIZE)
        
        # 边特征
        bond_feat = bond_features(mol, MAX_SIZE)
        
        # 拼接: [长度(1)] + [原子类型(ATOM_LEN)] + [键信息(MAX_SIZE*BOND_LEN)]
        graph = torch.cat([vec_length, atom_feat, bond_feat], 1).float()
        return graph
    except Exception as e:
        print(f'Error processing {smiles}: {e}')
        return None


# ==================== 读取数据 ====================
print("Loading data...")
df = pd.read_csv(DATA_FILE, names=["SMILES", "Length"])

df["mol"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
df = df[df["mol"].notna()].reset_index(drop=True)

df["num_atoms"] = df["mol"].apply(lambda x: x.GetNumAtoms())
df = df[df["num_atoms"] <= MAX_SIZE].reset_index(drop=True)

print(f"筛选后剩余 {len(df)} 个分子 (原子数 <= {MAX_SIZE})")

df['graph'] = df[['mol', 'Length', 'SMILES']].apply(
    lambda row: mol_to_graph(mol=row['mol'], length=row['Length'], smiles=row['SMILES']),
    axis=1
)
df = df[df['graph'].notna()].reset_index(drop=True)
data_list = df['graph'].to_list()

print(f"成功加载 {len(data_list)} 个分子")


# ==================== Dataset 类 ====================
class MolGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]


# ==================== 划分数据集 ====================
mol_dataset = MolGraphDataset(data_list)
train_size = int(len(mol_dataset) * (1 - DIVIDE_RATIO))
test_size = len(mol_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    mol_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# ==================== DataLoader ====================
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=4,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    num_workers=4,
)

print(f"训练集: {len(train_dataset)} 个样本, {len(train_loader)} 个批次")
print(f"测试集: {len(test_dataset)} 个样本, {len(test_loader)} 个批次")