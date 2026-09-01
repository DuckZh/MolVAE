"""
配置文件：定义所有超参数、路径、原子类型等
"""

import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdchem

# ==================== 路径配置 ====================
result_path = Path('./result')
result_path.mkdir(exist_ok=True, parents=True)

DATA_FILE = "/root/autodl-tmp/MolVAE_ckpt/moses_train_cleaned.csv"
PT_PATH = "./result/model.pth"
LOSS_PATH = "./result/losses.csv"
RESULT_PATH = "./result/result.csv"

# ==================== 训练参数 ====================
DIVIDE_RATIO = 0.1
GEN_NUM = 1000
BATCH_SIZE = 1024
EPOCH = 600
LR = 0.0003
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 原子类型 ====================
ATOM_LIST = ('C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Sn')
ATOM_LIST = [0] + [Chem.Atom(x).GetAtomicNum() for x in ATOM_LIST]
ATOM_LEN = len(ATOM_LIST)  # 14

# ==================== 化学键类型 ====================
BOND_LIST = (
    rdchem.BondType.ZERO,
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC
)
BOND_LEN = len(BOND_LIST)  # 5

# ==================== 编码/解码字典 ====================
atom_encoder_m = {l: i for i, l in enumerate(ATOM_LIST)}
atom_decoder_m = {i: l for i, l in enumerate(ATOM_LIST)}
bond_encoder_m = {l: i for i, l in enumerate(BOND_LIST)}
bond_decoder_m = {i: l for i, l in enumerate(BOND_LIST)}

# ==================== 模型维度 ====================
MAX_SIZE = 60

# 【关键修改】每个原子的特征 = 原子类型（去掉环信息）
ATOM_FEAT_LEN = ATOM_LEN  # 14，不再 +1

# 每行的特征数 = 1(长度) + ATOM_FEAT_LEN + MAX_SIZE * BOND_LEN
F_COL = 1 + ATOM_FEAT_LEN + MAX_SIZE * BOND_LEN
# F_COL = 1 + 14 + 60 * 5 = 315

IN_DIM = MAX_SIZE * F_COL

Z_DIM = 256
H1_DIM = 1024
H2_DIM = 512

# 键损失权重（暂时不加权）
BOND_LOSS_WEIGHT = 1.0

# 打印维度信息
print(f"\n{'='*60}")
print(f"维度配置信息:")
print(f"  ATOM_LEN: {ATOM_LEN}")
print(f"  BOND_LEN: {BOND_LEN}")
print(f"  ATOM_FEAT_LEN: {ATOM_FEAT_LEN}")
print(f"  MAX_SIZE: {MAX_SIZE}")
print(f"  F_COL: {F_COL}")
print(f"  IN_DIM: {IN_DIM}")
print(f"  BOND_LOSS_WEIGHT: {BOND_LOSS_WEIGHT}")
print(f"{'='*60}\n")