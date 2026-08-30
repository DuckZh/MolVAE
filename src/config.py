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

# 数据文件路径（使用 MOSES 数据集）
DATA_FILE = "../data/moses_train.csv"

# 模型权重保存路径
PT_PATH = "./result/model.pth"

# 损失记录保存路径
LOSS_PATH = "./result/losses.csv"

# 生成分子结果保存路径
RESULT_PATH = "./result/result.csv"

# ==================== 训练参数 ====================
DIVIDE_RATIO = 0.1
GEN_NUM = 1000
BATCH_SIZE = 1024
EPOCH = 500
LR = 0.00001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 原子类型 ====================
ATOM_LIST = ('C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Sn')
ATOM_LIST = [0] + [Chem.Atom(x).GetAtomicNum() for x in ATOM_LIST]
ATOM_LEN = len(ATOM_LIST)

# ==================== 化学键类型 ====================
BOND_LIST = (
    rdchem.BondType.ZERO,
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC
)
BOND_LEN = len(BOND_LIST)

# ==================== 编码/解码字典 ====================
atom_encoder_m = {l: i for i, l in enumerate(ATOM_LIST)}
atom_decoder_m = {i: l for i, l in enumerate(ATOM_LIST)}
bond_encoder_m = {l: i for i, l in enumerate(BOND_LIST)}
bond_decoder_m = {i: l for i, l in enumerate(BOND_LIST)}

# ==================== 模型维度 ====================
MAX_SIZE = 60
F_COL = 1 + ATOM_LEN + MAX_SIZE * BOND_LEN
IN_DIM = MAX_SIZE * F_COL
Z_DIM = 128
H1_DIM = 512
H2_DIM = 256
