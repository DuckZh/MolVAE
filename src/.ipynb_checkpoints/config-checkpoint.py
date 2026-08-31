# config.py

import os
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdchem

# ==================== 路径配置 ====================
# (这部分保持不变)
data_disk_dir = Path('/root/autodl-tmp/MolVAE_ckpt')
data_disk_dir.mkdir(exist_ok=True, parents=True)
DATA_FILE = "/root/autodl-tmp/MolVAE_ckpt/moses_train_cleaned.csv"
DATA_DISK_CKPT_DIR = "/root/autodl-tmp/MolVAE_ckpt"
PT_PATH = os.path.join(DATA_DISK_CKPT_DIR, 'model_final.pth')
LOSS_PATH = data_disk_dir / "losses.csv"
RESULT_PATH = data_disk_dir / "result.csv"

# ==================== 训练参数 ====================
# (这部分保持不变)
DIVIDE_RATIO = 0.1
GEN_NUM = 1000
BATCH_SIZE = 1024
EPOCH = 1000
LR = 0.0003
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 原子类型 ====================
# (这部分保持不变)
ATOM_LIST = ('C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Sn')
ATOM_LIST = [0] + [Chem.Atom(x).GetAtomicNum() for x in ATOM_LIST]
ATOM_LEN = len(ATOM_LIST) # 原子特征的长度

# ==================== 化学键类型 ====================
# 【删除】旧的键类型定义
# BOND_LIST = ...
# BOND_LEN = ...

# ==================== 编码/解码字典 ====================
# (这部分保持不变)
atom_encoder_m = {l: i for i, l in enumerate(ATOM_LIST)}
atom_decoder_m = {i: l for i, l in enumerate(ATOM_LIST)}
# 【删除】旧的键字典
# bond_encoder_m = ...
# bond_decoder_m = ...

# ==================== 模型维度 ====================
MAX_SIZE = 60

# 【核心修改】计算新的输入维度
# 1. 长度信息占 1 位
# 2. 每个原子的新特征 = 原子类型(ATOM_LEN) + 是否在环(1) + 是否芳香(1)
NEW_ATOM_FEATURE_LEN = ATOM_LEN + 2
# 3. 总长度 = 1 + (最大原子数 * 每个原子的新特征长度)
# 我们彻底抛弃了旧的键矩阵部分 (MAX_SIZE * MAX_SIZE * BOND_LEN)
F_COL = 1 + (MAX_SIZE * NEW_ATOM_FEATURE_LEN)

IN_DIM = F_COL # 输入维度就是 F_COL

Z_DIM = 256
H1_DIM = 1024
H2_DIM = 512