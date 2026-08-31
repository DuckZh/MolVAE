"""
文件名：clean_data.py
功能：清洗并标准化分子数据，生成干净的训练数据
用法：python clean_data.py
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from config import *

# --- 配置 ---
INPUT_FILE = os.path.join(DATA_DISK_CKPT_DIR, "moses_train.csv")  # 原始数据路径，按你的实际情况改
OUTPUT_FILE = os.path.join(DATA_DISK_CKPT_DIR, "moses_train_cleaned.csv")
MAX_ATOMS = 60  # 最大原子数限制

print(f"读取原始数据: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
total = len(df)
print(f"原始分子数: {total}")

# --- 清洗 ---
cleaned = []
stats = {
    'parse_fail': 0,
    'too_many_atoms': 0,
    'contains_metal': 0,
    'duplicate': 0,
}

seen_smiles = set()

for idx, row in df.iterrows():
    smiles = row['smiles'] if 'smiles' in df.columns else row.iloc[0]

    # 1. RDKit 解析
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        stats['parse_fail'] += 1
        continue

    # 2. 去盐：只保留最大的分子片段
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())

    # 3. 检查是否含金属/非有机原子
    organic_atoms = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P'}
    has_metal = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in organic_atoms:
            has_metal = True
            break
    if has_metal:
        stats['contains_metal'] += 1
        continue

    # 4. 原子数过滤
    if mol.GetNumAtoms() > MAX_ATOMS:
        stats['too_many_atoms'] += 1
        continue

    # 5. 标准化为 Canonical SMILES
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # 6. 去重
    if canonical in seen_smiles:
        stats['duplicate'] += 1
        continue
    seen_smiles.add(canonical)

    cleaned.append(canonical)

    # 每处理1万个打印一次进度
    if (idx + 1) % 10000 == 0:
        print(f"  已处理 {idx + 1}/{total} ...")

# --- 保存 ---
result_df = pd.DataFrame({'smiles': cleaned})
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*50}")
print(f"清洗完成！")
print(f"  原始分子数: {total}")
print(f"  清洗后分子数: {len(cleaned)}")
print(f"  过滤统计:")
print(f"    RDKit解析失败: {stats['parse_fail']}")
print(f"    含金属/非有机原子: {stats['contains_metal']}")
print(f"    原子数超限(>{MAX_ATOMS}): {stats['too_many_atoms']}")
print(f"    重复分子: {stats['duplicate']}")
print(f"  输出文件: {OUTPUT_FILE}")
print(f"{'='*50}")