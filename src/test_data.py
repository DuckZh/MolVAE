"""
测试数据加载
"""

import pandas as pd
from config import DATA_FILE

print("=" * 50)
print("测试数据加载")
print("=" * 50)

# 1. 直接读取原始文件
print("\n1. 直接读取原始文件（前5行）:")
with open(DATA_FILE, 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(f"   {i+1}: {line.strip()}")
        else:
            break

# 2. 用 pandas 读取，带列名
print("\n2. pandas 读取 (names=['SMILES', 'Length']):")
df = pd.read_csv(DATA_FILE, names=["SMILES", "Length"])
print(f"   形状: {df.shape}")
print(f"   列名: {df.columns.tolist()}")
print(f"   Length 列类型: {df['Length'].dtype}")
print(f"   前3行:")
print(df.head(3))

# 3. 筛选 Length <= MAX_SIZE
print("\n3. 筛选 Length <= 16:")
MAX_SIZE = 16
df_filtered = df[df['Length'].astype(int) <= MAX_SIZE]
print(f"   筛选后形状: {df_filtered.shape}")
print(f"   筛选后前3行:")
print(df_filtered.head(3))

# 4. 检查 mol 转换
print("\n4. 测试 mol 转换:")
from rdkit import Chem
sample_smiles = df_filtered['SMILES'].iloc[0] if len(df_filtered) > 0 else None
if sample_smiles:
    mol = Chem.MolFromSmiles(sample_smiles)
    print(f"   SMILES: {sample_smiles}")
    print(f"   mol 对象: {mol is not None}")
    if mol:
        print(f"   原子数: {mol.GetNumAtoms()}")
else:
    print("   ⚠️ 没有符合条件的分子！")
