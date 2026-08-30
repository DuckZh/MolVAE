"""
统计 MOSES 数据集中分子的原子数分布
"""

import pandas as pd
from rdkit import Chem

# 读取数据
df = pd.read_csv('/root/ai-for-chemistry-synthesis/data/raw/train.csv')
smiles_list = df['SMILES'].tolist()

# 统计原子数
atom_counts = []
for smi in smiles_list[:50000]:  # 先统计 5 万个
    mol = Chem.MolFromSmiles(smi)
    if mol:
        atom_counts.append(mol.GetNumAtoms())

print("=" * 60)
print("分子原子数统计 (基于 50,000 个分子)")
print("=" * 60)
print(f"总分子数: {len(atom_counts)}")
print(f"最小原子数: {min(atom_counts)}")
print(f"最大原子数: {max(atom_counts)}")
print(f"平均原子数: {sum(atom_counts)/len(atom_counts):.1f}")

# 分位数
percentiles = [50, 75, 90, 95, 99]
print("\n分位数:")
for p in percentiles:
    val = sorted(atom_counts)[int(len(atom_counts) * p / 100)]
    print(f"  {p}% 的分子原子数 <= {val}")

# 统计各原子数范围的分子数量
ranges = [(0, 16), (17, 24), (25, 32), (33, 40), (41, 50), (51, 100)]
print("\n原子数分布:")
for low, high in ranges:
    count = sum(1 for n in atom_counts if low <= n <= high)
    pct = count / len(atom_counts) * 100
    print(f"  {low:3d} - {high:3d}: {count:6d} 分子 ({pct:5.1f}%)")
