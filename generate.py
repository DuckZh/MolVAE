"""
测试生成脚本：分析模型生成的分子质量
"""

import torch
import pandas as pd
from train import VAE, vae_results, graph_to_mol
from config import *
from rdkit import Chem

print("=" * 60)
print("分子生成质量分析")
print("=" * 60)

# 1. 加载模型
print("\n加载模型...")
model = VAE().to(DEVICE)
model.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
model.eval()
print(f"✅ 模型加载成功: {PT_PATH}")

# 2. 生成分子（使用修改后的 vae_results）
print("\n生成 1000 个分子并分析...")
print("-" * 60)

# 注意：vae_results 现在会返回有效的分子 DataFrame
# 但完整的统计信息会打印在控制台，完整结果会保存为 _all.csv
result_valid = vae_results(model, gen_num=1000, save_all=True)

# 3. 读取完整结果文件
all_results = pd.read_csv(RESULT_PATH.replace('.csv', '_all.csv'))

# 4. 统计分析
print("\n" + "=" * 60)
print("统计分析")
print("=" * 60)

total = len(all_results)
valid = all_results['valid'].sum()
invalid = total - valid

print(f"\n📊 生成统计:")
print(f"  总生成数: {total}")
print(f"  有效分子: {valid} ({valid/total*100:.1f}%)")
print(f"  无效分子: {invalid} ({invalid/total*100:.1f}%)")

# 5. 分析无效分子的类型
if invalid > 0:
    print(f"\n🔍 无效分子类型分布:")
    invalid_types = all_results[~all_results['valid']]['SMILES'].value_counts()
    for k, v in invalid_types.items():
        print(f"  {k}: {v} ({v/invalid*100:.1f}%)")

# 6. 原子数分布
print(f"\n📏 原子数分布 (前10):")
atom_counts = all_results['atom_count'].value_counts().sort_index()
for atom_num, count in atom_counts.head(10).items():
    print(f"  {atom_num} 个原子: {count} ({count/total*100:.1f}%)")

# 7. 有效分子的属性统计
if valid > 0:
    print(f"\n🧪 有效分子属性:")
    print(f"  logP 范围: [{all_results[all_results['valid']]['logP'].min():.2f}, "
          f"{all_results[all_results['valid']]['logP'].max():.2f}]")
    print(f"  logP 平均: {all_results[all_results['valid']]['logP'].mean():.2f}")
    print(f"  MR 范围: [{all_results[all_results['valid']]['MR'].min():.2f}, "
          f"{all_results[all_results['valid']]['MR'].max():.2f}]")
    print(f"  MR 平均: {all_results[all_results['valid']]['MR'].mean():.2f}")

# 8. 显示有效分子示例
print(f"\n✅ 有效分子示例 (前10个):")
print("-" * 60)
for i, row in all_results[all_results['valid']].head(10).iterrows():
    print(f"  {row['SMILES']}")
    print(f"    logP: {row['logP']:.2f}, MR: {row['MR']:.2f}, 原子数: {int(row['atom_count'])}")

# 9. 显示无效分子示例
if invalid > 0:
    print(f"\n❌ 无效分子示例 (前10个):")
    print("-" * 60)
    for i, row in all_results[~all_results['valid']].head(10).iterrows():
        print(f"  {row['SMILES'][:80]}")
        if pd.notna(row['atom_count']):
            print(f"    原子数: {int(row['atom_count'])}")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)