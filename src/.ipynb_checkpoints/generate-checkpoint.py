# generate.py
# 功能：使用新格式解码生成分子

import os
import sys
import argparse
import torch
import pandas as pd
from rdkit import Chem
from config import *
from train import VAE # 确保你的 train.py 里定义了 VAE 类

# --- 解析命令行参数 ---
parser = argparse.ArgumentParser(description='批量生成分子 (适配新格式)')
parser.add_argument('--num', type=int, default=1000, help='要生成的分子数量')
parser.add_argument('--output', type=str, default=None, help='输出文件名（不含路径）')
args = parser.parse_args()

gen_num = args.num
output_name = args.output if args.output else f'generated_{gen_num}_new.csv'

# --- 1. 加载模型 ---
print(f"正在加载模型: {PT_PATH}")
model = VAE().to(DEVICE)
# 注意：如果 train.py 保存的是整个模型而不是 state_dict，这里可能需要调整
try:
    model.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
except:
    # 兼容直接保存 model 的情况
    checkpoint = torch.load(PT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint # 假设保存的就是整个模型对象
model.eval()
print("模型加载完成。")

# --- 2. 生成分子 ---
print(f"开始生成 {gen_num} 个分子...")
records = []

with torch.no_grad():
    # 从正态分布采样 Z
    z = torch.randn(gen_num, Z_DIM).to(DEVICE)
    # 解码得到重构的向量 (Batch, F_COL)
    sample = model.decoder(z) 
    
    for i in range(gen_num):
        # 取出第 i 个分子的向量，转为 CPU numpy 方便处理
        vec = sample[i].cpu().numpy()
        
        record = {
            'index': i,
            'raw_smiles': '',
            'valid': False,
            'reason': ''
        }
        
        try:
            # ================= 解析新格式 =================
            # 1. 获取原子数量 (第0位)
            # 注意：这里存的是 index (num-1)，所以要 +1
            atom_num_float = vec[0]
            atom_num = int(atom_num_float) + 1
            
            # 安全检查：防止解码出负数或超大的原子数
            if atom_num <= 0 or atom_num > MAX_SIZE:
                record['reason'] = f'Invalid atom num: {atom_num}'
                records.append(record)
                continue

            # 2. 解析原子特征
            # 结构: [长度(1)] + [原子1(ATOM_LEN+2)] + [原子2...]
            # 每个原子块的大小
            atom_block_size = ATOM_LEN + 2 
            
            mol = Chem.RWMol()
            
            for j in range(atom_num):
                # 计算当前原子特征在向量中的起始位置
                start_idx = 1 + j * atom_block_size
                end_idx = start_idx + atom_block_size
                
                atom_features = vec[start_idx:end_idx]
                
                # A. 解析原子类型 (One-hot 部分)
                # 取前 ATOM_LEN 个值中最大的索引
                type_probs = atom_features[:ATOM_LEN]
                atom_type_idx = type_probs.argmax()
                
                # 获取原子序数
                atomic_num = atom_decoder_m.get(atom_type_idx, 6) # 默认是碳(6)
                mol.AddAtom(Chem.Atom(atomic_num))
                
                # B. 解析结构特征 (是否在环, 是否芳香)
                # 注意：在全连接网络生成中，这些特征可能不精确（比如0.4），我们可以选择忽略
                # 或者仅作为参考。这里我们主要依赖 RDKit 的 Sanitize 来自动推断芳香性。
                # is_in_ring = atom_features[ATOM_LEN] > 0.5
                # is_aromatic = atom_features[ATOM_LEN + 1] > 0.5
                
            # ================= 构建分子 =================
            # 关键策略：
            # 由于我们移除了键矩阵，我们无法手动 AddBond。
            # 我们让 RDKit 根据原子的价态自动推断键 (SanitizeMol 会做这件事)。
            # 或者，我们可以尝试把 RWMol 转成 Mol 再转回 SMILES，让 RDKit 猜连接方式。
            
            # 1. 先尝试让 RDKit 自动补全键和芳香性
            final_mol = mol.GetMol()
            
            # 2. 严格校验 (这一步会自动加氢、推断键级、处理芳香性)
            Chem.SanitizeMol(final_mol)
            
            # 3. 获取 SMILES
            smiles = Chem.MolToSmiles(final_mol)
            record['raw_smiles'] = smiles
            record['valid'] = True
            
        except Exception as e:
            record['reason'] = str(e)
        
        records.append(record)

# --- 3. 统计结果 ---
df = pd.DataFrame(records)
valid_count = df['valid'].sum()
invalid_count = len(df) - valid_count

print(f"\n{'='*50}")
print(f"生成完成！")
print(f" 总数: {gen_num}")
print(f" 通过RDKit校验: {valid_count} ({valid_count/gen_num*100:.1f}%)")
print(f" 未通过校验: {invalid_count} ({invalid_count/gen_num*100:.1f}%)")
print(f"{'='*50}")

# 打印前5个失败原因（如果有的话）
if invalid_count > 0:
    failed = df[~df['valid']]
    print(f"\n前5个失败原因:")
    for _, row in failed.head(5).iterrows():
        print(f" 分子 {row['index']}: {row['reason']}")

# --- 4. 保存到CSV ---
output_path = os.path.join(DATA_DISK_CKPT_DIR, output_name)
df.to_csv(output_path, index=False)
print(f"\n结果已保存至: {output_path}")