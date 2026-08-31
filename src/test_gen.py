import torch
import numpy as np
from rdkit import Chem
from config import *
from train import VAE

# 加载模型
model = VAE().to(DEVICE)
model.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
model.eval()

# 生成 10 个样本看看
with torch.no_grad():
    z = torch.randn(10, Z_DIM).to(DEVICE)
    sample = model.decoder(z)
    print(f"样本形状: {sample.shape}")

    for i, matrix in enumerate(sample.view(10, 1, MAX_SIZE, F_COL).cpu()):
        try:
            matrix = matrix[0]
            # 1. 解析原子数量
            atom_num_logits = matrix[:, 0]
            atom_num = torch.argmax(atom_num_logits).item() + 1
            
            # 2. 解析原子类型
            atom_type_logits = matrix[:, 1:ATOM_LEN+1]
            nodes = torch.max(atom_type_logits, -1)[1]
            
            # 3. 解析化学键
            bonds_logits = matrix[:, ATOM_LEN+1:]
            bonds_logits = bonds_logits.reshape(MAX_SIZE, MAX_SIZE, BOND_LEN)
            edges = torch.max(bonds_logits, dim=-1)[1]

            # 截取有效部分
            nodes = nodes[:atom_num]
            edges = edges[:atom_num, :atom_num]

            print(f"分子 {i}: 解析原子数 = {atom_num}")
            print(f"  nodes shape: {nodes.shape}, edges shape: {edges.shape}")
            print(f"  node types: {nodes[:5].tolist()}")

            # --- 构建 RDKit 分子对象 ---
            mol = Chem.RWMol()
            # 添加原子
            for node_label in nodes.numpy():
                # 确保 atom_decoder_m 能正确处理索引
                atom_symbol = atom_decoder_m[node_label]
                mol.AddAtom(Chem.Atom(atom_symbol))
            
            # 添加化学键
            edge_indices = np.nonzero(edges.numpy())
            for start, end in zip(*edge_indices):
                if start < end: # 避免重复添加无向键
                    bond_type_idx = edges.numpy()[start, end]
                    # 确保 bond_decoder_m 能正确处理索引
                    bond_type = bond_decoder_m[bond_type_idx]
                    try:
                        mol.AddBond(int(start), int(end), bond_type)
                    except Exception as e:
                        print(f"    添加键时出错 (可能是价态问题): {e}")

            # --- 关键修改：分子后处理 ---
            try:
                # 第一次尝试标准化
                Chem.SanitizeMol(mol)
            except Exception as e:
                # 如果失败，尝试修复常见的芳香性问题
                # print(f"  初次校验失败: {e}，尝试修复...")
                for atom in mol.GetAtoms():
                    # 如果一个原子被标记为芳香性，但它不在环里，就取消标记
                    if atom.GetIsAromatic() and not atom.IsInRing():
                        atom.SetIsAromatic(False)
                # 修复后，再次尝试标准化
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e2:
                    print(f"  ❌ 修复后校验依然失败: {e2}")
                    continue # 跳过这个分子

            # 如果成功，转换为 SMILES
            smi = Chem.MolToSmiles(mol)
            print(f"  ✅ 生成 SMILES: {smi}")

        except Exception as e:
            print(f"  ❌ 解析或构建过程中出错: {e}")
