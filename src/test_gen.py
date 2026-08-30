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
        atom_num = matrix[:, 0]
        atom_num = torch.argmax(atom_num).item() + 1
        print(f"分子 {i}: 解析原子数 = {atom_num}")
        
        atom_type = matrix[:, 1:ATOM_LEN+1]
        atom_type = torch.max(atom_type, -1)[1]
        bonds = matrix[:, ATOM_LEN+1:]
        bonds = bonds.reshape(MAX_SIZE, MAX_SIZE, BOND_LEN)
        bonds = torch.max(bonds, dim=-1)[1]
        
        nodes = atom_type[:atom_num]
        edges = bonds[:atom_num, :atom_num]
        
        # 打印一些统计信息
        print(f"  nodes shape: {nodes.shape}, edges shape: {edges.shape}")
        print(f"  node types: {nodes[:5].tolist()}")
        
        # 尝试转换为 mol
        mol = Chem.RWMol()
        for node_label in nodes.numpy():
            mol.AddAtom(Chem.Atom(atom_decoder_m[node_label]))
        for start, end in zip(*np.nonzero(edges.numpy())):
            if start < end:
                try:
                    mol.AddBond(int(start), int(end), bond_decoder_m[edges.numpy()[start, end]])
                except:
                    pass
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            print(f"  ✅ 生成 SMILES: {smi[:50]}...")
        except Exception as e:
            print(f"  ❌ 分子无效: {e}")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
