from rdkit import Chem
import torch

def get_atom_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    for atom in mol.GetAtoms():
        is_aromatic = float(atom.GetIsAromatic())
        is_in_ring = float(atom.IsInRing())
        features.append([is_aromatic, is_in_ring])
    
    return torch.tensor(features, dtype=torch.float32)

# 测试咖啡因
caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
features = get_atom_features(caffeine)

print(f"SMILES: {caffeine}")
print(f"原子数: {features.shape[0]}")
print(f"特征矩阵:\n{features}")

# 详细打印每个原子的信息
mol = Chem.MolFromSmiles(caffeine)
print("\n逐原子分析:")
for i, atom in enumerate(mol.GetAtoms()):
    symbol = atom.GetSymbol()
    is_arom = atom.GetIsAromatic()
    is_ring = atom.IsInRing()
    print(f"  原子 {i}: {symbol} | 芳香: {is_arom} | 在环中: {is_ring}")