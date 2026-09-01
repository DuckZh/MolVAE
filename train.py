"""
训练主程序：VAE 分子生成模型
新增：键损失加权，促进完整分子生成
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem import AllChem
from config import *

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings(action='ignore')


# ==================== 特征矩阵 → 分子转换 ====================
def graph_to_mol(node_labels, adjacency, atom_decoder_m, bond_decoder_m, strict=False):
    """
    从特征矩阵重建分子
    """
    mol = Chem.RWMol()
    
    # 1. 添加原子
    for node_label in node_labels:
        atom_type = atom_decoder_m[node_label]
        if atom_type == 0:
            continue
        atom = Chem.Atom(atom_type)
        mol.AddAtom(atom)
    
    if mol.GetNumAtoms() == 0:
        return None
    
    # 2. 添加键
    for start, end in zip(*np.nonzero(adjacency)):
        if start < end and start < mol.GetNumAtoms() and end < mol.GetNumAtoms():
            bond_type = bond_decoder_m[adjacency[start, end]]
            if bond_type != Chem.rdchem.BondType.ZERO:
                try:
                    mol.AddBond(int(start), int(end), bond_type)
                except:
                    pass
    
    # 3. Sanitize
    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return mol
        except:
            return mol

# ==================== 推理生成分子 ====================
def vae_results(vae, gen_num=GEN_NUM, batch_size=256, save_all=True):
    all_smiles = []
    all_logp = []
    all_mr = []
    all_valid = []
    all_error_types = []
    all_atom_counts = []
    
    vae.eval()
    with torch.no_grad():
        num_batches = (gen_num + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, gen_num - batch_idx * batch_size)
            z = torch.randn(current_batch_size, Z_DIM).to(DEVICE)
            samples = vae.decoder(z)
            
            for idx in range(current_batch_size):
                try:
                    matrix = samples[idx].view(MAX_SIZE, F_COL).cpu()
                    
                    # 1. 解析原子数
                    atom_num_vec = matrix[:, 0]
                    atom_num = torch.argmax(atom_num_vec).item() + 1
                    if atom_num <= 1:
                        all_smiles.append("NO MOL (atom_num <= 1)")
                        all_logp.append(None)
                        all_mr.append(None)
                        all_valid.append(False)
                        all_error_types.append("NO_ATOMS")
                        all_atom_counts.append(atom_num)
                        continue
                    
                    # 2. 解析原子类型 (第1到ATOM_LEN列)
                    atom_type = matrix[:, 1:ATOM_LEN+1]
                    atom_type = torch.max(atom_type, -1)[1]
                    
                    # 3. 【修改】没有环信息了，直接跳过
                    # ring_info 已被移除
                    
                    # 4. 解析键信息 (从 ATOM_LEN+1 开始)
                    bond_start = 1 + ATOM_LEN  # 1(长度) + ATOM_LEN(原子类型)
                    bonds = matrix[:, bond_start:]
                    bonds = bonds.reshape(MAX_SIZE, MAX_SIZE, BOND_LEN)
                    bonds = torch.max(bonds, dim=-1)[1]
                    
                    # 5. 取有效原子部分
                    nodes = atom_type[:atom_num].numpy()
                    edges = bonds[:atom_num, :atom_num].numpy()
                    
                    # 6. 尝试生成分子（不再需要 ring_flags）
                    mol = graph_to_mol(nodes, edges, atom_decoder_m, bond_decoder_m, strict=False)
                    
                    # 7. 验证分子
                    if mol and mol.GetNumAtoms() > 0:
                        try:
                            Chem.SanitizeMol(mol)
                            smiles = Chem.MolToSmiles(mol)
                            if smiles and '.' not in smiles:
                                all_smiles.append(smiles)
                                all_logp.append(MolLogP(mol))
                                all_mr.append(MolMR(mol))
                                all_valid.append(True)
                                all_error_types.append("VALID")
                                all_atom_counts.append(mol.GetNumAtoms())
                            else:
                                all_smiles.append(f"MULTI_FRAG: {smiles}")
                                all_logp.append(None)
                                all_mr.append(None)
                                all_valid.append(False)
                                all_error_types.append("MULTI_FRAG")
                                all_atom_counts.append(mol.GetNumAtoms())
                        except Exception as e:
                            try:
                                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                                smiles = Chem.MolToSmiles(mol)
                                if smiles and '.' not in smiles:
                                    all_smiles.append(smiles)
                                    all_logp.append(MolLogP(mol))
                                    all_mr.append(MolMR(mol))
                                    all_valid.append(True)
                                    all_error_types.append("VALID_KEKULIZE")
                                    all_atom_counts.append(mol.GetNumAtoms())
                                else:
                                    all_smiles.append(f"KEKULIZE_FAIL: {smiles}")
                                    all_logp.append(None)
                                    all_mr.append(None)
                                    all_valid.append(False)
                                    all_error_types.append("KEKULIZE_FAIL")
                                    all_atom_counts.append(mol.GetNumAtoms())
                            except:
                                all_smiles.append(f"ERROR: {str(e)[:30]}")
                                all_logp.append(None)
                                all_mr.append(None)
                                all_valid.append(False)
                                all_error_types.append("SANITIZE_ERROR")
                                all_atom_counts.append(mol.GetNumAtoms())
                    else:
                        all_smiles.append("NO_MOL")
                        all_logp.append(None)
                        all_mr.append(None)
                        all_valid.append(False)
                        all_error_types.append("NO_MOL")
                        all_atom_counts.append(0)
                        
                except Exception as e:
                    all_smiles.append(f"EXCEPTION: {str(e)[:30]}")
                    all_logp.append(None)
                    all_mr.append(None)
                    all_valid.append(False)
                    all_error_types.append("EXCEPTION")
                    all_atom_counts.append(0)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'SMILES': all_smiles,
        'logP': all_logp,
        'MR': all_mr,
        'valid': all_valid,
        'atom_count': all_atom_counts,
        'error_type': all_error_types
    })
    
    valid_count = sum(all_valid)
    print(f"生成 {len(result_df)} 个分子，其中 {valid_count} 个有效 ({valid_count/len(result_df)*100:.1f}%)")
    
    # 显示错误类型分布
    error_counts = result_df['error_type'].value_counts()
    print("\n错误类型分布:")
    for error_type, count in error_counts.items():
        if error_type != "VALID":
            print(f"  {error_type}: {count} ({count/len(result_df)*100:.1f}%)")
    
    if save_all:
        result_path_all = RESULT_PATH.replace('.csv', '_all.csv')
        result_df.to_csv(result_path_all, index=False)
        print(f"完整结果保存至: {result_path_all}")
    
    return result_df[result_df['valid']].copy()

# ==================== 损失函数（键加权版） ====================
def loss_function(recon_x, x, mu, log_var, beta, bond_weight=1.0):
    x_flat = torch.clamp(x.view(-1, IN_DIM), 0.0, 1.0)
    
    # 计算每个位置的BCE损失
    BCE_all = F.binary_cross_entropy(recon_x, x_flat, reduction='none')
    
    # 原子部分和键部分的分界
    bond_start = 1 + ATOM_LEN  # 键信息的起始位置
    
    atom_loss = BCE_all[:, :bond_start].sum()
    bond_loss = BCE_all[:, bond_start:].sum()
    
    BCE = atom_loss + bond_weight * bond_loss
    
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + beta * KLD


# ==================== VAE 模型 ====================
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(IN_DIM, H1_DIM)
        self.fc2 = nn.Linear(H1_DIM, H2_DIM)
        self.fc31 = nn.Linear(H2_DIM, Z_DIM)
        self.fc32 = nn.Linear(H2_DIM, Z_DIM)
        self.fc4 = nn.Linear(Z_DIM, H2_DIM)
        self.fc5 = nn.Linear(H2_DIM, H1_DIM)
        self.fc6 = nn.Linear(H1_DIM, IN_DIM)
        self.dropout = nn.Dropout(0.2)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc31(h), self.fc32(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = self.dropout(h)
        h = F.relu(self.fc5(h))
        h = self.dropout(h)
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, IN_DIM))
        z = self.sampling(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var


# ==================== 训练/测试函数 ====================
def train_epoch(model, loader, optimizer, beta, bond_weight=BOND_LOSS_WEIGHT):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_bce = 0
    total_kld = 0
    
    for graph in loader:
        graph = graph.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, log_var = model(graph)
        loss = loss_function(recon, graph, mu, log_var, beta, bond_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss


def test_epoch(model, loader, beta, bond_weight=BOND_LOSS_WEIGHT):
    """测试一个epoch"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(DEVICE)
            recon, mu, log_var = model(graph)
            loss = loss_function(recon, graph, mu, log_var, beta, bond_weight)
            total_loss += loss.item()
    return total_loss


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"设备: {DEVICE}")
    from dataset import train_loader, test_loader
    
    # 检查特征维度是否正确
    print(f"特征维度检查: IN_DIM={IN_DIM}, F_COL={F_COL}")
    sample_batch = next(iter(train_loader))
    print(f"输入数据形状: {sample_batch.shape}")
    print(f"期望: (batch_size, {MAX_SIZE}, {F_COL})")
    
    model = VAE().to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )
    
    print("\n开始训练...")
    print("=" * 60)
    print(f"键损失权重: 2.0 (原子损失权重: 1.0)")
    print("=" * 60)
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    
    # KL Annealing: 更温和的退火
    for epoch in range(1, EPOCH + 1):
        if epoch <= 100:
            beta = 0.0
        elif epoch <= 300:
            beta = (epoch - 100) / 200.0
        else:
            beta = 1.0
        
        train_loss = train_epoch(model, train_loader, optimizer, beta) / len(train_loader.dataset)
        test_loss = test_epoch(model, test_loader, beta) / len(test_loader.dataset)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), PT_PATH.replace('.pth', '_best.pth'))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCH} | Beta: {beta:.2f} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), PT_PATH)
    print(f"\n模型已保存: {PT_PATH}")
    
    # 保存损失
    pd.DataFrame({
        'Epoch': range(1, EPOCH + 1),
        'Train Loss': train_losses,
        'Test Loss': test_losses
    }).to_csv(LOSS_PATH, index=False)
    print(f"损失已保存: {LOSS_PATH}")
    
    # 生成分子
    print("\n生成分子...")
    model.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
    result_df = vae_results(model, save_all=True)
    result_df.to_csv(RESULT_PATH, index=False)
    print(f"生成 {len(result_df)} 个有效分子，保存至: {RESULT_PATH}")
    print("\n✅ 完成!")