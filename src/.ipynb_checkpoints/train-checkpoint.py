"""
文件名：train.py
功能：VAE 分子生成模型训练 (温和KL退火 + 梯度裁剪 + 数据盘路径)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
import os
import datetime

# ==================== 1. 导入配置 ====================
from config import *
from dataset import train_loader, test_loader

# ==================== 2. 设置数据盘路径 ====================
DATA_DISK_CKPT_DIR = "/root/autodl-tmp/MolVAE_ckpt"

if not os.path.exists(DATA_DISK_CKPT_DIR):
    os.makedirs(DATA_DISK_CKPT_DIR)
    print(f"✅ 已在数据盘创建目录: {DATA_DISK_CKPT_DIR}")

# 禁用 RDKit 警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings(action='ignore')

# ==================== 3. 特征矩阵 → 分子转换 ====================
def graph_to_mol(node_labels, adjacency, atom_decoder_m, bond_decoder_m, strict=False):
    mol = Chem.RWMol()
    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(atom_decoder_m[node_label]))
    
    for start, end in zip(*np.nonzero(adjacency)):
        if start < end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adjacency[start, end]])
    
    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None
    return mol

# ==================== 4. 推理生成分子 ====================
def vae_results(vae, gen_num=GEN_NUM):
    with torch.no_grad():
        z = torch.randn(int(gen_num), Z_DIM).to(DEVICE)
        sample = vae.decoder(z)
        smiles_list = []
        logp_list = []
        mr_list = []
        
        for matrix in sample.view(int(gen_num), 1, MAX_SIZE, F_COL).cpu():
            try:
                matrix = matrix[0]
                atom_num = matrix[:, 0]
                atom_num = torch.argmax(atom_num).item() + 1
                atom_type = matrix[:, 1:ATOM_LEN+1]
                atom_type = torch.max(atom_type, -1)[1]
                bonds = matrix[:, ATOM_LEN+1:]
                bonds = bonds.reshape(MAX_SIZE, MAX_SIZE, BOND_LEN)
                bonds = torch.max(bonds, dim=-1)[1]
                
                nodes = atom_type[:atom_num]
                edges = bonds[:atom_num, :atom_num]
                
                mol = graph_to_mol(
                    nodes.numpy(), 
                    edges.numpy(), 
                    atom_decoder_m, 
                    bond_decoder_m, 
                    strict=True
                )
                
                if mol and '.' not in Chem.MolToSmiles(mol):
                    smiles_list.append(Chem.MolToSmiles(mol))
                    logp_list.append(MolLogP(mol))
                    mr_list.append(MolMR(mol))
            except Exception as e:
                continue
        
        min_len = min(len(smiles_list), len(logp_list), len(mr_list))
        return pd.DataFrame({
            'SMILES': smiles_list[:min_len],
            'logP': logp_list[:min_len],
            'MR': mr_list[:min_len]
        })

# ==================== 5. 损失函数 ====================
def loss_function(recon_x, x, mu, log_var, beta):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, IN_DIM), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + beta * KLD

# ==================== 6. VAE 模型 ====================
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

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, IN_DIM))
        z = self.sampling(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

# ==================== 7. 训练/测试函数 ====================
def train_epoch(model, loader, optimizer, beta):
    model.train()
    total_loss = 0
    for graph in loader:
        graph = graph.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, log_var = model(graph)
        loss = loss_function(recon, graph, mu, log_var, beta)
        loss.backward()
        
        # ✅ 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def test_epoch(model, loader, beta):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(DEVICE)
            recon, mu, log_var = model(graph)
            loss = loss_function(recon, graph, mu, log_var, beta)
            total_loss += loss.item()
    return total_loss

# ==================== 8. 主程序 ====================
if __name__ == "__main__":
    print(f"设备: {DEVICE}")
    
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("\n开始训练...")
    train_losses, test_losses = [], []
    
    # ==================== 温和单调退火 KL Annealing ====================
    # 前100个epoch beta=0（专注重建），然后线性增长到1
    for epoch in range(1, EPOCH + 1):
        beta = min(1.0, max(0.0, (epoch - 100) / 200))
        
        train_loss = train_epoch(model, train_loader, optimizer, beta) / len(train_loader.dataset)
        test_loss = test_epoch(model, test_loader, beta) / len(test_loader.dataset)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCH} | Beta: {beta:.2f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # ==================== 滚动保存 Checkpoint ====================
        if epoch % 100 == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            ckpt_filename = f"model_epoch{epoch}_{timestamp}.pth"
            save_path = os.path.join(DATA_DISK_CKPT_DIR, ckpt_filename)
            
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                }, save_path)
                print(f"✅ Checkpoint 已保存至数据盘: {save_path}")
            except Exception as e:
                print(f"❌ 保存失败: {e}")

    # 训练结束后保存最终模型和结果
    final_save_path = os.path.join(DATA_DISK_CKPT_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"\n最终模型已保存: {final_save_path}")
    
    pd.DataFrame({
        'Epoch': range(1, EPOCH + 1),
        'Train Loss': train_losses,
        'Test Loss': test_losses
    }).to_csv(os.path.join(DATA_DISK_CKPT_DIR, "losses.csv"), index=False)
    print(f"损失已保存: {os.path.join(DATA_DISK_CKPT_DIR, 'losses.csv')}")
    
    print("\n✅ 完成!")