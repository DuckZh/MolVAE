"""
训练主程序：VAE 分子生成模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
from config import *

# 禁用 RDKit 警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings(action='ignore')


# ==================== 特征矩阵 → 分子转换 ====================
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


# ==================== 推理生成分子 ====================
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
                strict=False
            )
            if mol and '.' not in Chem.MolToSmiles(mol):
                smiles_list.append(Chem.MolToSmiles(mol))
                logp_list.append(MolLogP(mol))
                mr_list.append(MolMR(mol))
        except Exception as e:
            continue
    
    # 确保三个列表长度一致
    min_len = min(len(smiles_list), len(logp_list), len(mr_list))
    return pd.DataFrame({
        'SMILES': smiles_list[:min_len], 
        'logP': logp_list[:min_len], 
        'MR': mr_list[:min_len]
    })


# ==================== 损失函数 ====================
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, IN_DIM), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * 0.001
    return BCE + KLD


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


# ==================== 训练/测试函数 ====================
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for graph in loader:
        graph = graph.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, log_var = model(graph)
        loss = loss_function(recon, graph, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def test_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(DEVICE)
            recon, mu, log_var = model(graph)
            loss = loss_function(recon, graph, mu, log_var)
            total_loss += loss.item()
    return total_loss


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"设备: {DEVICE}")
    from dataset import train_loader, test_loader
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print("\n开始训练...")
    train_losses, test_losses = [], []
    for epoch in range(1, EPOCH + 1):
        train_loss = train_epoch(model, train_loader, optimizer) / len(train_loader.dataset)
        test_loss = test_epoch(model, test_loader) / len(test_loader.dataset)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCH} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    torch.save(model.state_dict(), PT_PATH)
    print(f"\n模型已保存: {PT_PATH}")
    pd.DataFrame({
        'Epoch': range(1, EPOCH + 1),
        'Train Loss': train_losses,
        'Test Loss': test_losses
    }).to_csv(LOSS_PATH, index=False)
    print(f"损失已保存: {LOSS_PATH}")
    model.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
    print("\n生成分子...")
    result_df = vae_results(model)
    result_df.to_csv(RESULT_PATH, index=False)
    print(f"生成 {len(result_df)} 个有效分子，保存至: {RESULT_PATH}")
    print("\n✅ 完成!")
