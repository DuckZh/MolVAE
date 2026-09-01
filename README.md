
# MolVAE - 分子生成 VAE

基于 VAE 的二维分子生成模型。

## 安装

```bash
pip install -r requirements.txt
```

## 数据

使用 MOSES 数据集，放在 data/ 目录下。

## 训练

```bash
python train.py
```

## 生成

```bash
python generate.py
```

## 结果

- 模型权重: result/model.pth
- 损失记录: result/losses.csv
- 生成结果: result/result.csv

## requirements.txt

```txt
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
rdkit-pypi>=2021.09.4
```