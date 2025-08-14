# DeepISP 環境構築ガイド

## Conda環境の構築（推奨）

### 1. environment.ymlを使用する方法
```bash
# 環境を作成
conda env create -f environment.yml

# 環境を有効化
conda activate DeepISP
```

### 2. 手動で環境を構築する方法
```bash
# 新しい環境を作成
conda create -n DeepISP python=3.9

# 環境を有効化
conda activate DeepISP

# condaでパッケージをインストール
conda install -c conda-forge numpy=1.21.5 pandas=1.4.4 matplotlib=3.6.2 scikit-learn=1.1.3 jupyter notebook=6.5.2 ipython=8.7.0 tqdm=4.64.1

# PyTorchをインストール（Mac M1/M2の場合）
pip install torch torchvision torchaudio

# PyTorchをインストール（その他の環境の場合）
# conda install pytorch torchvision torchaudio -c pytorch
```

## pip環境の構築（代替方法）

```bash
# 仮想環境を作成
python -m venv venv_deepisp

# 環境を有効化（Mac/Linux）
source venv_deepisp/bin/activate

# 環境を有効化（Windows）
# venv_deepisp\Scripts\activate

# requirements.txtからインストール
pip install -r requirements.txt
```

## 環境の確認

```bash
# Python環境の確認
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

# Jupyter Notebookの起動
jupyter notebook
```

## トラブルシューティング

### Mac M1/M2でPyTorchが動作しない場合
```bash
# MPSサポート付きPyTorchをインストール
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### メモリ不足エラーが発生する場合
CPUモードで実行してください：
```python
device = "cpu"  # instead of "mps" or "cuda"
```