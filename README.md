# DeepISP

Deep learning models for ISP (Imo Sphenoidal Perimetry) to HFA (Humphrey Field Analyzer) conversion

## Overview

ISPデータからHFAパラメータを予測する深層学習モデル:
- **ISP2HFA**: ISPからHFA視野検査結果（MD, PSD, VFI, Pattern/Total Deviation Map）を予測
- **ISP2HFAprog**: ISPから視野進行指標（MD slope, VFI slope, 進行確率）を予測

## Requirements

```bash
pip install -r requirements.txt
```

**Note**: NumPy 2.xで互換性の問題が発生する場合:
```bash
pip install "numpy<2.0"
```

## Usage

### Basic Inference

```bash
# 28点のISPデータから推論実行
python scripts/main.py --isp "1,0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0"

# JSONファイルから入力
python scripts/main.py --isp scripts/example_input.json

# 結果を保存
python scripts/main.py --isp scripts/example_input.json --output results.json

# 年齢を指定（デフォルト: 50）
python scripts/main.py --isp "1,0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0" --age 65
```

### ISP Simulation

HFA測定値からISPデータを生成:

```bash
# HFA24-2データ（54点）からISPを生成
python scripts/isp_simulation.py \
  --hfa24 "30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30" \
  --age 50

# HFA10-2データ（68点）からISPを生成
python scripts/isp_simulation.py \
  --hfa10 "32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32" \
  --age 50
```

### Visualization

ISPやHFAデータの可視化:

```bash
# ISPデータの可視化
python scripts/visualization.py --type isp \
  --isp-data "1,0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0" \
  --output isp_visual.png

# HFA24データの可視化
python scripts/visualization.py --type hfa24 \
  --hfa-data "0,0,1,2,0,0,1,1,0,0,2,3,0,1,0,0,1,2,0,0,1,1,0,0,0,0,1,2,0,0,1,1,0,0,2,3,0,1,0,0,1,2,0,0,1,1,0,0,0,1,2,0" \
  --output hfa24_visual.png

# ISPとHFAの比較
python scripts/visualization.py --type compare \
  --isp-data "1,0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0" \
  --hfa-data "0,0,1,2,0,0,1,1,0,0,2,3,0,1,0,0,1,2,0,0,1,1,0,0,0,0,1,2,0,0,1,1,0,0,2,3,0,1,0,0,1,2,0,0,1,1,0,0,0,1,2,0" \
  --output comparison.png
```

## Directory Structure

```
DeepISP/
├── scripts/
│   ├── main.py                    # メイン推論スクリプト
│   ├── models.py                  # モデル定義
│   ├── isp2hfa_inference.py      # ISP2HFA推論
│   ├── isp2hfaprog_inference.py  # ISP2HFAprog推論
│   ├── isp_simulation.py         # HFAからISP生成
│   ├── visualization.py          # 可視化
│   └── example_input.json        # 入力例
├── datasets_sample/               # サンプルデータセット
│   ├── solid/
│   │   ├── coord_isp.csv        # ISP測定点座標
│   │   └── coord_hfa24.csv      # HFA24測定点座標
│   └── flex/
│       └── *.csv                 # サンプルデータ
├── params/                        # モデルパラメータ
│   ├── ISP2HFA/
│   │   └── simulated_240318/
│   └── ISP2HFAprog/
│       └── simulated_240318/
└── requirements.txt              # 依存パッケージ
```

## Data Format

### Input ISP Data
- 28点の0/1データ（0: 見えない, 1: 見える）
- カンマ区切り文字列またはJSON配列として入力

### Output Format

#### ISP2HFA Output
```json
{
  "MD": -5.2,           // Mean Deviation (dB)
  "PSD": 4.3,          // Pattern Standard Deviation (dB)
  "VFI": 85.0,         // Visual Field Index (%)
  "GHT_prob": 0.23,    // Glaucoma Hemifield Test異常確率
  "pattern_map": [...], // Pattern Deviation Map (52点, 0-4のクラス)
  "total_map": [...]    // Total Deviation Map (52点, 0-4のクラス)
}
```

#### ISP2HFAprog Output
```json
{
  "MD_slope": -0.5,     // MD変化率 (dB/year)
  "VFI_slope": -1.2,    // VFI変化率 (%/year)
  "MD_prog_prob": 0.85, // MD進行確率
  "VFI_prog_prob": 0.72 // VFI進行確率
}
```

## Model Information

### ISP2HFA Model
- Architecture: 3層MLP with skip connections
- Hidden dimensions: [64, 64]
- Input: 29次元（ISP 28点 + 年齢）
- Output: 6タスク（pattern, total, MD, PSD, VFI, GHT）

### ISP2HFAprog Model
- Architecture: 3層MLP with skip connections
- Hidden dimensions: [16, 16]
- Input: 29次元（ISP 28点 + 年齢）
- Output: 4タスク（MD_slope, VFI_slope, MD_prog, VFI_prog）

## Environment Variables

サンプルデータを使用する場合:
```bash
export DATASETS_DIR=./datasets_sample
python scripts/main.py --isp "1,0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0"
```

実データを使用する場合（デフォルト）:
```bash
# DATASETS_DIRを設定しない場合、./datasetsが使用される
python scripts/main.py --isp "1,0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0"
```

## Notes

- モデルの重みファイルは`params/`ディレクトリに配置してください
- 実データは`datasets/`ディレクトリに配置してください（gitignore対象）
- サンプルデータは`datasets_sample/`に含まれています

## License

[License information here]

## Citation

```bibtex
[Citation information here]
```