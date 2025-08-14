import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys 
import torch
import torch.nn as nn

# figureの設定
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['figure.titlesize'] = 15
# matplotlib.rcParams['font.style'] = 'italic'
figsize = (3, 3)


# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_curve, auc

# 自作ライブラリ
from src.preprocessor import preprocess_targets
from src.visualize import plot_HFA24

import warnings
warnings.filterwarnings('ignore')


# 模様の評価用：calcのときに盲点はnp.nanで埋めておく
def calc_class_metrics(y_true_class, y_pred_class):
    accuracies, presicions, recalls, f1s = [], [], [], []
    for idx in range(y_true_class.shape[-1]):
        accuracies.append(accuracy_score(y_true_class[:, idx], y_pred_class[:, idx]))
        presicions.append(precision_score(y_true_class[:, idx], y_pred_class[:, idx], average="micro"))
        recalls.append(recall_score(y_true_class[:, idx], y_pred_class[:, idx], average="micro"))
        f1s.append(f1_score(y_true_class[:, idx], y_pred_class[:, idx], average="micro"))

    accuracies = np.array(accuracies)
    accuracies = np.insert(accuracies, 25, np.nan)
    accuracies = np.insert(accuracies, 34, np.nan)

    presicions = np.array(presicions)
    presicions = np.insert(presicions, 25, np.nan)
    presicions = np.insert(presicions, 34, np.nan)

    recalls = np.array(recalls)
    recalls = np.insert(recalls, 25, np.nan)
    recalls = np.insert(recalls, 34, np.nan)

    f1s = np.array(f1s)
    f1s = np.insert(f1s, 25, np.nan)
    f1s = np.insert(f1s, 34, np.nan)

    return {
        "Accuracy": accuracies,
        "Precision": presicions,
        "Recall": recalls,
        "F1-score": f1s
    }

def plot_class_metrics(y_pred, y_true, target_name=None, save_path=None, cmap="YlGn"):
    res_class = calc_class_metrics(y_pred, y_true)
    for score_name, scores in res_class.items():
        # if score_name != "F1-score": continue
        plot_HFA24(scores, score_name=score_name, target_name=target_name, save_path=save_path, cmap=cmap)

# 連続値の評価用
def calc_reg_metrics(y_true_scaled, y_pred_scaled, scaler):
    reg_true_scores = scaler.inverse_transform(y_true_scaled).flatten()
    reg_pred_scores = scaler.inverse_transform(y_pred_scaled).flatten()
    
    return {"metric": {
        "RMSE":mean_squared_error(reg_true_scores, reg_pred_scores)**0.5,
        "MAE":mean_absolute_error(reg_true_scores, reg_pred_scores)
        }, 
        "True scores": reg_true_scores,
        "Pred scores": reg_pred_scores
    }

def plot_reg_metrics(y_pred_scaled, y_true_scaled, scaler, target_name=None, save_path=None):
    # 辞書 {metrics, y_true, y_pred}
    res_reg = calc_reg_metrics(y_pred_scaled, y_true_scaled, scaler=scaler)

    offset = 0.3

    y_true = res_reg["True scores"]
    y_pred = res_reg["Pred scores"]

    if target_name == "MD_slope":
        min_, max_ = -2, 0
    elif target_name == "VFI_slope":
        min_, max_ = -4, 0
    else:        
        min_ = min(y_true.min(), y_pred.min())
        max_ = max(y_true.max(), y_pred.max())

    rmse = mean_squared_error(y_true, y_pred)**0.5
    mae = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=figsize)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(y_pred, y_true, marker=".", color="blue", s=10)

    plt.plot(
        [max_+offset, min_-offset],
        [max_+offset, min_-offset],
        color="darkorange",
        lw = 1,
        linestyle="dashed", 
        )
    
    plt.grid(alpha = 0.3)
    plt.xlim(min_-offset, max_+offset)
    plt.ylim(min_-offset, max_+offset)

    plt.title(f"{target_name}", fontsize=15)
    plt.xlabel(f"Predicted")
    plt.ylabel(f"True")
    # plt.legend(loc="lower right")

    # plt.tight_layout()
    plt.subplots_adjust(left=0.27, right=0.9, top=0.95, bottom=0.1)
    plt.savefig(f"{save_path}/{target_name}_scatter.pdf")


# バイナリ値の評価用
def calc_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def plot_roc_curve(y_true, y_pred, target_name=None, save_path=None):
    # バイナリラベルと予測値の形状を一致させる
    y_pred = y_pred.ravel()

    # ROC曲線の計算
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # ROC曲線のプロット
    plt.figure(figsize=figsize)
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.plot(fpr, tpr, color='blue', lw=1, label=f'AUC = {roc_auc:.2f})')
    plt.plot(fpr, tpr, color='green', lw=1)
    plt.plot([0, 1], [0, 1], color='darkorange', lw=1, linestyle='dashed')

    plt.grid(alpha = 0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{target_name}', fontsize=15)
    # plt.legend(loc="lower right")

    # plt.tight_layout()
    plt.subplots_adjust(left=0.27, right=0.9, top=0.95, bottom=0.1)
    plt.savefig(f"{save_path}/{target_name}_ROC.pdf")


def save_predictions_per_sample(y_test_tensors, y_pred_tensors, scalers):
    # テンソルを展開する
    y_true_pattern, y_true_total, y_true_md, y_true_psd, y_true_vfi, y_true_ght = y_test_tensors
    y_pred_pattern, y_pred_total, y_pred_md, y_pred_psd, y_pred_vfi, y_pred_ght = y_pred_tensors

    # 実際のデータのDataFrameを生成する関数
    def real_to_df(tensor, label):
        if "pattern" in label or "total" in label:
            columns = [f'{label}_{i}' for i in range(tensor.shape[1])]
        else:
            columns = [label]

        df = pd.DataFrame(tensor.numpy(), columns=columns)

        if 'md' in label: df['md_true'] = scalers["MD"].inverse_transform(tensor).flatten()
        if 'psd' in label: df['psd_true'] = scalers["PSD"].inverse_transform(tensor).flatten()
        if 'vfi' in label: df['vfi_true'] = scalers["VFI"].inverse_transform(tensor).flatten()
        # if 'ght' in label: df['ght_true'] = torch.sigmoid(tensor).numpy()

        return df

    # 予測データのDataFrameを生成する関数
    def pred_to_df(tensor, label, reshape=False, num_classes=None):
        # patternとtotalの模様予測
        if reshape and num_classes:
            tensor_reshaped = tensor.view(tensor.shape[0], -1, num_classes)
            columns_raw = [f'{label}_{i}_{j}' for i in range(tensor_reshaped.shape[1]) for j in range(num_classes)]
            df = pd.DataFrame(tensor_reshaped.numpy().reshape(tensor.shape[0], -1), columns=columns_raw)

        # MD,PSD,VFI,GHTの場合
        else:
            if tensor.dim() == 1: tensor = tensor.unsqueeze(1)  # ghtを含む
            # columns = [f'{label}{i}' for i in range(tensor.shape[1])] if tensor.shape[1] > 1 else [label]
            columns = [label]
            df = pd.DataFrame(tensor.numpy(), columns=columns)

        # それぞれの場合の追加処理
        if 'md' in label: df['md_pred'] = scalers["MD"].inverse_transform(tensor).flatten()
        if 'psd' in label: df['psd_pred'] = scalers["PSD"].inverse_transform(tensor).flatten()
        if 'vfi' in label: df['vfi_pred'] = scalers["VFI"].inverse_transform(tensor).flatten()
        if 'ght' in label: df['ght_pred'] = torch.sigmoid(tensor).numpy()

        return df

    # 予測データの最大確率クラスを決定する関数
    def pred_max_class_to_df(tensor, label):
        tensor_reshaped = tensor.view(tensor.shape[0], 52, 5)  # 予測データを再構成
        max_indices = tensor_reshaped.argmax(dim=2)  # 最大確率クラスを取得
        columns = [f'{label}_{i}' for i in range(max_indices.shape[1])]
        return pd.DataFrame(max_indices.numpy(), columns=columns)

    # true
    df_true_md = real_to_df(y_true_md, 'md_true_scaled')
    df_true_psd = real_to_df(y_true_psd, 'psd_true_scaled')
    df_true_vfi = real_to_df(y_true_vfi, 'vfi_true_scaled')
    df_true_ght = real_to_df(y_true_ght.unsqueeze(1), 'ght_true')  # ghtの形状を調整
    df_true_pattern = real_to_df(y_true_pattern, 'pattern_true')
    df_true_total = real_to_df(y_true_total, 'total_true')

    # pred
    df_pred_md = pred_to_df(y_pred_md, 'md_pred_scaled')
    df_pred_psd = pred_to_df(y_pred_psd, 'psd_pred_scaled')
    df_pred_vfi = pred_to_df(y_pred_vfi, 'vfi_pred_scaled')
    df_pred_ght = pred_to_df(y_pred_ght, 'ght_pred_logit')  # ghtの特別な処理を含む
    df_pred_pattern_max_class = pred_max_class_to_df(y_pred_pattern, 'pattern_pred')
    df_pred_total_max_class = pred_max_class_to_df(y_pred_total, 'total_pred')
    df_pred_pattern_raw = pred_to_df(y_pred_pattern, 'pattern_pred_logit', reshape=True, num_classes=5)
    df_pred_total_raw = pred_to_df(y_pred_total, 'total_pred_logit', reshape=True, num_classes=5)

    # 実際のデータと予測データのDataFrameを結合
    df = pd.concat([
        df_true_md, df_true_psd, df_true_vfi, df_true_ght,
        df_true_pattern, df_true_total,
        df_pred_pattern_max_class, df_pred_total_max_class,
        df_pred_pattern_raw, df_pred_total_raw, df_pred_md, df_pred_psd, df_pred_vfi, df_pred_ght
    ], axis=1)

    return df

