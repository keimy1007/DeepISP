import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import sys
from src.util import exam_cfg

# figureの設定
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['figure.titlesize'] = 15
# matplotlib.rcParams['font.style'] = 'italic'


# 必要な外部データの読み込み
import matplotlib.image as mpimg
images = [mpimg.imread(f'images/hfa_map/image{i}.png') for i in range(5)]
coord_isp = pd.read_csv("datasets/solid/coord_isp.csv", index_col=0)
coord_hfa24 = pd.read_csv("datasets/solid/coord_hfa24.csv", index_col=0)


################ 数字をHFA24やISP上にプロット ################

# 盲点のnanパディングされてる想定
def plot_HFA24(scores, score_name, cmap="YlGn", target_name=None, save_path=None, digittype=None):
    figsize = (3, 3)

    # cmap = "Oranges"

    plt.figure(figsize=figsize)
    plt.gca().set_aspect('equal', adjustable='box')

    cmap = plt.get_cmap(cmap)
    valid_scores = [score for score in scores if not np.isnan(score)]
    norm = plt.Normalize(vmin=min(valid_scores), vmax=max(valid_scores))
    colors = [cmap(norm(score)) if not np.isnan(score) else None for score in scores]

    for i, color in enumerate(colors):
        if color is not None:  # NaNでない場合のみプロット
            facecolor = list(color)
            facecolor[-1] *= 0.6  # alphaを減らす
            edgecolor = list(color)
            
            x, y = exam_cfg.get_hfa24_coord(i + 1)
            plt.scatter(x, y, s=180, c=[facecolor], edgecolors=[edgecolor], linewidths=0.5)
            # スコアがNaNでない場合のみテキストを表示
            if not np.isnan(scores[i]):
                # テキストの背景に白い四角を追加
                plt.gca().add_patch(plt.Rectangle((x - 2.3, y - 1), 4.6, 2, color='white', alpha=0.3))
                if digittype == "int":
                    plt.text(x, y - 0.05, f"{int(scores[i])}", ha='center', va='center', fontsize=6)
                else:
                    plt.text(x, y - 0.05, f"{scores[i]:.2f}", ha='center', va='center', fontsize=6)

    # カラーバー
    # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Accuracy')
    # cbar.set_label(score_name, fontsize=12)

    plt.xlim(-30, 25)
    plt.ylim(-25, 25)
    plt.xlabel('Degrees')
    plt.ylabel('Degrees')
    plt.title(f'{target_name}', fontsize=15)
    
    plt.grid(alpha=0)
    color = "black"
    plt.axhline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # Y=0 の水平線（X軸）
    plt.axvline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # X=0 の垂直線（Y軸）

    # plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.97, bottom=0.1)
    plt.savefig(f"{save_path}/{target_name}_{score_name}_on_HFA24.pdf", transparent=True)

    return


def plot_HFA10(scores, score_name, cmap="YlGn", target_name=None, save_path=None, digittype=None):
    figsize = (3, 3)
    # cmap = "Oranges"

    plt.figure(figsize=figsize)
    plt.gca().set_aspect('equal', adjustable='box')

    cmap = plt.get_cmap(cmap)
    valid_scores = [score for score in scores if not np.isnan(score)]
    norm = plt.Normalize(vmin=min(valid_scores), vmax=max(valid_scores))
    colors = [cmap(norm(score)) if not np.isnan(score) else None for score in scores]

    for i, color in enumerate(colors):
        if color is not None:  # NaNでない場合のみプロット
            facecolor = list(color)
            facecolor[-1] *= 0.6  # alphaを減らす
            edgecolor = list(color)
            
            x, y = exam_cfg.get_hfa10_coord(i + 1)
            plt.scatter(x, y, s=40, c=[facecolor], edgecolors=[edgecolor], linewidths=0.5)
            # スコアがNaNでない場合のみテキストを表示
            if not np.isnan(scores[i]):
                # テキストの背景に白い四角を追加
                plt.gca().add_patch(plt.Rectangle((x - 2.3, y - 1), 4.6, 2, color='white', alpha=0.3))
                if digittype == "int":
                    plt.text(x, y - 0.05, f"{int(scores[i])}", ha='center', va='center', fontsize=5)
                else:
                    plt.text(x, y - 0.05, f"{scores[i]:.2f}", ha='center', va='center', fontsize=5)


    # カラーバー
    # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Accuracy')
    # cbar.set_label(score_name, fontsize=12)


    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.xlabel('Degrees')
    plt.ylabel('Degrees')
    plt.title(f'{target_name}', fontsize=15)
    
    plt.grid(alpha=0)
    color = "black"
    plt.axhline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # Y=0 の水平線（X軸）
    plt.axvline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # X=0 の垂直線（Y軸）

    # plt.tight_layout()
    plt.subplots_adjust(left=0.3, right=0.8, top=0.7, bottom=0.3)
    plt.savefig(f"{save_path}/{target_name}_{score_name}_on_HFA10.pdf", transparent=True)

    return



def plot_ISP(scores, score_name, cmap="YlGn", target_name=None, save_path=None, digittype=None):
    figsize = (3, 2.4)

    # cmap = "Oranges"

    plt.figure(figsize=figsize)

    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=0, vmax=0.03)
    colors = [cmap(norm(score)) for score in scores]


    for i, color in enumerate(colors):
        # facecolorを薄めに設定し、edgecolorを濃いめに設定
        facecolor = list(color)
        facecolor[-1] *= 0.6  # alphaを減らす
        edgecolor = list(color)
        # edgecolor[-1] *= 0.9  # alphaを増やす
        
        x,y = exam_cfg.get_isp_coord(i+1)
        plt.scatter(x, y, s=20, c=[facecolor], edgecolors=[edgecolor], linewidths=1)
        # テキストの背景に白い四角を追加
        plt.gca().add_patch(plt.Rectangle((x-1, y-0.5), 2, 1, color='white', alpha=0.3))

        if digittype == "int":
            plt.text(x, y-0.05, f"{int(scores[i])}", ha='center', va='center', fontsize=6)
        else:
            plt.text(x, y-0.05, f"{scores[i]:2f}", ha='center', va='center', fontsize=6)

    # カラーバー
    # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Accuracy')
    # cbar.set_label(score_name, fontsize=12)

    plt.xlabel('Degrees')
    plt.ylabel('Degrees')
    plt.title(f'{target_name}', fontsize=15)


    plt.grid(alpha=0)
    color = "black"
    plt.axhline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # Y=0 の水平線（X軸）
    plt.axvline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # X=0 の垂直線（Y軸）

    plt.subplots_adjust(left=0.25, right=0.97, top=0.85, bottom=0.25)

    plt.savefig(f"{save_path}/{target_name}_{score_name}_on_ISP.pdf", transparent=True)

    return


################ チャンピオンデータのプロット ################

def visualize_isp_sample(isp_plot, figname, save_path, color1="black", color0="red"):
    plt.figure(figsize=(3, 2.4))
    for i, value in enumerate(isp_plot):
        isp_index = i + 1
        if isp_index in coord_isp.index:
            x = coord_isp.loc[isp_index, 'x']
            y = coord_isp.loc[isp_index, 'y']
            symbol = '○' if value == 1 else '×'  # 1の場合はマル、0の場合はバツ
            color = color1 if value == 1 else color0
            plt.scatter(x, y, color="pink", s=0)  # 丸のサイズを大きくする
            plt.text(x, y, symbol, color=color, ha='center', va='center', fontname="DejaVu Sans", fontsize=10)  # マルまたはバツを表示

    # X軸とY軸に青色の直線を追加
    color = "black"
    plt.axhline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # Y=0 の水平線（X軸）
    plt.axvline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # X=0 の垂直線（Y軸）

    plt.title("ISP", fontsize=15)

    plt.xlabel("Degrees")
    plt.ylabel("Degrees")
    plt.xlim(-30, 12)  
    plt.ylim(-18, 18)
    plt.grid(False)
    
    # plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.97, top=0.85, bottom=0.25)
    plt.savefig(f"{save_path}/{figname}.pdf", transparent=True)



def visualize_hfa_sample(hfa_plot, figname, save_path, title=None):

   # 盲点を追加
    hfa_plot = np.insert(hfa_plot, 25, -1)
    hfa_plot = np.insert(hfa_plot, 34, -1)

    plt.figure(figsize=(3, 3))
    plt.gca().set_aspect('equal', adjustable='box')

    for i, value in enumerate(hfa_plot):
        # 盲点はスルーする
        if value == -1: continue

        hfa_index = i + 1
        if hfa_index in coord_hfa24.index:
            x = coord_hfa24.loc[hfa_index, 'x']
            y = coord_hfa24.loc[hfa_index, 'y']

            colors = [(val/255, 0, 0, alpha) for val,alpha in zip([220,190,160], [0.6, 0.8, 1.0])]
            color = 'blue' if value == 0 else colors[0] if value == 1 else colors[0] if value == 2 else colors[1] if value == 3 else colors[2]

            plt.scatter(x, y, color="pink", s=0)  # 丸のサイズを大きくする

            # if value == 0:
            #     symbol = '○'
            #     plt.text(x, y, symbol, color=color, ha='center', va='center', fontname="DejaVu Sans", fontsize=2)  # マルまたはバツを表示
            # else:
            #     offset = 1.2
            #     plt.imshow(images[value], extent=(x - offset, x + offset, y - offset, y + offset))

            offset = 1.6
            plt.imshow(images[value], extent=(x - offset, x + offset, y - offset, y + offset))


    # X軸とY軸に青色の直線を追加
    color = "black"
    plt.axhline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # Y=0 の水平線（X軸）
    plt.axvline(0, color=color, alpha=1, linestyle='--', lw=0.5)  # X=0 の垂直線（Y軸）

    plt.title(f"{title}", fontsize=15)
    plt.xlabel("Degrees")
    plt.ylabel("Degrees")
    plt.xlim(-32, 27)  
    plt.ylim(-27, 27)
    plt.grid(False)
    
    # plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.15)
    plt.savefig(f"{save_path}/{figname}.pdf", transparent=True)

