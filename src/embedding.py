import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import sys
from src.util import exam_cfg

from sklearn.manifold import TSNE
from umap import UMAP

# figureの設定
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['figure.titlesize'] = 15
# matplotlib.rcParams['font.style'] = 'italic'
figsize = (3, 3)


################ TSNE ################


def plot_TSNE(X_df, y_df, group_df, var_name=None, color1="red", color2="lightgreen", vmin=0, vmax=100, save_path=None, random_state=42):
    reducer = TSNE(n_components=2, random_state=random_state)
    X_2d = reducer.fit_transform(X_df)

    # y_df に基づいた色のグラデーションを設定
    colors = y_df.values.flatten()
    # cmap = mcolors.LinearSegmentedColormap.from_list("custom1", ["pink", "blue"])

    if vmin <= vmax:
        cmap = mcolors.LinearSegmentedColormap.from_list("custom1", [color1, color2])
    else:
        vmin,vmax = vmax,vmin
        cmap = mcolors.LinearSegmentedColormap.from_list("custom1", [color2, color1])

    # 2Dの分布をプロット
    plt.figure(figsize=(5, 4))

    # マーカーのプロット：group_df のカテゴリーごとに記号を変える
    unique_categories = np.unique(group_df.values.flatten())
    markers = ['.', 'x']  # マルとバツの記号
    for category, marker in zip(unique_categories, markers):
        indices = group_df.values.flatten() == category

        # バツ
        if marker == 'x':
            if var_name:
                plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c=colors[indices], cmap=cmap, vmin=vmin, vmax=vmax, marker=marker, edgecolor='red', s=25, lw=1, label="Actual")
            else:
                # redかorange
                plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c="blueviolet", alpha=1, marker=marker, s=25, lw=1, label="Actual")
           
        # マル
        else:
            if var_name:
                plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c=colors[indices], cmap=cmap, vmin=vmin, vmax=vmax, marker=marker, s=15, label="Simulated")
            else:
                # blueかgreen
                plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c="darkorange", alpha=1, marker=marker, s=15, label="Simulated")


    if not var_name:
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=9, frameon=True, edgecolor='black', facecolor='white', borderpad=0.2, labelspacing=0.2, handletextpad=0.2)

    
    title = "ISP + Age" if not var_name else var_name
    plt.title(title, fontsize=15)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # レイアウトの微調整
    if var_name:
        # plt.colorbar(label=var_name)
        plt.colorbar()
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    else:
        # カラーバーなし
        plt.subplots_adjust(left=0.15, right=0.75, top=0.9, bottom=0.15)

    # plt.tight_layout()
    plt.savefig(f"{save_path}/TSNE_{var_name}.pdf", dpi=300)



################ UMAP ################

# UMAPは結局使用せず
    
def plot_UMAP(X_df, y_df, group_df, var_name, random_state=42):
    reducer = UMAP(n_components=2, random_state=random_state)
    X_2d = reducer.fit_transform(X_df)

    # y_df に基づいた色のグラデーションを設定
    colors = y_df.values.flatten()
    cmap = mcolors.LinearSegmentedColormap.from_list("custom2", ["pink", "blue"])

    # 2Dの分布をプロット
    plt.figure(figsize=(10, 8))

    # group_df のカテゴリーごとに記号を変えてプロット
    unique_categories = np.unique(group_df.values.flatten())
    markers = ['.', 'x']  # マルとバツの記号

    for category, marker in zip(unique_categories, markers):
        indices = group_df.values.flatten() == category

        if marker == 'x':  # バツマーカーの場合
            plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c=colors[indices], cmap=cmap, marker=marker, linewidths=2, edgecolor='red', s=80, label=category)
            # plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c="red", marker=marker, linewidths=2, edgecolor='red', s=80, label=category)
        else:  # 他のマーカーの場合
            plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c=colors[indices], cmap=cmap, marker=marker, label=category)
            # plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c="blue", marker=marker, label=category)


    plt.colorbar(label=var_name)
    plt.title(f"2D visualization of {var_name} using UMAP")
    plt.xlabel("UMAP feature 1")
    plt.ylabel("UMAP feature 2")
    plt.legend()
    plt.show()
