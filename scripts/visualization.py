#!/usr/bin/env python3
"""
ISPとHFAの測定点可視化スクリプト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import json


def get_isp_coord(index):
    """ISP測定点の座標を取得"""
    import os
    datasets_dir = os.getenv('DATASETS_DIR', './datasets')
    coord_isp = pd.read_csv(f"{datasets_dir}/solid/coord_isp.csv")
    row = coord_isp[coord_isp['index'] == index].iloc[0]
    return row['x'], row['y']


def get_hfa24_coord(index):
    """HFA24測定点の座標を取得（盲点を考慮）"""
    import os
    datasets_dir = os.getenv('DATASETS_DIR', './datasets')
    coord_hfa24 = pd.read_csv(f"{datasets_dir}/solid/coord_hfa24.csv")
    
    # インデックスを調整（26, 35が盲点）
    if index >= 26:
        index += 1
    if index >= 35:
        index += 1
    
    row = coord_hfa24[coord_hfa24['index_24'] == index].iloc[0]
    return row['x'], row['y']


def visualize_isp(isp_data, output_path=None, title="ISP", show_values=True):
    """
    ISP測定点を可視化
    
    Args:
        isp_data: 28点のISPデータ（1,0のリスト）
        output_path: 保存先パス（Noneの場合は表示のみ）
        title: グラフタイトル
        show_values: 値を表示するか
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    # ISPデータをプロット
    for i in range(28):
        x, y = get_isp_coord(i + 1)
        value = isp_data[i]
        
        # 色を設定（1:見える=緑、0:見えない=赤）
        if value == 1:
            color = 'green'
            alpha = 0.6
        else:
            color = 'red'
            alpha = 0.3
        
        circle = plt.Circle((x, y), 1.5, color=color, alpha=alpha)
        ax.add_patch(circle)
        
        if show_values:
            ax.text(x, y, str(value), ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 軸の設定
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Degrees', fontsize=12)
    ax.set_ylabel('Degrees', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"ISP図を保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_hfa24(hfa_data, output_path=None, title="HFA24", cmap='RdYlGn_r', show_values=True):
    """
    HFA24測定点を可視化（52点、盲点除く）
    
    Args:
        hfa_data: 52点のHFAデータ（0-4のクラス値のリスト）
        output_path: 保存先パス（Noneの場合は表示のみ）
        title: グラフタイトル
        cmap: カラーマップ
        show_values: 値を表示するか
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    
    # カラーマップの設定
    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=0, vmax=4)
    
    # HFAデータをプロット
    for i in range(52):
        x, y = get_hfa24_coord(i + 1)
        value = hfa_data[i]
        
        # 色を取得
        color = cmap_obj(norm(value))
        
        # 四角形として表示（HFAらしく）
        rect = plt.Rectangle((x - 1.5, y - 1.5), 3, 3, 
                            facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)
        
        if show_values:
            ax.text(x, y, str(value), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # 盲点を表示
    for blind_spot in [26, 35]:
        x, y = get_hfa24_coord(blind_spot)
        circle = plt.Circle((x, y), 1.5, color='black', alpha=0.2)
        ax.add_patch(circle)
        ax.text(x, y, 'BS', ha='center', va='center', fontsize=8, fontweight='bold', color='gray')
    
    # カラーバー
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Deviation Class', fontsize=10)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Normal', 'p<5%', 'p<2%', 'p<1%', 'p<0.5%'])
    
    # 軸の設定
    ax.set_xlim(-32, 32)
    ax.set_ylim(-28, 28)
    ax.axhline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Degrees', fontsize=12)
    ax.set_ylabel('Degrees', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"HFA図を保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_comparison(isp_data, hfa_pattern, hfa_total, output_path=None):
    """
    ISPとHFA（Pattern/Total Deviation）を並べて表示
    
    Args:
        isp_data: 28点のISPデータ
        hfa_pattern: 52点のPattern Deviationデータ
        hfa_total: 52点のTotal Deviationデータ
        output_path: 保存先パス
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ISPプロット
    ax = axes[0]
    ax.set_aspect('equal')
    for i in range(28):
        x, y = get_isp_coord(i + 1)
        value = isp_data[i]
        color = 'green' if value == 1 else 'red'
        alpha = 0.6 if value == 1 else 0.3
        circle = plt.Circle((x, y), 1.5, color=color, alpha=alpha)
        ax.add_patch(circle)
        ax.text(x, y, str(value), ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Degrees')
    ax.set_title('ISP', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Pattern Deviationプロット
    ax = axes[1]
    ax.set_aspect('equal')
    cmap_obj = plt.get_cmap('RdYlGn_r')
    norm = mcolors.Normalize(vmin=0, vmax=4)
    for i in range(52):
        x, y = get_hfa24_coord(i + 1)
        value = hfa_pattern[i]
        color = cmap_obj(norm(value))
        rect = plt.Rectangle((x - 1.5, y - 1.5), 3, 3, 
                            facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, str(value), ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(-32, 32)
    ax.set_ylim(-28, 28)
    ax.axhline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Degrees')
    ax.set_title('Pattern Deviation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Total Deviationプロット
    ax = axes[2]
    ax.set_aspect('equal')
    for i in range(52):
        x, y = get_hfa24_coord(i + 1)
        value = hfa_total[i]
        color = cmap_obj(norm(value))
        rect = plt.Rectangle((x - 1.5, y - 1.5), 3, 3, 
                            facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, str(value), ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(-32, 32)
    ax.set_ylim(-28, 28)
    ax.axhline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Degrees')
    ax.set_title('Total Deviation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # カラーバー追加（共通）
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1:], shrink=0.6, location='right', pad=0.02)
    cbar.set_label('Deviation Class', fontsize=10)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Normal', 'p<5%', 'p<2%', 'p<1%', 'p<0.5%'])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"比較図を保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='ISPとHFA測定点の可視化')
    parser.add_argument('--type', choices=['isp', 'hfa', 'both'], required=True,
                        help='可視化タイプ')
    parser.add_argument('--isp-data', type=str,
                        help='ISPデータ（カンマ区切りまたはJSONファイル）')
    parser.add_argument('--hfa-pattern', type=str,
                        help='HFA Pattern Deviationデータ（カンマ区切りまたはJSONファイル）')
    parser.add_argument('--hfa-total', type=str,
                        help='HFA Total Deviationデータ（カンマ区切りまたはJSONファイル）')
    parser.add_argument('--output', type=str,
                        help='出力ファイルパス')
    parser.add_argument('--title', type=str, default='',
                        help='グラフタイトル')
    
    args = parser.parse_args()
    
    # データの読み込み
    def parse_data(data_str, expected_len):
        if data_str.endswith('.json'):
            with open(data_str, 'r') as f:
                data = json.load(f)
        else:
            data = [int(x.strip()) for x in data_str.split(',')]
        
        if len(data) != expected_len:
            raise ValueError(f"データ長が不正です。期待: {expected_len}, 実際: {len(data)}")
        
        return data
    
    # 可視化実行
    if args.type == 'isp':
        if not args.isp_data:
            raise ValueError("ISPデータが必要です")
        isp_data = parse_data(args.isp_data, 28)
        visualize_isp(isp_data, args.output, args.title or "ISP")
        
    elif args.type == 'hfa':
        if not args.hfa_pattern:
            raise ValueError("HFA Pattern Deviationデータが必要です")
        hfa_data = parse_data(args.hfa_pattern, 52)
        visualize_hfa24(hfa_data, args.output, args.title or "HFA Pattern Deviation")
        
    elif args.type == 'both':
        if not all([args.isp_data, args.hfa_pattern, args.hfa_total]):
            raise ValueError("ISP、Pattern Deviation、Total Deviationのすべてのデータが必要です")
        isp_data = parse_data(args.isp_data, 28)
        hfa_pattern = parse_data(args.hfa_pattern, 52)
        hfa_total = parse_data(args.hfa_total, 52)
        visualize_comparison(isp_data, hfa_pattern, hfa_total, args.output)


if __name__ == "__main__":
    main()