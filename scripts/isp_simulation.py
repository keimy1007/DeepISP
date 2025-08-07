#!/usr/bin/env python3
"""
HFA測定点データからISPデータを擬似生成するスクリプト
notebook/[ks240222]Fig1_flow_diagram.ipynbを参考に実装
"""

import numpy as np
import pandas as pd
import argparse
import json


# HFA30-2のインデックスをHFA24-2のインデックスにマッピング
MAPPING_HFA30_TO_HFA24 = [
    6, 7, 8, 9,
    12, 13, 14, 15, 16, 17,
    20, 21, 22, 23, 24, 25, 26, 27,
    29, 30, 31, 32, 33, 34, 35, 36, 37,
    39, 40, 41, 42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55, 56, 57,
    60, 61, 62, 63, 64, 65,
    68, 69, 70, 71
]


def load_coordinate_data():
    """ISPとHFAの座標データを読み込み"""
    import os
    datasets_dir = os.getenv('DATASETS_DIR', './datasets')
    coord_isp = pd.read_csv(f"{datasets_dir}/solid/coord_isp.csv")
    coord_hfa24 = pd.read_csv(f"{datasets_dir}/solid/coord_hfa24.csv")
    return coord_isp, coord_hfa24


def get_age_threshold(age, point_index, coord_isp):
    """
    年齢に応じた閾値を取得
    
    Args:
        age: 年齢
        point_index: ISP測定点インデックス（0-27）
        coord_isp: ISP座標データ
    
    Returns:
        閾値
    """
    row = coord_isp.iloc[point_index]
    
    if age < 20:
        return row["th20"]  # 20歳未満は20歳の閾値を使用
    elif age < 30:
        return row["th20"]
    elif age < 40:
        return row["th30"]
    elif age < 50:
        return row["th40"]
    elif age < 60:
        return row["th50"]
    elif age < 70:
        return row["th60"]
    else:
        return row["th70over"]


def simulate_isp_from_hfa24(hfa24_data, age=50):
    """
    HFA24-2データ（54点）からISPデータ（28点）を生成
    
    Args:
        hfa24_data: 54点のHFA24-2測定値リスト（dB値）
        age: 年齢（デフォルト50歳）
    
    Returns:
        28点のISPデータ（1:見える, 0:見えない）
    """
    coord_isp, coord_hfa24 = load_coordinate_data()
    isp_data = []
    
    # HFA30インデックスからHFA24インデックスへの変換辞書
    hfa30_to_hfa24_dict = {num: i+1 for i, num in enumerate(MAPPING_HFA30_TO_HFA24)}
    
    for i in range(28):
        # 対応するHFA24の測定点を特定
        if coord_isp.iloc[i]["kensa"] == "24-2":
            point_num = int(coord_isp.iloc[i]["index24"])
            # HFA30のインデックスをHFA24のインデックスに変換
            hfa24_index = hfa30_to_hfa24_dict.get(point_num)
            if hfa24_index is None:
                # マッピングがない場合は見えるとする
                isp_value = 1
            else:
                # HFA24データから値を取得（マリオット盲点考慮）
                if hfa24_index <= 25:
                    actual_index = hfa24_index - 1
                elif hfa24_index <= 34:
                    actual_index = hfa24_index - 2  # 26番目の盲点をスキップ
                else:
                    actual_index = hfa24_index - 3  # 26番目と35番目の盲点をスキップ
                
                if actual_index < len(hfa24_data):
                    hfa_value = hfa24_data[actual_index]
                    threshold = get_age_threshold(age, i, coord_isp)
                    isp_value = 1 if hfa_value >= threshold else 0
                else:
                    isp_value = 1
        else:
            # HFA10-2の場合（このスクリプトではHFA24のみ対応）
            isp_value = 1
        
        isp_data.append(isp_value)
    
    return isp_data


def simulate_isp_from_hfa10(hfa10_data, age=50):
    """
    HFA10-2データ（68点）からISPデータ（28点）を生成
    
    Args:
        hfa10_data: 68点のHFA10-2測定値リスト（dB値）
        age: 年齢（デフォルト50歳）
    
    Returns:
        28点のISPデータ（1:見える, 0:見えない）
    """
    coord_isp, _ = load_coordinate_data()
    isp_data = []
    
    for i in range(28):
        if coord_isp.iloc[i]["kensa"] == "10-2":
            point_num = int(coord_isp.iloc[i]["index10"])
            if point_num <= len(hfa10_data):
                hfa_value = hfa10_data[point_num - 1]
                threshold = get_age_threshold(age, i, coord_isp)
                isp_value = 1 if hfa_value >= threshold else 0
            else:
                isp_value = 1
        else:
            # HFA24-2の領域は見えるとする
            isp_value = 1
        
        isp_data.append(isp_value)
    
    return isp_data


def simulate_isp_combined(hfa24_data, hfa10_data, age=50):
    """
    HFA24-2とHFA10-2の両方のデータからISPデータを生成
    
    Args:
        hfa24_data: 54点のHFA24-2測定値リスト（dB値）
        hfa10_data: 68点のHFA10-2測定値リスト（dB値）
        age: 年齢（デフォルト50歳）
    
    Returns:
        28点のISPデータ（1:見える, 0:見えない）
    """
    coord_isp, _ = load_coordinate_data()
    isp_data = []
    
    # HFA30インデックスからHFA24インデックスへの変換辞書
    hfa30_to_hfa24_dict = {num: i+1 for i, num in enumerate(MAPPING_HFA30_TO_HFA24)}
    
    for i in range(28):
        if coord_isp.iloc[i]["kensa"] == "24-2":
            # HFA24-2のデータを使用
            point_num = int(coord_isp.iloc[i]["index24"])
            hfa24_index = hfa30_to_hfa24_dict.get(point_num)
            if hfa24_index is None:
                isp_value = 1
            else:
                # HFA24データから値を取得（マリオット盲点考慮）
                if hfa24_index <= 25:
                    actual_index = hfa24_index - 1
                elif hfa24_index <= 34:
                    actual_index = hfa24_index - 2
                else:
                    actual_index = hfa24_index - 3
                
                if actual_index < len(hfa24_data):
                    hfa_value = hfa24_data[actual_index]
                    threshold = get_age_threshold(age, i, coord_isp)
                    isp_value = 1 if hfa_value >= threshold else 0
                else:
                    isp_value = 1
        else:
            # HFA10-2のデータを使用
            point_num = int(coord_isp.iloc[i]["index10"])
            if point_num <= len(hfa10_data):
                hfa_value = hfa10_data[point_num - 1]
                threshold = get_age_threshold(age, i, coord_isp)
                isp_value = 1 if hfa_value >= threshold else 0
            else:
                isp_value = 1
        
        isp_data.append(isp_value)
    
    return isp_data


def main():
    parser = argparse.ArgumentParser(description='HFAデータからISPデータを擬似生成')
    parser.add_argument('--hfa24', type=str, 
                        help='HFA24-2データ（54点、カンマ区切りまたはJSONファイル）')
    parser.add_argument('--hfa10', type=str,
                        help='HFA10-2データ（68点、カンマ区切りまたはJSONファイル）')
    parser.add_argument('--age', type=int, default=50,
                        help='年齢（デフォルト: 50）')
    parser.add_argument('--output', type=str,
                        help='出力ファイルパス（JSON形式）')
    
    args = parser.parse_args()
    
    # データの読み込み
    def parse_data(data_str):
        if data_str.endswith('.json'):
            with open(data_str, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'values' in data:
                    data = data['values']
        else:
            data = [float(x.strip()) for x in data_str.split(',')]
        return data
    
    # ISP生成
    if args.hfa24 and args.hfa10:
        # 両方のデータがある場合
        hfa24_data = parse_data(args.hfa24)
        hfa10_data = parse_data(args.hfa10)
        if len(hfa24_data) != 54:
            raise ValueError(f"HFA24データは54点必要です。入力: {len(hfa24_data)}点")
        if len(hfa10_data) != 68:
            raise ValueError(f"HFA10データは68点必要です。入力: {len(hfa10_data)}点")
        
        isp_data = simulate_isp_combined(hfa24_data, hfa10_data, args.age)
        source = "HFA24+HFA10"
        
    elif args.hfa24:
        # HFA24のみ
        hfa24_data = parse_data(args.hfa24)
        if len(hfa24_data) != 54:
            raise ValueError(f"HFA24データは54点必要です。入力: {len(hfa24_data)}点")
        
        isp_data = simulate_isp_from_hfa24(hfa24_data, args.age)
        source = "HFA24"
        
    elif args.hfa10:
        # HFA10のみ
        hfa10_data = parse_data(args.hfa10)
        if len(hfa10_data) != 68:
            raise ValueError(f"HFA10データは68点必要です。入力: {len(hfa10_data)}点")
        
        isp_data = simulate_isp_from_hfa10(hfa10_data, args.age)
        source = "HFA10"
        
    else:
        raise ValueError("HFA24またはHFA10のデータが必要です")
    
    # 結果の出力
    result = {
        'isp_data': isp_data,
        'source': source,
        'age': args.age,
        'num_visible': sum(isp_data),
        'num_invisible': 28 - sum(isp_data)
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"ISPデータを生成しました: {args.output}")
    else:
        print(f"生成されたISPデータ（{source}, 年齢{args.age}歳）:")
        print(f"  データ: {isp_data}")
        print(f"  見える点: {result['num_visible']}/28")
        print(f"  見えない点: {result['num_invisible']}/28")
    
    return result


if __name__ == "__main__":
    main()