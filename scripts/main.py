#!/usr/bin/env python3
"""
DeepISP推論メインスクリプト
28点のISPデータを入力として、ISP2HFAとISP2HFAprogの両方の推論を実行
"""

import argparse
import json
import numpy as np
from isp2hfa_inference import predict_isp2hfa
from isp2hfaprog_inference import predict_isp2hfaprog


def parse_isp_input(input_str):
    """
    ISP入力データをパース
    
    Args:
        input_str: カンマ区切りの文字列 or JSONファイルパス
    
    Returns:
        list: 28点のISPデータ
    """
    # JSONファイルの場合
    if input_str.endswith('.json'):
        with open(input_str, 'r') as f:
            data = json.load(f)
            if 'isp_data' in data:
                isp_data = data['isp_data']
            else:
                isp_data = data
    # カンマ区切り文字列の場合
    else:
        isp_data = [int(x.strip()) for x in input_str.split(',')]
    
    # 検証
    if len(isp_data) != 28:
        raise ValueError(f"ISPデータは28点である必要があります。入力: {len(isp_data)}点")
    
    if not all(x in [0, 1] for x in isp_data):
        raise ValueError("ISPデータは0または1のみを含む必要があります")
    
    return isp_data


def main():
    parser = argparse.ArgumentParser(description='DeepISP推論スクリプト')
    parser.add_argument('--isp', type=str, required=True,
                        help='28点のISPデータ (カンマ区切り or JSONファイルパス)')
    parser.add_argument('--model', choices=['both', 'hfa', 'prog'], default='both',
                        help='実行するモデル (default: both)')
    parser.add_argument('--hfa-model-path', type=str, 
                        default='./params/ISP2HFA/simulated_240318',
                        help='ISP2HFAモデルのパス')
    parser.add_argument('--prog-model-path', type=str, 
                        default='./params/ISP2HFAprog/simulated_240318',
                        help='ISP2HFAprogモデルのパス')
    parser.add_argument('--output', type=str, help='結果を保存するJSONファイルパス')
    
    args = parser.parse_args()
    
    # ISPデータのパース
    try:
        isp_data = parse_isp_input(args.isp)
    except Exception as e:
        print(f"エラー: {e}")
        return 1
    
    results = {}
    
    # ISP2HFA推論
    if args.model in ['both', 'hfa']:
        print("\n=== ISP2HFA推論結果 ===")
        try:
            hfa_result = predict_isp2hfa(isp_data, args.hfa_model_path)
            results['ISP2HFA'] = hfa_result
            
            print(f"使用モデル数: {hfa_result['num_models']}")
            print(f"MD: {hfa_result['MD']:.2f} dB")
            print(f"PSD: {hfa_result['PSD']:.2f} dB")
            print(f"VFI: {hfa_result['VFI']:.1f} %")
            print(f"GHT異常確率: {hfa_result['GHT_probability']:.1%}")
            
            # Pattern mapとTotal mapの概要
            pattern_map = hfa_result['pattern_map']
            total_map = hfa_result['total_map']
            print(f"\nPattern Deviation Map:")
            print(f"  正常点 (class 0): {np.sum(pattern_map == 0)}/52")
            print(f"  軽度異常点 (class 1-2): {np.sum((pattern_map >= 1) & (pattern_map <= 2))}/52")
            print(f"  重度異常点 (class 3-4): {np.sum(pattern_map >= 3)}/52")
            
            print(f"\nTotal Deviation Map:")
            print(f"  正常点 (class 0): {np.sum(total_map == 0)}/52")
            print(f"  軽度異常点 (class 1-2): {np.sum((total_map >= 1) & (total_map <= 2))}/52")
            print(f"  重度異常点 (class 3-4): {np.sum(total_map >= 3)}/52")
            
        except Exception as e:
            print(f"ISP2HFA推論エラー: {e}")
            results['ISP2HFA'] = {'error': str(e)}
    
    # ISP2HFAprog推論
    if args.model in ['both', 'prog']:
        print("\n=== ISP2HFAprog推論結果 ===")
        try:
            prog_result = predict_isp2hfaprog(isp_data, args.prog_model_path)
            results['ISP2HFAprog'] = prog_result
            
            print(f"使用モデル数: {prog_result['num_models']}")
            print(f"MD slope: {prog_result['MD_slope']:.3f} dB/year")
            print(f"VFI slope: {prog_result['VFI_slope']:.3f} %/year")
            print(f"MD進行確率: {prog_result['MD_progression_probability']:.1%}")
            print(f"VFI進行確率: {prog_result['VFI_progression_probability']:.1%}")
            
            # 進行判定
            if prog_result['MD_progression_probability'] > 0.5:
                print("  → MD進行の可能性が高い")
            if prog_result['VFI_progression_probability'] > 0.5:
                print("  → VFI進行の可能性が高い")
                
        except Exception as e:
            print(f"ISP2HFAprog推論エラー: {e}")
            results['ISP2HFAprog'] = {'error': str(e)}
    
    # 結果の保存
    if args.output:
        # numpy配列をリストに変換
        if 'ISP2HFA' in results and 'pattern_map' in results['ISP2HFA']:
            results['ISP2HFA']['pattern_map'] = results['ISP2HFA']['pattern_map'].tolist()
            results['ISP2HFA']['total_map'] = results['ISP2HFA']['total_map'].tolist()
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n結果を {args.output} に保存しました")
    
    return 0


if __name__ == "__main__":
    exit(main())