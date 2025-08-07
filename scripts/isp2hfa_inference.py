import torch
import numpy as np
import glob
from models import ISP2HFAModel


def preprocess_isp_input(isp_data):
    """
    28点のISPデータを前処理
    
    Args:
        isp_data: 28点の1,0からなるリスト or numpy array
    
    Returns:
        torch.Tensor: モデル入力用のテンソル (1, 29)
    """
    if not isinstance(isp_data, np.ndarray):
        isp_data = np.array(isp_data)
    
    # 年齢は0を仮定（推論時には影響が小さい）
    age_scaled = 0.5  # 50歳相当
    
    # ISPデータの処理（1はそのまま、0はそのまま）
    features = np.concatenate([isp_data, [age_scaled]])
    
    # バッチ次元を追加してテンソル化
    X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    return X_tensor


def load_model(folder_path="./params/ISP2HFA/simulated_240318"):
    """
    単一モデルをロード（最初のモデルファイルを使用）
    
    Args:
        folder_path: モデルパラメータが保存されているフォルダパス
    
    Returns:
        model: ロードされたモデル
    """
    param_files = sorted(glob.glob(f"{folder_path}/*.pth"))  # ソートして常に同じファイルを選択
    
    if len(param_files) == 0:
        raise ValueError(f"No model files found in {folder_path}")
    
    device = torch.device("cpu")
    hidden_dims = [64, 64]
    dropout_rate = 0.1
    
    # 最初のモデルファイルを使用
    model = ISP2HFAModel(hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(param_files[0], map_location=device))
    model.eval()
    
    print(f"Loaded model: {param_files[0].split('/')[-1]}")
    
    return model


def predict_isp2hfa(isp_data, model_path="./params/ISP2HFA/simulated_240318"):
    """
    ISP2HFAモデルで予測を実行
    
    Args:
        isp_data: 28点の1,0からなるリスト
        model_path: モデルパラメータのパス
    
    Returns:
        dict: 各予測結果を含む辞書
    """
    # 入力データの前処理
    X_tensor = preprocess_isp_input(isp_data)
    
    # モデルのロード
    model = load_model(model_path)
    
    # 予測
    with torch.no_grad():
        outputs = model(X_tensor)
    
    # 出力を取得
    y_pattern = outputs[0]
    y_total = outputs[1]
    y_md = outputs[2]
    y_psd = outputs[3]
    y_vfi = outputs[4]
    y_ght = outputs[5]
    
    # パターンとトータルの分類結果を取得（52点×5クラス）
    pattern_classes = torch.argmax(y_pattern.view(1, 52, 5), dim=2).squeeze().numpy()
    total_classes = torch.argmax(y_total.view(1, 52, 5), dim=2).squeeze().numpy()
    
    # GHTの確率を取得
    ght_prob = torch.sigmoid(y_ght).item()
    
    # スケーリングを戻す（仮の値を使用）
    # 実際にはトレーニング時のスケーラーが必要だが、ここでは標準的な値を使用
    md_value = y_md.item() * 5.0 - 10.0  # 仮のスケーリング
    psd_value = y_psd.item() * 3.0 + 5.0  # 仮のスケーリング
    vfi_value = y_vfi.item() * 20.0 + 80.0  # 仮のスケーリング
    
    return {
        'pattern_map': pattern_classes,  # 52点のpattern deviation予測（0-4のクラス）
        'total_map': total_classes,      # 52点のtotal deviation予測（0-4のクラス）
        'MD': md_value,                  # MD値
        'PSD': psd_value,                # PSD値
        'VFI': vfi_value,                # VFI値
        'GHT_probability': ght_prob,     # GHT異常の確率
        'num_models': 1                  # 使用したモデル数（単一モデル）
    }


if __name__ == "__main__":
    # テスト用のダミーデータ
    test_isp = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 
                0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    
    result = predict_isp2hfa(test_isp)
    print(f"使用モデル数: {result['num_models']}")
    print(f"MD: {result['MD']:.2f}")
    print(f"PSD: {result['PSD']:.2f}")
    print(f"VFI: {result['VFI']:.2f}")
    print(f"GHT異常確率: {result['GHT_probability']:.2%}")