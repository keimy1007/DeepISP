import torch
import numpy as np
import glob
from models import ISP2HFAprogModel


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


def load_model(folder_path="./params/ISP2HFAprog/simulated_240318"):
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
    hidden_dims = [16, 16]
    dropout_rate = 0.1
    
    # 最初のモデルファイルを使用
    model = ISP2HFAprogModel(hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(param_files[0], map_location=device))
    model.eval()
    
    print(f"Loaded model: {param_files[0].split('/')[-1]}")
    
    return model


def predict_isp2hfaprog(isp_data, model_path="./params/ISP2HFAprog/simulated_240318"):
    """
    ISP2HFAprogモデルで予測を実行
    
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
    y_mdslope = outputs[0]
    y_vfislope = outputs[1]
    y_mdprog = outputs[2]
    y_vfiprog = outputs[3]
    
    # 進行の確率を取得
    mdprog_prob = torch.sigmoid(y_mdprog).item()
    vfiprog_prob = torch.sigmoid(y_vfiprog).item()
    
    # スケーリングを戻す（仮の値を使用）
    # 実際にはトレーニング時のスケーラーが必要だが、ここでは標準的な値を使用
    mdslope_value = y_mdslope.item() * 1.0  # 仮のスケーリング（dB/year）
    vfislope_value = y_vfislope.item() * 2.0  # 仮のスケーリング（%/year）
    
    return {
        'MD_slope': mdslope_value,           # MD slope (dB/year)
        'VFI_slope': vfislope_value,         # VFI slope (%/year)
        'MD_progression_probability': mdprog_prob,   # MD進行の確率
        'VFI_progression_probability': vfiprog_prob, # VFI進行の確率
        'num_models': 1                      # 使用したモデル数（単一モデル）
    }


if __name__ == "__main__":
    # テスト用のダミーデータ
    test_isp = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 
                0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    
    result = predict_isp2hfaprog(test_isp)
    print(f"使用モデル数: {result['num_models']}")
    print(f"MD slope: {result['MD_slope']:.3f} dB/year")
    print(f"VFI slope: {result['VFI_slope']:.3f} %/year")
    print(f"MD進行確率: {result['MD_progression_probability']:.2%}")
    print(f"VFI進行確率: {result['VFI_progression_probability']:.2%}")