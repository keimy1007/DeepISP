import pandas as pd
from sklearn.preprocessing import StandardScaler


def proceprocess_features(isp_data, choose_isp="isp_bin", twice_val=1):
    """
    isp_data: pd.DataFrame
    """
    isp_data = isp_data.copy()

    # 年齢のスケーリング
    isp_data["age"] = isp_data["age"] / 100

    # 特徴量の変換
    for i in range(1, 29):
        column_name = f"{i}_isp"  # デフォルトの列名
        if f"{i}_isp" not in isp_data.columns:
            column_name = f"{i}_{choose_isp}"  # 'X_isp'が存在しない場合はchoose_ispを用いる

        isp_data[f"{i}_isp"] = isp_data[column_name].astype(int)

    for col in isp_data.columns:
        if col.endswith("_isp"):
            isp_data[col] = isp_data[col].map({1: 1, 2: twice_val, 3: 3})

    # 3を0に変換
    isp_data = isp_data.replace(3, 0)

    feature_columns = [f"{i}_isp" for i in range(1, 29)] + ["age"]
    return isp_data[feature_columns]




def preprocess_targets(isp_data, scalers=None, fit_scaler=False):
    y_lists = []
    columns_to_use = [
        [f"{i}_prob_pattern" for i in range(1, 55) if i not in [26, 35]], 
        [f"{i}_prob_total" for i in range(1, 55) if i not in [26, 35]],
        ["MD"], ["PSD"], ["VFI"], ["GHT"]
    ]

    if scalers is None:
        scalers = {col: StandardScaler() for col in ["MD", "PSD", "VFI"]}

    for target_columns in columns_to_use:
        if target_columns in [["MD"], ["PSD"], ["VFI"]]:
            scaler = scalers[target_columns[0]]
            if fit_scaler:
                transformed = scaler.fit_transform(isp_data[target_columns])
            else:
                transformed = scaler.transform(isp_data[target_columns])
            y_lists.append(pd.DataFrame(transformed, columns=target_columns))
        else:
            y_lists.append(isp_data[target_columns])

    return y_lists, scalers
