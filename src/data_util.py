import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

import torch
from torch.utils.data import DataLoader, TensorDataset

############# 乱数の固定 #############
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)


############# データのカウント #############

def count_samples(df):
    # 全体のサンプル数
    cnt_vf = len(df)
    
    # ユニークな目（IDとlateralityのペア）の数
    cnt_eye = df.drop_duplicates(subset=['ID', 'laterality']).shape[0]
    
    # ユニークな患者（ID）の数
    cnt_patient = df['ID'].nunique()
    
    return cnt_vf, cnt_eye, cnt_patient

############# データローダーの作成 #############

def create_dataloader(X, Y_lists, batch_size=64):
    dataset = TensorDataset(X, *Y_lists)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            worker_init_fn=seed_worker, 
                            generator=g)
    return dataloader


def create_cv_dataloader(X_real_tensor, y_real_tensors, n_splits=3, batch_size=64):
    # kf = KFold(n_splits=n_splits, shuffle = True, random_state=42)
    kf = KFold(n_splits=n_splits, shuffle = True, random_state=42)
    X_train_CV = []
    X_test_CV = []
    y_train_CV = []
    y_test_CV = []
    train_loader_CV = []
    test_loader_CV = []

    for train_index, test_index in kf.split(X_real_tensor):
        # データの分割
        X_train, X_test = X_real_tensor[train_index], X_real_tensor[test_index]
        X_train_CV.append(X_train)
        X_test_CV.append(X_test)

        y_train_tensors, y_test_tensors = [], []
        for tensor in y_real_tensors:
            y_train, y_test = tensor[train_index], tensor[test_index]
            y_train_tensors.append(y_train)
            y_test_tensors.append(y_test)
        y_train_CV.append(y_train_tensors)
        y_test_CV.append(y_test_tensors)

        # データローダーの作成
        train_loader = create_dataloader(X_train, y_train_tensors, batch_size=batch_size)
        test_loader = create_dataloader(X_test, y_test_tensors, batch_size=batch_size)

        train_loader_CV.append(train_loader)
        test_loader_CV.append(test_loader)

    return X_train_CV, X_test_CV, y_train_CV, y_test_CV, train_loader_CV, test_loader_CV



def create_loocv_dataloader(X_real_tensor, y_real_tensors, batch_size=64):
    loo = LeaveOneOut()
    X_train_CV = []
    X_test_CV = []
    y_train_CV = []
    y_test_CV = []
    train_loader_CV = []
    test_loader_CV = []

    for train_index, test_index in loo.split(X_real_tensor):
        # データの分割
        X_train, X_test = X_real_tensor[train_index], X_real_tensor[test_index]
        X_train_CV.append(X_train)
        X_test_CV.append(X_test)

        y_train_tensors, y_test_tensors = [], []
        for tensor in y_real_tensors:
            y_train, y_test = tensor[train_index], tensor[test_index]
            y_train_tensors.append(y_train)
            y_test_tensors.append(y_test)
        y_train_CV.append(y_train_tensors)
        y_test_CV.append(y_test_tensors)

        # データローダーの作成
        train_loader = create_dataloader(X_train, y_train_tensors, batch_size=batch_size)
        test_loader = create_dataloader(X_test, y_test_tensors, batch_size=batch_size)

        train_loader_CV.append(train_loader)
        test_loader_CV.append(test_loader)

    return X_train_CV, X_test_CV, y_train_CV, y_test_CV, train_loader_CV, test_loader_CV
