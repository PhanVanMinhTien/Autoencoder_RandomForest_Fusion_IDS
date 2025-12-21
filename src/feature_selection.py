# src/feature_selection.py
from mrmr import mrmr_classif
import pandas as pd
from . import config

def run_mrmr(X, y, k=20):
    """
    Chạy thuật toán mRMR để tìm features (Chỉ dùng khi cần tìm mới).
    Nếu đã có list trong config thì không cần gọi hàm này.
    """
    print(f"[mRMR] Selecting top {k} features...")
    # mRMR yêu cầu X là DataFrame
    X_df = pd.DataFrame(X) 
    selected_features = mrmr_classif(X=X_df, y=y, K=k)
    print(f"[mRMR] Selected: {selected_features}")
    return selected_features