# src/preprocessing.py
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from . import config

def normalize_column_names(df):
    """
    Chuẩn hóa tên cột:
    1. Xóa khoảng trắng (Strip)
    2. Đổi tên cột 2018 -> 2017 (dựa trên config)
    3. Xóa các cột trong DROP_COLS (Identifier, Sparse, Duplicate)
    """
    # 1. Strip whitespace
    df.columns = df.columns.str.strip()
    
    # 2. Rename (2018 -> 2017)
    # Bước này cực kỳ quan trọng cho Cross-dataset test
    df = df.rename(columns=config.RENAME_2018_TO_2017)
    
    # 3. Drop Columns (nếu tồn tại)
    cols_to_drop = [c for c in config.DROP_COLS if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        
    return df

def normalize_labels(series, binary_mode=True):
    """
    Chuẩn hóa nhãn từ Raw -> Canonical -> Binary/Multi
    Tham số:
        series: pd.Series chứa nhãn gốc
        binary_mode: True (Default) -> Trả về 0/1. False -> Trả về tên nhãn Canonical.
    """
    # 1. Raw -> Canonical (Chuẩn hóa tên tấn công về nhóm)
    s = series.astype(str).str.strip().str.upper()
    s = s.map(lambda x: config.RAW_TO_CANONICAL.get(x, "OTHER"))
    
    # 2. Canonical -> Binary (hoặc giữ nguyên)
    # Lưu ý: Logic cũ của bạn dựa vào config.BINARY_MODE global. 
    # Ở đây ta ưu tiên tham số truyền vào, nếu không có thì fallback về config (hoặc True mặc định).
    if binary_mode:
        # 0: BENIGN, 1: Attack
        return s.apply(lambda x: 0 if x == "BENIGN" else 1)
    
    return s

def get_scaler():
    return StandardScaler()

# =========================================================================
# HÀM 1: Load dữ liệu mẫu để chạy Feature Selection (mRMR)
# =========================================================================
def load_data_for_mrmr(sample_size=200000):
    """
    Load mẫu ngẫu nhiên từ cả 2 bộ dataset.
    Dùng để chạy thuật toán mRMR.
    """
    files = list(config.DIR_2017.glob("*.csv")) + list(config.DIR_2018.glob("*.csv"))
    if not files: raise FileNotFoundError("Không tìm thấy file CSV!")
    
    samples_per_file = max(1, sample_size // len(files))
    
    # Lấy danh sách cột chuẩn từ 1 file 2017 (để làm mốc validation)
    try:
        sample_2017 = pd.read_csv(next(config.DIR_2017.glob("*.csv")), nrows=1)
        sample_2017 = normalize_column_names(sample_2017)
        valid_cols = [c for c in sample_2017.columns if "Label" not in c and "Class" not in c]
    except Exception as e:
        print(f"⚠️ Không đọc được mẫu 2017: {e}")
        return None, None, None

    X_list = []
    y_list = []
    
    print(f"🚀 Sampling data ({sample_size} rows) for mRMR...")
    
    for f in files:
        try:
            # Đọc chunk lớn hơn cần thiết chút để random
            chunk = pd.read_csv(f, nrows=samples_per_file * 2, low_memory=False)
            
            # 1. Chuẩn hóa tên & Xóa cột rác
            chunk = normalize_column_names(chunk)
            
            # 2. Tìm Label
            label_col = next((c for c in ["Label", "Class", "label", "class"] if c in chunk.columns), None)
            if not label_col: continue
            
            # 3. Chuẩn hóa Label (Mặc định Binary cho mRMR)
            y_chunk = normalize_labels(chunk[label_col], binary_mode=True)
            
            # 4. Ép cột theo chuẩn (Điền 0 nếu thiếu)
            for col in valid_cols:
                if col not in chunk.columns:
                    chunk[col] = 0.0
            
            X_chunk = chunk[valid_cols]
            
            # 5. Lấy mẫu ngẫu nhiên
            if len(X_chunk) > samples_per_file:
                indices = np.random.choice(len(X_chunk), samples_per_file, replace=False)
                X_chunk = X_chunk.iloc[indices]
                y_chunk = y_chunk.iloc[indices]
            
            # Clean NaN
            X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            X_list.append(X_chunk)
            y_list.append(y_chunk)
            
        except Exception as e:
            print(f"⚠️ Error reading {f.name}: {e}")

    X_final = pd.concat(X_list, ignore_index=True)
    y_final = pd.concat(y_list, ignore_index=True)
    
    print(f"✅ Data for mRMR Ready: {X_final.shape}")
    return X_final, y_final, valid_cols

# =========================================================================
# HÀM 2: Load dữ liệu để Train Model (Chunking Mode)
# =========================================================================
def load_mixed_datasets_chunked(binary_mode=True):
    """
    Load toàn bộ dữ liệu (theo chunk) để train.
    Chỉ giữ lại các cột trong config.SELECTED_FEATURES.
    Đảm bảo thứ tự cột cố định cho Autoencoder.
    """
    print(f"🔄 Loading mixed datasets (Chunking Mode, Binary={binary_mode})...")
    files = list(config.DIR_2017.glob("*.csv")) + list(config.DIR_2018.glob("*.csv"))
    
    X_list = []
    y_list = []
    
    for f in files:
        # print(f"Processing {f.name}...") 
        for chunk in pd.read_csv(f, chunksize=config.CHUNK_SIZE, low_memory=False):
            try:
                # 1. Chuẩn hóa tên & Xóa cột rác
                chunk = normalize_column_names(chunk)
                
                # 2. Tìm Label
                label_col = next((c for c in ["Label", "Class", "label", "class"] if c in chunk.columns), None)
                if not label_col: continue
                
                # 3. Chuẩn hóa Label
                y_chunk = normalize_labels(chunk[label_col], binary_mode=binary_mode)
                
                # 4. Lọc Feature & Fill Missing (QUAN TRỌNG)
                # Phải dùng đúng danh sách 67 features trong config để đảm bảo thứ tự
                for col in config.SELECTED_FEATURES:
                    if col not in chunk.columns:
                        chunk[col] = 0.0
                
                # Ép lấy đúng thứ tự cột trong config
                X_chunk = chunk[config.SELECTED_FEATURES]
                
                # Xử lý vô cùng và NaN
                X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                X_list.append(X_chunk)
                y_list.append(y_chunk)
            except Exception as e:
                print(f"⚠️ Error chunk in {f.name}: {e}")
                continue
            
    if not X_list:
        raise ValueError("Không load được dữ liệu nào! Kiểm tra lại đường dẫn.")

    X_final = pd.concat(X_list, ignore_index=True)
    y_final = pd.concat(y_list, ignore_index=True)
    
    print(f"✅ Loaded Mixed Total: {X_final.shape}")
    print(f"   Labels distribution:\n{y_final.value_counts()}")
    
    # Trả về numpy array
    return X_final.values, y_final.values

# =========================================================================
# HÀM 3 (MỚI): Load riêng dataset năm 2017 hoặc 2018
# =========================================================================
def load_single_dataset_year(year, binary_mode=True):
    """
    Load riêng dataset năm 2017 hoặc 2018.
    Áp dụng toàn bộ quy trình chuẩn hóa (rename 2018->2017, filter features)
    để đảm bảo tương thích với model đã train trên tập mixed.
    
    Args:
        year: '2017' hoặc '2018' (string hoặc int)
        binary_mode: True/False
    
    Returns:
        X (DataFrame), y (Series) - Trả về DataFrame để dễ split/analyze
    """
    print(f"🔄 Loading dataset year {year} (Binary={binary_mode})...")
    
    # 1. Xác định file
    if str(year) == '2017':
        files = list(config.DIR_2017.glob("*.csv"))
    elif str(year) == '2018':
        files = list(config.DIR_2018.glob("*.csv"))
    else:
        raise ValueError("Year must be '2017' or '2018'")
        
    if not files:
        raise FileNotFoundError(f"No files found for year {year}")

    X_list = []
    y_list = []
    
    for f in files:
        # print(f"  - Reading {f.name}...")
        try:
            # Dùng chunksize để tránh tràn RAM nếu file lớn
            for chunk in pd.read_csv(f, chunksize=config.CHUNK_SIZE, low_memory=False):
                
                # --- QUY TRÌNH GIỐNG HỆT LOAD_MIXED ---
                
                # 1. Chuẩn hóa tên (bao gồm rename 2018 -> 2017)
                chunk = normalize_column_names(chunk)
                
                # 2. Tìm Label
                label_col = next((c for c in ["Label", "Class", "label", "class"] if c in chunk.columns), None)
                if not label_col: continue
                
                # 3. Chuẩn hóa Label
                y_chunk = normalize_labels(chunk[label_col], binary_mode=binary_mode)
                
                # 4. Lọc Features (Bắt buộc phải khớp config.SELECTED_FEATURES)
                for col in config.SELECTED_FEATURES:
                    if col not in chunk.columns:
                        chunk[col] = 0.0
                
                X_chunk = chunk[config.SELECTED_FEATURES]
                X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                X_list.append(X_chunk)
                y_list.append(y_chunk)
                
        except Exception as e:
            print(f"⚠️ Error reading {f.name}: {e}")

    if not X_list:
        raise ValueError(f"Không load được dữ liệu nào cho năm {year}")

    X_final = pd.concat(X_list, ignore_index=True)
    y_final = pd.concat(y_list, ignore_index=True)
    
    print(f"✅ Loaded {year}. Shape: {X_final.shape}")
    
    # Trả về DataFrame/Series để dễ dàng train_test_split và debug trong notebook
    return X_final, y_final




# Old version deleted below:

# # src/preprocessing.py
# import pandas as pd
# import numpy as np
# import glob
# from sklearn.preprocessing import StandardScaler
# from . import config

# def normalize_column_names(df):
#     """
#     Chuẩn hóa tên cột:
#     1. Xóa khoảng trắng (Strip)
#     2. Đổi tên cột 2018 -> 2017 (dựa trên config)
#     3. Xóa các cột trong DROP_COLS (Identifier, Sparse, Duplicate)
#     """
#     # 1. Strip whitespace
#     df.columns = df.columns.str.strip()
    
#     # 2. Rename (2018 -> 2017)
#     df = df.rename(columns=config.RENAME_2018_TO_2017)
    
#     # 3. Drop Columns (nếu tồn tại)
#     cols_to_drop = [c for c in config.DROP_COLS if c in df.columns]
#     if cols_to_drop:
#         df.drop(columns=cols_to_drop, inplace=True)
        
#     return df

# def normalize_labels(series):
#     """Chuẩn hóa nhãn từ Raw -> Canonical -> Binary/Multi"""
#     s = series.astype(str).str.strip().str.upper()
#     s = s.map(lambda x: config.RAW_TO_CANONICAL.get(x, "OTHER"))
    
#     if config.BINARY_MODE:
#         return s.apply(lambda x: 0 if x == "BENIGN" else 1)
#     return s

# def get_scaler():
#     return StandardScaler()

# # =========================================================================
# # HÀM 1: Load dữ liệu mẫu để chạy Feature Selection (mRMR)
# # =========================================================================
# def load_data_for_mrmr(sample_size=200000):
#     """
#     Load mẫu ngẫu nhiên từ cả 2 bộ dataset, tự động đồng bộ tên cột.
#     Dùng để chạy thuật toán mRMR.
#     """
#     files = list(config.DIR_2017.glob("*.csv")) + list(config.DIR_2018.glob("*.csv"))
#     if not files: raise FileNotFoundError("Không tìm thấy file CSV!")
    
#     samples_per_file = max(1, sample_size // len(files))
    
#     # Lấy danh sách cột chuẩn từ 1 file 2017 (để làm mốc)
#     try:
#         sample_2017 = pd.read_csv(next(config.DIR_2017.glob("*.csv")), nrows=1)
#         sample_2017 = normalize_column_names(sample_2017)
#         # Các cột hợp lệ là cột còn lại sau khi drop và không phải Label
#         valid_cols = [c for c in sample_2017.columns if "Label" not in c and "Class" not in c]
#     except Exception as e:
#         print(f"⚠️ Không đọc được mẫu 2017: {e}")
#         return None, None, None

#     X_list = []
#     y_list = []
    
#     print(f"🚀 Sampling data ({sample_size} rows) for mRMR...")
    
#     for f in files:
#         try:
#             # Đọc 1 chunk nhỏ
#             chunk = pd.read_csv(f, nrows=samples_per_file * 2, low_memory=False)
            
#             # 1. Chuẩn hóa tên & Xóa cột rác
#             chunk = normalize_column_names(chunk)
            
#             # 2. Tìm Label
#             label_col = next((c for c in ["Label", "Class", "label", "class"] if c in chunk.columns), None)
#             if not label_col: continue
            
#             # 3. Chuẩn hóa Label
#             y_chunk = normalize_labels(chunk[label_col])
            
#             # 4. Ép cột theo chuẩn (Điền 0 nếu thiếu)
#             for col in valid_cols:
#                 if col not in chunk.columns:
#                     chunk[col] = 0.0
            
#             X_chunk = chunk[valid_cols]
            
#             # 5. Lấy mẫu ngẫu nhiên
#             if len(X_chunk) > samples_per_file:
#                 indices = np.random.choice(len(X_chunk), samples_per_file, replace=False)
#                 X_chunk = X_chunk.iloc[indices]
#                 y_chunk = y_chunk.iloc[indices]
            
#             # Clean NaN
#             X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
            
#             X_list.append(X_chunk)
#             y_list.append(y_chunk)
            
#         except Exception as e:
#             print(f"⚠️ Error reading {f.name}: {e}")

#     X_final = pd.concat(X_list, ignore_index=True)
#     y_final = pd.concat(y_list, ignore_index=True)
    
#     print(f"✅ Data for mRMR Ready: {X_final.shape}")
#     return X_final, y_final, valid_cols

# # =========================================================================
# # HÀM 2: Load dữ liệu để Train Model (Chunking Mode)
# # =========================================================================
# def load_mixed_datasets_chunked():
#     """
#     Load toàn bộ dữ liệu (theo chunk) để train.
#     Chỉ giữ lại các cột trong config.SELECTED_FEATURES.
#     """
#     print(f"🔄 Loading mixed datasets (Chunking Mode)...")
#     files = list(config.DIR_2017.glob("*.csv")) + list(config.DIR_2018.glob("*.csv"))
    
#     X_list = []
#     y_list = []
    
#     for f in files:
#         # print(f"   📄 Reading: {f.name}...")
#         for chunk in pd.read_csv(f, chunksize=config.CHUNK_SIZE, low_memory=False):
#             # 1. Chuẩn hóa tên & Xóa cột rác
#             chunk = normalize_column_names(chunk)
            
#             # 2. Tìm Label
#             label_col = next((c for c in ["Label", "Class", "label", "class"] if c in chunk.columns), None)
#             if not label_col: continue
            
#             y_chunk = normalize_labels(chunk[label_col])
            
#             # 3. Lọc Feature (Chỉ lấy cột đã chọn từ mRMR)
#             # Nếu thiếu cột (do khác version) -> điền 0
#             for col in config.SELECTED_FEATURES:
#                 if col not in chunk.columns:
#                     chunk[col] = 0.0
            
#             X_chunk = chunk[config.SELECTED_FEATURES]
#             X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
            
#             X_list.append(X_chunk)
#             y_list.append(y_chunk)
            
#     X_final = pd.concat(X_list, ignore_index=True)
#     y_final = pd.concat(y_list, ignore_index=True)
    
#     print(f"✅ Loaded Total: {X_final.shape}")
#     print(f"   Labels:\n{y_final.value_counts()}")
#     return X_final.values, y_final.values


# # =========================================================================
# # HÀM 3: Load riêng dataset năm 2017 hoặc 2018
# # =========================================================================
# def load_single_dataset_year(year, binary_mode=True):
#     """
#     Load riêng dataset năm 2017 hoặc 2018.
#     year: '2017' hoặc '2018'
#     """
#     import glob
#     import os
    
#     # 1. Xác định đường dẫn file
#     if year == '2017':
#         file_pattern = str(config.DATASET_CIC_IDS2017_DIR / "*.csv")
#     elif year == '2018':
#         file_pattern = str(config.DATASET_CSE_CIC_IDS2018_DIR / "*.csv")
#     else:
#         raise ValueError("Year must be '2017' or '2018'")
        
#     csv_files = glob.glob(file_pattern)
#     print(f"📂 Tìm thấy {len(csv_files)} files cho dataset {year}")

#     X_list = []
#     y_list = []
    
#     # 2. Load từng chunk và chuẩn hóa
#     for f in csv_files:
#         print(f"  - Reading {os.path.basename(f)}...")
#         try:
#             for chunk in pd.read_csv(f, chunksize=100000, low_memory=False):
#                 # --- QUAN TRỌNG: CHUẨN HÓA TÊN CỘT (SCHEMA STANDARDIZATION) ---
#                 chunk.columns = chunk.columns.str.strip() # Xóa khoảng trắng
                
#                 # Fix lỗi tên cột cụ thể của 2018 vs 2017 nếu có (ví dụ Timestamp)
#                 # Nhưng quan trọng nhất là strip()
                
#                 # Tìm cột label
#                 label_col = next((c for c in ["Label", "Class", "label", "class"] if c in chunk.columns), None)
#                 if not label_col: continue
                
#                 # Xử lý nhãn
#                 y_chunk, _ = normalize_labels(chunk[label_col], binary_mode=binary_mode)
                
#                 # Xử lý features (Bỏ cột rác + cột định danh)
#                 X_chunk = process_and_clean_data(chunk.drop(columns=[label_col]))
                
#                 # --- LỌC BỚT ĐỂ TRÁNH TRÀN RAM (Optional) ---
#                 # Nếu chỉ test code, có thể lấy sample. Nếu chạy thật, comment dòng dưới.
#                 # chunk = chunk.sample(frac=0.1) 

#                 X_list.append(X_chunk)
#                 y_list.append(y_chunk)
                
#         except Exception as e:
#             print(f"⚠️ Lỗi đọc file {f}: {e}")

#     # 3. Gộp lại
#     if not X_list:
#         raise ValueError(f"Không load được dữ liệu nào cho năm {year}")

#     X = pd.concat(X_list, ignore_index=True)
#     y = pd.concat(y_list, ignore_index=True)
    
#     # Đảm bảo fillna lần cuối
#     X = X.fillna(0)
    
#     print(f"✅ Load xong {year}. Shape: {X.shape}")
#     return X, y